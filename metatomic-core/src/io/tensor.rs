use byteorder::{BigEndian, LittleEndian, NativeEndian, ReadBytesExt, WriteBytesExt};

use dlpk::{DLDataType, DLDataTypeCode, DLDevice, DLPackTensor, DLPackTensorRef, DLPackVersion};
use metatensor::MtsArray;
use metatensor::c_api::{MTS_SUCCESS, mts_array_t, mts_create_array_callback_t};

use crate::Error;

use super::{Endianness, check_for_extra_bytes};
use super::npy_header::{Header, DataType};

/// Parse an NPY type descriptor string (e.g. `"<f8"`) into a DLPack data type code,
/// bit width, and byte order.
fn npy_descr_to_dtype(descr: &str) -> Result<(DLDataTypeCode, u8, Endianness), Error> {
    if descr.len() < 3 {
        return Err(Error::Serialization(format!("invalid type descriptor: {}", descr)));
    }

    let endian = match &descr[0..1] {
        "<" => Endianness::Little,
        "=" | "|"  => Endianness::Native,
        ">" => Endianness::Big,
        // not applicable for single-byte types
        _ => return Err(Error::Serialization(format!("unknown endianness in type descriptor: {}", descr))),
    };

    let type_char = &descr[1..2];
    let size_str = &descr[2..];
    let size: u8 = size_str.parse().map_err(|_| {
        Error::Serialization(format!("invalid size in type descriptor: {}", descr))
    })?;

    let (code, bits) = match (type_char, size) {
        ("f", 4) => (DLDataTypeCode::kDLFloat, 32),
        ("f", 8) => (DLDataTypeCode::kDLFloat, 64),
        ("i", 1) => (DLDataTypeCode::kDLInt, 8),
        ("i", 2) => (DLDataTypeCode::kDLInt, 16),
        ("i", 4) => (DLDataTypeCode::kDLInt, 32),
        ("i", 8) => (DLDataTypeCode::kDLInt, 64),
        ("u", 1) => (DLDataTypeCode::kDLUInt, 8),
        ("u", 2) => (DLDataTypeCode::kDLUInt, 16),
        ("u", 4) => (DLDataTypeCode::kDLUInt, 32),
        ("u", 8) => (DLDataTypeCode::kDLUInt, 64),
        ("b", 1) => (DLDataTypeCode::kDLBool, 8),
        ("c", 8) => (DLDataTypeCode::kDLComplex, 64),
        ("c", 16) => (DLDataTypeCode::kDLComplex, 128),
        ("f", 2) => (DLDataTypeCode::kDLFloat, 16),
        _ => return Err(Error::Serialization(format!("unsupported type descriptor: {}", descr))),
    };

    Ok((code, bits, endian))
}


fn read_as<T, R>(reader: &mut R, tensor: dlpk::DLPackTensorRefMut<'_>, cb: impl Fn(&mut R, &mut T) -> Result<(), std::io::Error>) -> Result<(), Error>
where R: std::io::Read,
      T: dlpk::DLPackPointerCast + 'static
{
    let mut view: ndarray::ArrayViewMutD<T> = tensor.try_into()
        .map_err(|e| Error::Serialization(format!("failed to convert DLPack to ndarray mutable view: {}", e)))?;

    for value in &mut view {
        cb(reader, value)?;
    }

    Ok(())
}

// Read a data array from the given reader, using numpy's NPY format
#[allow(clippy::too_many_lines)]
pub fn read_tensor<R>(mut reader: R, create_array: mts_create_array_callback_t) -> Result<DLPackTensor, Error>
    where R: std::io::Read
{
    let create_array = create_array.ok_or_else(|| Error::InvalidParameter("create_array callback is NULL".into()))?;
    let header = super::npy_header::Header::from_reader(&mut reader)?;
    if header.fortran_order {
        return Err(Error::Serialization("data can not be loaded from fortran-order arrays".into()));
    }

    let descr = if let super::npy_header::DataType::Scalar(s) = &header.type_descriptor {
        s.as_str()
    } else {
        return Err(Error::Serialization("structured arrays are not supported".into()));
    };

    let (file_code, file_bits, endian) = npy_descr_to_dtype(descr)?;

    let dl_dtype = DLDataType { code: file_code, bits: file_bits, lanes: 1 };

    let shape = header.shape;
    let mut array = mts_array_t::null();
    let status = unsafe {
        create_array(shape.as_ptr(), shape.len(), dl_dtype, &mut array)
    };

    let array = if status == MTS_SUCCESS {
        MtsArray::from_raw(array)
    } else {
        // TODO: how can we propagate the error from the callback?
        return Err(Error::Serialization("failed to create array".into()));
    };

    let device = DLDevice::cpu();
    let version = DLPackVersion::current();
    let mut dl_tensor = array.as_dlpack(device, None, version)?;

    let num_elements: usize = shape.iter().product();
    if num_elements == 0 {
        check_for_extra_bytes(&mut reader)?;
        return Ok(dl_tensor);
    }

    let tensor = dl_tensor.as_mut();

    // Endianness is handled inside each arm to avoid tripling the number of
    // match arms (which inflates uncovered-line counts for big/native paths
    // that are not exercised in tests on little-endian CI).
    match (file_code, file_bits) {
        // Standard Floats
        (DLDataTypeCode::kDLFloat, 32) => read_as::<f32, _>(&mut reader, tensor, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => r.read_f32::<LittleEndian>()?,
                Endianness::Big => r.read_f32::<BigEndian>()?,
                Endianness::Native => r.read_f32::<NativeEndian>()?,
            };
            Ok(())
        }),
        (DLDataTypeCode::kDLFloat, 64) => read_as::<f64, _>(&mut reader, tensor, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => r.read_f64::<LittleEndian>()?,
                Endianness::Big => r.read_f64::<BigEndian>()?,
                Endianness::Native => r.read_f64::<NativeEndian>()?,
            };
            Ok(())
        }),

        // Standard Ints
        (DLDataTypeCode::kDLInt, 8) => read_as::<i8, _>(&mut reader, tensor, |r: &mut R, v| {
            *v = r.read_i8()?;
            Ok(())
        }),
        (DLDataTypeCode::kDLInt, 16) => read_as::<i16, _>(&mut reader, tensor, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => r.read_i16::<LittleEndian>()?,
                Endianness::Big => r.read_i16::<BigEndian>()?,
                Endianness::Native => r.read_i16::<NativeEndian>()?,
            };
            Ok(())
        }),
        (DLDataTypeCode::kDLInt, 32) => read_as::<i32, _>(&mut reader, tensor, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => r.read_i32::<LittleEndian>()?,
                Endianness::Big => r.read_i32::<BigEndian>()?,
                Endianness::Native => r.read_i32::<NativeEndian>()?,
            };
            Ok(())
        }),
        (DLDataTypeCode::kDLInt, 64) => read_as::<i64, _>(&mut reader, tensor, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => r.read_i64::<LittleEndian>()?,
                Endianness::Big => r.read_i64::<BigEndian>()?,
                Endianness::Native => r.read_i64::<NativeEndian>()?,
            };
            Ok(())
        }),

        // Unsigned Ints
        (DLDataTypeCode::kDLUInt, 8) => read_as::<u8, _>(&mut reader, tensor, |r: &mut R, v| {
            *v = r.read_u8()?;
            Ok(())
        }),
        (DLDataTypeCode::kDLUInt, 16) => read_as::<u16, _>(&mut reader, tensor, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => r.read_u16::<LittleEndian>()?,
                Endianness::Big => r.read_u16::<BigEndian>()?,
                Endianness::Native => r.read_u16::<NativeEndian>()?,
            };
            Ok(())
        }),
        (DLDataTypeCode::kDLUInt, 32) => read_as::<u32, _>(&mut reader, tensor, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => r.read_u32::<LittleEndian>()?,
                Endianness::Big => r.read_u32::<BigEndian>()?,
                Endianness::Native => r.read_u32::<NativeEndian>()?,
            };
            Ok(())
        }),
        (DLDataTypeCode::kDLUInt, 64) => read_as::<u64, _>(&mut reader, tensor, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => r.read_u64::<LittleEndian>()?,
                Endianness::Big => r.read_u64::<BigEndian>()?,
                Endianness::Native => r.read_u64::<NativeEndian>()?,
            };
            Ok(())
        }),

        // Boolean (Read as u8)
        (DLDataTypeCode::kDLBool, 8) => read_as::<bool, _>(&mut reader, tensor, |r: &mut R, v| {
            *v = r.read_u8()? != 0;
            Ok(())
        }),

        // Complex Numbers (Read as array of 2 floats)
        (DLDataTypeCode::kDLComplex, 64) => read_as::<[f32; 2], _>(&mut reader, tensor, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => [r.read_f32::<LittleEndian>()?, r.read_f32::<LittleEndian>()?],
                Endianness::Big => [r.read_f32::<BigEndian>()?, r.read_f32::<BigEndian>()?],
                Endianness::Native => [r.read_f32::<NativeEndian>()?, r.read_f32::<NativeEndian>()?],
            };
            Ok(())
        }),
        (DLDataTypeCode::kDLComplex, 128) => read_as::<[f64; 2], _>(&mut reader, tensor, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => [r.read_f64::<LittleEndian>()?, r.read_f64::<LittleEndian>()?],
                Endianness::Big => [r.read_f64::<BigEndian>()?, r.read_f64::<BigEndian>()?],
                Endianness::Native => [r.read_f64::<NativeEndian>()?, r.read_f64::<NativeEndian>()?],
            };
            Ok(())
        }),

        _ => Err(Error::Serialization(format!(
            "unsupported dtype for reading: {:?} {} bits", file_code, file_bits
        ))),
    }?;

    check_for_extra_bytes(&mut reader)?;
    Ok(dl_tensor)
}

fn dlpack_to_npy_descr(code: DLDataTypeCode, bits: u8) -> Result<String, Error> {
    let endian = if cfg!(target_endian = "little") { "<" } else { ">" };

    let (type_char, type_size) = match (code, bits) {
        (DLDataTypeCode::kDLInt, 8) => ("i", 1),
        (DLDataTypeCode::kDLInt, 16) => ("i", 2),
        (DLDataTypeCode::kDLInt, 32) => ("i", 4),
        (DLDataTypeCode::kDLInt, 64) => ("i", 8),
        (DLDataTypeCode::kDLUInt, 8) => ("u", 1),
        (DLDataTypeCode::kDLUInt, 16) => ("u", 2),
        (DLDataTypeCode::kDLUInt, 32) => ("u", 4),
        (DLDataTypeCode::kDLUInt, 64) => ("u", 8),
        (DLDataTypeCode::kDLFloat, 32) => ("f", 4),
        (DLDataTypeCode::kDLFloat, 64) => ("f", 8),
        (DLDataTypeCode::kDLBool, 8) => ("b", 1),
        (DLDataTypeCode::kDLComplex, 64) => ("c", 8),
        (DLDataTypeCode::kDLComplex, 128) => ("c", 16),
        (DLDataTypeCode::kDLFloat, 16) => ("f", 2),
        _ => return Err(Error::Serialization(
            format!("unsupported DLPack dtype: code {:?}, bits {:?}", code, bits)
                                            )
        ),
    };

    Ok(format!("{}{}{}", endian, type_char, type_size))
}


fn write_as<T, W>(writer: &mut W, tensor: dlpk::DLPackTensorRef<'_>, cb: impl Fn(&mut W, T) -> Result<(), std::io::Error>) -> Result<(), Error>
where W: std::io::Write,
      T: Copy + dlpk::DLPackPointerCast + 'static
{
    let view: ndarray::ArrayViewD<T> = tensor.try_into()
        .map_err(|e| Error::Serialization(format!("failed to convert DLPack to ndarray view: {}", e)))?;

    for &value in &view {
        cb(writer, value)?;
    }

    Ok(())
}

// Write an array to the given writer, using numpy's NPY format
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn write_tensor<W: std::io::Write>(writer: &mut W, tensor: DLPackTensorRef<'_>) -> Result<(), Error> {
    let dtype = tensor.raw.dtype;
    let (code, bits) = (dtype.code, dtype.bits);

    // Validate Lanes
    if dtype.lanes != 1 {
        return Err(Error::Serialization(format!(
            "unsupported DLPack dtype: lanes != 1 ({})", dtype.lanes
        )));
    }

    // Write Header
    let tdesc = dlpack_to_npy_descr(code, bits)?;
    let header = Header {
        type_descriptor: DataType::Scalar(tdesc),
        fortran_order: false,
        shape: tensor.shape().iter().map(|&s| s as usize).collect(),
    };

    header.write(&mut *writer)?;

    // Get metadata for size and pointer for data
    let num_elements: usize = header.shape.iter().product();
    if num_elements == 0 {
        return Ok(());
    }

    match (code, bits) {
        // Standard Floats
        (DLDataTypeCode::kDLFloat, 32) => write_as::<f32, _>(writer, tensor, |w: &mut W, v| w.write_f32::<NativeEndian>(v)),
        (DLDataTypeCode::kDLFloat, 64) => write_as::<f64, _>(writer, tensor, |w: &mut W, v| w.write_f64::<NativeEndian>(v)),

        // Standard Ints
        (DLDataTypeCode::kDLInt, 8) => write_as::<i8, _>(writer, tensor, |w: &mut W, v| w.write_i8(v)),
        (DLDataTypeCode::kDLInt, 16) => write_as::<i16, _>(writer, tensor, |w: &mut W, v| w.write_i16::<NativeEndian>(v)),
        (DLDataTypeCode::kDLInt, 32) => write_as::<i32, _>(writer, tensor, |w: &mut W, v| w.write_i32::<NativeEndian>(v)),
        (DLDataTypeCode::kDLInt, 64) => write_as::<i64, _>(writer, tensor, |w: &mut W, v| w.write_i64::<NativeEndian>(v)),

        // Unsigned Ints
        (DLDataTypeCode::kDLUInt, 8) => write_as::<u8, _>(writer, tensor, |w: &mut W, v| w.write_u8(v)),
        (DLDataTypeCode::kDLUInt, 16) => write_as::<u16, _>(writer, tensor, |w: &mut W, v| w.write_u16::<NativeEndian>(v)),
        (DLDataTypeCode::kDLUInt, 32) => write_as::<u32, _>(writer, tensor, |w: &mut W, v| w.write_u32::<NativeEndian>(v)),
        (DLDataTypeCode::kDLUInt, 64) => write_as::<u64, _>(writer, tensor, |w: &mut W, v| w.write_u64::<NativeEndian>(v)),

        // Boolean, stored as u8
        (DLDataTypeCode::kDLBool, 8) => write_as::<bool, _>(writer, tensor, |w: &mut W, v| w.write_u8(u8::from(v))),

        // Complex Numbers
        (DLDataTypeCode::kDLComplex, 64) => write_as::<[f32; 2], _>(writer, tensor, |w: &mut W, v| {
            w.write_f32::<NativeEndian>(v[0])?;
            w.write_f32::<NativeEndian>(v[1])
        }),
        (DLDataTypeCode::kDLComplex, 128) => write_as::<[f64; 2], _>(writer, tensor, |w: &mut W, v| {
            w.write_f64::<NativeEndian>(v[0])?;
            w.write_f64::<NativeEndian>(v[1])
        }),

        _ => Err(Error::Serialization(format!("unsupported dtype for writing: {:?} {} bits", code, bits))),
    }
}
