use std::collections::BTreeMap;

use json::JsonValue;

use crate::{Error, Quantity};
use crate::units::validate_unit;

/// Options for the calculation of a pair list (neighbor list)
#[derive(Debug, Clone)]
pub struct PairListOptions {
    /// Cutoff radius for this pair list in the length unit of the model
    pub cutoff: f64,
    /// Whether the list is a full list (contains both the pair `i -> j` and `j -> i`)
    /// or a half list (contains only `i -> j`)
    pub full_list: bool,
    /// Whether the list guarantees that only atoms within the cutoff are
    /// included (strict) or may also include pairs slightly beyond the cutoff
    /// (non-strict)
    pub strict: bool,
    /// List of strings describing who requested this pair list
    pub requestors: Vec<String>,
}

impl std::cmp::PartialEq for PairListOptions {
    fn eq(&self, other: &Self) -> bool {
        self.cutoff == other.cutoff
            && self.full_list == other.full_list
            && self.strict == other.strict
    }
}

impl std::cmp::Eq for PairListOptions {}

impl std::cmp::PartialOrd for PairListOptions {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Ord for PairListOptions {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cutoff.partial_cmp(&other.cutoff).expect("cutoff is NaN")
            .then_with(|| self.full_list.cmp(&other.full_list))
            .then_with(|| self.strict.cmp(&other.strict))
    }
}

impl From<PairListOptions> for JsonValue {
    fn from(value: PairListOptions) -> Self {
        let mut result = JsonValue::new_object();
        result["type"] = "metatomic_pair_options".into();
        // store the bit pattern so the float round-trips exactly
        result["cutoff"] = format!("{:#x}", value.cutoff.to_bits()).into();
        result["full_list"] = value.full_list.into();
        result["strict"] = value.strict.into();
        result["requestors"] = value.requestors.into();
        return result;
    }
}

impl<'a> TryFrom<&'a JsonValue> for PairListOptions {
    type Error = Error;

    fn try_from(value: &'a JsonValue) -> Result<Self, Self::Error> {
        if !value.is_object() {
            return Err(Error::Serialization(
                "invalid JSON data for PairListOptions, expected an object".into()
            ));
        }

        if value["type"].as_str() != Some("metatomic_pair_options") {
            return Err(Error::Serialization(
                "'type' in JSON for PairListOptions must be 'metatomic_pair_options'".into()
            ));
        }

        let cutoff = value["cutoff"].as_str().ok_or_else(|| Error::Serialization(
            "'cutoff' in JSON for PairListOptions must be a hex-encoded string".into()
        ))?;
        let bits = u64::from_str_radix(cutoff.strip_prefix("0x").unwrap_or(cutoff), 16)
            .map_err(|_| Error::Serialization(
                "'cutoff' in JSON for PairListOptions must be a hex-encoded string".into()
            ))?;
        let cutoff = f64::from_bits(bits);

        if !cutoff.is_finite() || cutoff <= 0.0 {
            return Err(Error::Serialization(
                "'cutoff' in JSON for PairListOptions must be a finite positive number".into()
            ));
        }

        let full_list = value["full_list"].as_bool().ok_or_else(|| Error::Serialization(
            "'full_list' in JSON for PairListOptions must be a boolean".into()
        ))?;

        let strict = value["strict"].as_bool().ok_or_else(|| Error::Serialization(
            "'strict' in JSON for PairListOptions must be a boolean".into()
        ))?;

        let mut requestors = Vec::new();
        if value.has_key("requestors") {
            if !value["requestors"].is_array() {
                return Err(Error::Serialization(
                    "'requestors' in JSON for PairListOptions must be an array".into()
                ));
            }

            for requestor in value["requestors"].members() {
                let requestor = requestor.as_str().ok_or_else(|| Error::Serialization(
                    "'requestors' in JSON for PairListOptions must be an array of strings".into()
                ))?;
                // ignore empty strings and duplicates, keeping first-seen order
                if !requestor.is_empty() && !requestors.iter().any(|r| r == requestor) {
                    requestors.push(requestor.to_string());
                }
            }
        }

        return Ok(PairListOptions { cutoff, full_list, strict, requestors });
    }
}

// ========================================================================== //
// ========================================================================== //
// ========================================================================== //

/// References for a model, divided into three categories: references about the
/// model as a whole, references about the architecture of the model, and
/// references about the implementation of the model. Each category is a list of
/// strings, which can be DOIs, URLs, or any other format the model author finds
/// useful.
#[derive(Debug, Clone)]
pub struct References {
    /// The references about the model as a whole, e.g. a paper describing the
    /// model or a website presenting it.
    model: Vec<String>,
    /// The references about the architecture of the model, e.g. papers
    /// describing the mathematical form of the model.
    architecture: Vec<String>,
    /// The references about the implementation of the model, e.g. a link to
    /// the source code repository or a paper describing the software.
    implementation: Vec<String>,
}

impl From<References> for JsonValue {
    fn from(value: References) -> Self {
        let mut result = JsonValue::new_object();
        result["model"] = value.model.into();
        result["architecture"] = value.architecture.into();
        result["implementation"] = value.implementation.into();
        return result;
    }
}


fn read_references(object: &JsonValue, key: &str) -> Result<Vec<String>, Error> {
    let mut references = Vec::new();
    if !object[key].is_array() {
        return Err(Error::Serialization(
            format!("'{}' in references of ModelMetadata must be an array", key)
        ));
    }
    for reference in object[key].members() {
        let reference = reference.as_str().ok_or_else(|| Error::Serialization(
            format!("'{}' in references of ModelMetadata must be an array of strings", key)
        ))?;
        references.push(reference.to_string());
    }
    Ok(references)
}

impl<'a> TryFrom<&'a JsonValue> for References {
    type Error = Error;

    fn try_from(value: &'a JsonValue) -> Result<Self, Self::Error> {
        if !value.is_object() {
            return Err(Error::Serialization(
                "invalid JSON data for references in ModelMetadata, expected an object".into()
            ));
        }

        let model = read_references(value, "model")?;
        let architecture = read_references(value, "architecture")?;
        let implementation = read_references(value, "implementation")?;

        Ok(References { model, architecture, implementation })
    }
}


/// Metadata about a model
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// The name of the model, e.g. `"MyCoolModel v1.2"`
    pub name: String,
    /// The authors of the model, e.g. `["Alice Smith", "Bob Johnson
    /// <bobj@example.com>"]`
    pub authors: Vec<String>,
    /// A description of the model
    pub description: String,
    /// References for the model that should be cited when using it
    pub references: References,
    /// Any other key-value pairs the model author wants to include in the
    /// metadata. This can be used for any purpose.
    pub extra: BTreeMap<String, String>,
}

impl From<ModelMetadata> for JsonValue {
    fn from(value: ModelMetadata) -> Self {
        let mut result = JsonValue::new_object();
        result["type"] = "metatomic_model_metadata".into();
        result["name"] = value.name.into();
        result["authors"] = value.authors.into();
        result["description"] = value.description.into();
        result["references"] = value.references.into();
        result["extra"] = value.extra.into();
        return result;
    }
}

impl<'a> TryFrom<&'a JsonValue> for ModelMetadata {
    type Error = Error;

    fn try_from(value: &'a JsonValue) -> Result<Self, Self::Error> {
        if !value.is_object() {
            return Err(Error::Serialization(
                "invalid JSON data for ModelMetadata, expected an object".into()
            ));
        }

        if value["type"].as_str() != Some("metatomic_model_metadata") {
            return Err(Error::Serialization(
                "'type' in JSON for ModelMetadata must be 'metatomic_model_metadata'".into()
            ));
        }

        let name = value["name"].as_str().ok_or_else(|| Error::Serialization(
            "'name' in JSON for ModelMetadata must be a string".into()
        ))?;

        if !value["authors"].is_array() {
            return Err(Error::Serialization(
                "'authors' in JSON for ModelMetadata must be an array".into()
            ));
        }

        let authors = value["authors"].members().map(|author| {
            author.as_str().ok_or_else(|| Error::Serialization(
                "'authors' in JSON for ModelMetadata must be an array of strings".into()
            )).map(|s| s.to_string())
        }).collect::<Result<Vec<String>, Error>>()?;

        let description = value["description"].as_str().ok_or_else(|| Error::Serialization(
            "'description' in JSON for ModelMetadata must be a string".into()
        ))?.to_string();

        let references = References::try_from(&value["references"])?;

        if !value["extra"].is_object() {
            return Err(Error::Serialization(
                "'extra' in JSON for ModelMetadata must be an object".into()
            ));
        }

        let mut extra = BTreeMap::new();
        for (key, value) in value["extra"].entries() {
            let value = value.as_str().ok_or_else(|| Error::Serialization(
                "'extra' in JSON for ModelMetadata must be an object with string values".into()
            ))?;
            extra.insert(key.to_string(), value.to_string());
        }

        Ok(ModelMetadata {
            name: name.to_string(),
            authors: authors,
            description: description,
            references: references,
            extra: extra,
        })
    }
}

/// The data type of a model, used for all inputs and outputs. The model can
/// still internally use a different data type for its calculations, but it will
/// get inputs in this type and must produce outputs in this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit floating point, following the IEEE 754 standard
    Float32,
    /// 64-bit floating point, following the IEEE 754 standard
    Float64,
}

impl From<DType> for JsonValue {
    fn from(value: DType) -> Self {
        match value {
            DType::Float32 => "float32".into(),
            DType::Float64 => "float64".into(),
        }
    }
}

impl<'a> TryFrom<&'a JsonValue> for DType {
    type Error = Error;

    fn try_from(value: &'a JsonValue) -> Result<Self, Self::Error> {
        if let Some(s) = value.as_str() {
            match s {
                "float32" => Ok(DType::Float32),
                "float64" => Ok(DType::Float64),
                _ => Err(Error::Serialization(
                    "invalid string for dtype in JSON for ModelCapabilities, expected 'float32' or 'float64'".into()
                )),
            }
        } else {
            Err(Error::Serialization(
                "dtype in JSON for ModelCapabilities must be a string".into()
            ))
        }
    }
}

/// A device on which a model can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Device(dlpk::DLDeviceType);

impl From<Device> for JsonValue {
    fn from(value: Device) -> Self {
        match value.0 {
            dlpk::DLDeviceType::kDLCPU => "cpu".into(),
            dlpk::DLDeviceType::kDLCUDA => "cuda".into(),
            dlpk::DLDeviceType::kDLROCM => "rocm".into(),
            dlpk::DLDeviceType::kDLMetal => "metal".into(),
            dlpk::DLDeviceType::kDLCUDAHost | dlpk::DLDeviceType::kDLCUDAManaged => {
                // These refer to memory devices more than execution devices
                panic!("Do not use kDLCUDAHost or kDLCUDAManaged, use kDLCUDA instead.");
            }
            dlpk::DLDeviceType::kDLROCMHost => {
                // This refers to a memory device more than an execution device
                panic!("Do not use kDLROCMHost, use kDLROCM instead.");
            }
            _ => {
                // We don't want to expose other device types until we have a
                // use case for them, and we don't want to accidentally leak
                // them if they're added in the future
                panic!("unsupported device type: {:?}", value.0);
            }
        }
    }
}

impl<'a> TryFrom<&'a JsonValue> for Device {
    type Error = Error;

    fn try_from(value: &'a JsonValue) -> Result<Self, Self::Error> {
        if let Some(s) = value.as_str() {
            match s {
                "cpu" => Ok(Device(dlpk::DLDeviceType::kDLCPU)),
                "cuda" => Ok(Device(dlpk::DLDeviceType::kDLCUDA)),
                "rocm" => Ok(Device(dlpk::DLDeviceType::kDLROCM)),
                "metal" => Ok(Device(dlpk::DLDeviceType::kDLMetal)),
                _ => Err(Error::Serialization(
                    "invalid string for device in JSON for ModelCapabilities, expected 'cpu', 'cuda', 'rocm', or 'metal'".into()
                )),
            }
        } else {
            Err(Error::Serialization(
                "device in JSON for ModelCapabilities must be a string".into()
            ))
        }
    }
}

/// Capabilities about a model: which outputs it provides, which atoms it
/// supports, etc.
#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    /// The outputs this model can provide
    pub outputs: Vec<Quantity>,
    /// The atomic types this model supports. The meaning of the integers in
    /// this list is up to the model, and is not required to be the atomic
    /// numbers.
    pub atomic_types: Vec<i64>,
    /// The interaction range of the model (in the length unit of the model),
    /// i.e. the maximum distance between two atoms for which the model's output
    /// can depend on their relative position.
    pub interaction_range: f64,
    /// The length unit of the model, e.g. "angstrom" or "nanometer". This is
    /// used to interpret the `interaction_range` and convert the inputs.
    pub length_unit: String,
    /// The devices on which the model can run, e.g. `["cpu", "cuda"]`.
    pub supported_devices: Vec<Device>,
    /// The data type of the model, used for all inputs and outputs.
    pub dtype: DType,
}

impl From<ModelCapabilities> for JsonValue {
    fn from(value: ModelCapabilities) -> Self {
        let mut result = JsonValue::new_object();
        result["type"] = "metatomic_model_capabilities".into();
        result["outputs"] = value.outputs.into();
        result["atomic_types"] = value.atomic_types.into();
        result["interaction_range"] = value.interaction_range.into();
        result["length_unit"] = value.length_unit.into();
        result["supported_devices"] = value.supported_devices.into();
        result["dtype"] = value.dtype.into();
        return result;
    }
}

impl<'a> TryFrom<&'a JsonValue> for ModelCapabilities {
    type Error = Error;

    fn try_from(value: &'a JsonValue) -> Result<Self, Self::Error> {
        if !value.is_object() {
            return Err(Error::Serialization(
                "invalid JSON data for ModelCapabilities, expected an object".into()
            ));
        }

        if value["type"].as_str() != Some("metatomic_model_capabilities") {
            return Err(Error::Serialization(
                "'type' in JSON for ModelCapabilities must be 'metatomic_model_capabilities'".into()
            ));
        }

        let mut outputs = Vec::new();
        if !value["outputs"].is_array() {
            return Err(Error::Serialization(
                "'outputs' in JSON for ModelCapabilities must be an array".into()
            ));
        }
        for output in value["outputs"].members() {
            outputs.push(Quantity::try_from(output)?);
        }


        let mut atomic_types = Vec::new();
        if !value["atomic_types"].is_array() {
            return Err(Error::Serialization(
                "'atomic_types' in JSON for ModelCapabilities must be an array".into()
            ));
        }

        for atomic_type in value["atomic_types"].members() {
            let atomic_type = atomic_type.as_i64().ok_or_else(|| Error::Serialization(
                "'atomic_types' in JSON for ModelCapabilities must be an array of integers".into()
            ))?;
            atomic_types.push(atomic_type);
        }

        let interaction_range = value["interaction_range"].as_f64().ok_or_else(|| Error::Serialization(
            "'interaction_range' in JSON for ModelCapabilities must be a number".into()
        ))?;
        if interaction_range < 0.0 {
            return Err(Error::Serialization(
                "'interaction_range' in JSON for ModelCapabilities must be non-negative".into()
            ));
        }

        let length_unit = value["length_unit"].as_str().ok_or_else(|| Error::Serialization(
            "'length_unit' in JSON for ModelCapabilities must be a string".into()
        ))?.to_string();
        validate_unit(&length_unit, "m", Some("'length_unit' in JSON for ModelCapabilities"))?;

        let mut supported_devices = Vec::new();
        if !value["supported_devices"].is_array() {
            return Err(Error::Serialization(
                "'supported_devices' in JSON for ModelCapabilities must be an array".into()
            ));
        }
        for device in value["supported_devices"].members() {
            supported_devices.push(Device::try_from(device)?);
        }

        let dtype = DType::try_from(&value["dtype"])?;

        Ok(ModelCapabilities {
            outputs,
            atomic_types,
            interaction_range,
            length_unit,
            supported_devices,
            dtype,
        })
    }
}

#[cfg(test)]
mod tests {
    mod pair_list_options {
        use super::super::*;

        fn example() -> PairListOptions {
            PairListOptions {
                cutoff: 3.5,
                full_list: true,
                strict: false,
                requestors: vec!["nl-1".to_string(), "nl-2".to_string()],
            }
        }

        #[test]
        fn roundtrip() {
            let options = example();
            let json: JsonValue = options.clone().into();

            assert_eq!(json["type"].as_str(), Some("metatomic_pair_options"));
            assert_eq!(json["cutoff"].as_str(), Some(format!("{:#x}", 3.5_f64.to_bits()).as_str()));
            assert_eq!(json["full_list"].as_bool(), Some(true));
            assert_eq!(json["strict"].as_bool(), Some(false));

            let parsed = PairListOptions::try_from(&json).unwrap();
            assert_eq!(parsed.cutoff.to_bits(), options.cutoff.to_bits());
            assert_eq!(parsed.full_list, options.full_list);
            assert_eq!(parsed.strict, options.strict);
            assert_eq!(parsed.requestors, options.requestors);
        }

        #[test]
        fn cutoff_keeps_full_precision() {
            let mut options = example();
            options.cutoff = 1.0 / 3.0;
            let parsed = PairListOptions::try_from(&JsonValue::from(options.clone())).unwrap();
            assert_eq!(parsed.cutoff.to_bits(), options.cutoff.to_bits());
        }

        #[test]
        fn requestors_are_optional() {
            let mut json: JsonValue = example().into();
            json.remove("requestors");
            let parsed = PairListOptions::try_from(&json).unwrap();
            assert!(parsed.requestors.is_empty());
        }

        #[test]
        fn rejects_invalid_json() {
            // each case corrupts exactly one field of an otherwise valid object
            let with_cutoff = |value: f64| {
                let mut json = JsonValue::from(example());
                json["cutoff"] = format!("{:#x}", value.to_bits()).into();
                json
            };

            let mut wrong_type = JsonValue::from(example());
            wrong_type["type"] = "something-else".into();

            let mut missing_cutoff = JsonValue::from(example());
            missing_cutoff.remove("cutoff");

            let mut non_hex_cutoff = JsonValue::from(example());
            non_hex_cutoff["cutoff"] = "not-hex".into();

            let mut non_boolean_flag = JsonValue::from(example());
            non_boolean_flag["full_list"] = "yes".into();

            let mut non_array_requestors = JsonValue::from(example());
            non_array_requestors["requestors"] = "nl-1".into();

            let mut non_string_requestor = JsonValue::from(example());
            non_string_requestor["requestors"] = json::array![ "nl-1", 42 ];

            let cases = [
                (JsonValue::from("not an object"),
                    "serialization error: invalid JSON data for PairListOptions, expected an object"),
                (wrong_type,
                    "serialization error: 'type' in JSON for PairListOptions must be 'metatomic_pair_options'"),
                (missing_cutoff,
                    "serialization error: 'cutoff' in JSON for PairListOptions must be a hex-encoded string"),
                (non_hex_cutoff,
                    "serialization error: 'cutoff' in JSON for PairListOptions must be a hex-encoded string"),
                (with_cutoff(f64::NAN),
                    "serialization error: 'cutoff' in JSON for PairListOptions must be a finite positive number"),
                (with_cutoff(f64::INFINITY),
                    "serialization error: 'cutoff' in JSON for PairListOptions must be a finite positive number"),
                (with_cutoff(-1.0),
                    "serialization error: 'cutoff' in JSON for PairListOptions must be a finite positive number"),
                (with_cutoff(0.0),
                    "serialization error: 'cutoff' in JSON for PairListOptions must be a finite positive number"),
                (non_boolean_flag,
                    "serialization error: 'full_list' in JSON for PairListOptions must be a boolean"),
                (non_array_requestors,
                    "serialization error: 'requestors' in JSON for PairListOptions must be an array"),
                (non_string_requestor,
                    "serialization error: 'requestors' in JSON for PairListOptions must be an array of strings"),
            ];

            for (json, expected) in cases {
                let error = PairListOptions::try_from(&json).expect_err("expected an error");
                assert_eq!(error.to_string(), expected);
            }
        }

        #[test]
        fn requestors_skip_empty_and_duplicates() {
            let mut json: JsonValue = example().into();
            json["requestors"] = json::array![ "a", "", "b", "a" ];

            let parsed = PairListOptions::try_from(&json).unwrap();
            assert_eq!(parsed.requestors, vec!["a".to_string(), "b".to_string()]);
        }
    }

        mod model_metadata {
        use super::super::*;

        fn example() -> ModelMetadata {
            ModelMetadata {
                name: "test-model".into(),
                authors: vec!["Alice".into(), "Bob <bob@test.com>".into()],
                description: "A test model".into(),
                references: References {
                    model: vec!["doi:10.1234/test".into()],
                    architecture: vec!["doi:10.1234/arch".into()],
                    implementation: vec!["https://github.com/test".into()],
                },
                extra: BTreeMap::from([
                    ("key1".into(), "value1".into()),
                    ("key2".into(), "value2".into()),
                ]),
            }
        }

        #[test]
        fn roundtrip() {
            let metadata = example();
            let json: JsonValue = metadata.clone().into();

            assert_eq!(json["type"].as_str(), Some("metatomic_model_metadata"));
            assert_eq!(json["name"].as_str(), Some("test-model"));
            assert_eq!(json["authors"][0].as_str(), Some("Alice"));
            assert_eq!(json["authors"][1].as_str(), Some("Bob <bob@test.com>"));
            assert_eq!(json["description"].as_str(), Some("A test model"));
            assert_eq!(json["references"]["model"][0].as_str(), Some("doi:10.1234/test"));
            assert_eq!(json["references"]["architecture"][0].as_str(), Some("doi:10.1234/arch"));
            assert_eq!(json["references"]["implementation"][0].as_str(), Some("https://github.com/test"));
            assert_eq!(json["extra"]["key1"].as_str(), Some("value1"));
            assert_eq!(json["extra"]["key2"].as_str(), Some("value2"));

            let parsed = ModelMetadata::try_from(&json).unwrap();
            assert_eq!(parsed.name, metadata.name);
            assert_eq!(parsed.authors, metadata.authors);
            assert_eq!(parsed.description, metadata.description);
            assert_eq!(parsed.references.model, metadata.references.model);
            assert_eq!(parsed.references.architecture, metadata.references.architecture);
            assert_eq!(parsed.references.implementation, metadata.references.implementation);
            assert_eq!(parsed.extra, metadata.extra);
        }

        #[test]
        fn rejects_invalid_json() {
            let mut wrong_type = JsonValue::from(example());
            wrong_type["type"] = "something-else".into();

            let mut missing_name = JsonValue::from(example());
            missing_name.remove("name");

            let mut non_string_name = JsonValue::from(example());
            non_string_name["name"] = 42.into();

            let mut non_array_authors = JsonValue::from(example());
            non_array_authors["authors"] = "Alice".into();

            let mut non_string_author = JsonValue::from(example());
            non_string_author["authors"] = json::array!["Alice", 42];

            let mut missing_description = JsonValue::from(example());
            missing_description.remove("description");

            let mut non_object_extra = JsonValue::from(example());
            non_object_extra["extra"] = "not-an-object".into();

            let mut non_string_extra_value = JsonValue::from(example());
            non_string_extra_value["extra"] = json::object!{ "key" => 42 };

            let mut non_object_references = JsonValue::from(example());
            non_object_references["references"] = "not-an-object".into();

            let cases = [
                (JsonValue::from("not an object"),
                    "serialization error: invalid JSON data for ModelMetadata, expected an object"),
                (wrong_type,
                    "serialization error: 'type' in JSON for ModelMetadata must be 'metatomic_model_metadata'"),
                (missing_name,
                    "serialization error: 'name' in JSON for ModelMetadata must be a string"),
                (non_string_name,
                    "serialization error: 'name' in JSON for ModelMetadata must be a string"),
                (non_array_authors,
                    "serialization error: 'authors' in JSON for ModelMetadata must be an array"),
                (non_string_author,
                    "serialization error: 'authors' in JSON for ModelMetadata must be an array of strings"),
                (missing_description,
                    "serialization error: 'description' in JSON for ModelMetadata must be a string"),
                (non_object_extra,
                    "serialization error: 'extra' in JSON for ModelMetadata must be an object"),
                (non_string_extra_value,
                    "serialization error: 'extra' in JSON for ModelMetadata must be an object with string values"),
                (non_object_references,
                    "serialization error: invalid JSON data for references in ModelMetadata, expected an object"),
            ];

            for (json, expected) in cases {
                let error = ModelMetadata::try_from(&json).expect_err("expected an error");
                assert_eq!(error.to_string(), expected);
            }
        }
    }

    mod model_capabilities {
        use super::super::*;

        fn example() -> ModelCapabilities {
            ModelCapabilities {
                outputs: vec![
                    Quantity {
                        name: "energy".into(),
                        unit: "eV".into(),
                        description: Some("total energy".into()),
                        gradients: vec![crate::Gradients::Positions],
                        sample_kind: crate::SampleKind::System,
                    },
                    Quantity {
                        name: "charge".into(),
                        unit: "e".into(),
                        description: None,
                        gradients: vec![],
                        sample_kind: crate::SampleKind::Atom,
                    },
                ],
                atomic_types: vec![1, 6, 8],
                interaction_range: 5.0,
                length_unit: "Angstrom".into(),
                supported_devices: vec![Device(dlpk::DLDeviceType::kDLCPU), Device(dlpk::DLDeviceType::kDLCUDA)],
                dtype: DType::Float32,
            }
        }

        #[test]
        fn roundtrip() {
            let capabilities = example();
            let json: JsonValue = capabilities.clone().into();

            assert_eq!(json["type"].as_str(), Some("metatomic_model_capabilities"));
            assert_eq!(json["outputs"][0]["name"].as_str(), Some("energy"));
            assert_eq!(json["outputs"][1]["name"].as_str(), Some("charge"));
            assert_eq!(json["atomic_types"][0].as_i64(), Some(1));
            assert_eq!(json["atomic_types"][1].as_i64(), Some(6));
            assert_eq!(json["atomic_types"][2].as_i64(), Some(8));
            assert_eq!(json["interaction_range"].as_f64(), Some(5.0));
            assert_eq!(json["length_unit"].as_str(), Some("Angstrom"));
            assert_eq!(json["supported_devices"][0].as_str(), Some("cpu"));
            assert_eq!(json["supported_devices"][1].as_str(), Some("cuda"));
            assert_eq!(json["dtype"].as_str(), Some("float32"));

            let parsed = ModelCapabilities::try_from(&json).unwrap();
            assert_eq!(parsed.outputs.len(), 2);
            assert_eq!(parsed.outputs[0].name, "energy");
            assert_eq!(parsed.outputs[1].name, "charge");
            assert_eq!(parsed.atomic_types, vec![1, 6, 8]);
            assert_eq!(parsed.interaction_range.to_bits(), 5.0_f64.to_bits());
            assert_eq!(parsed.length_unit, "Angstrom");
            assert_eq!(parsed.supported_devices.len(), 2);
            assert_eq!(parsed.dtype, DType::Float32);
        }

        #[test]
        fn rejects_invalid_json() {
            let mut wrong_type = JsonValue::from(example());
            wrong_type["type"] = "something-else".into();

            let mut non_array_outputs = JsonValue::from(example());
            non_array_outputs["outputs"] = "energy".into();

            let mut non_array_atomic_types = JsonValue::from(example());
            non_array_atomic_types["atomic_types"] = "1".into();

            let mut non_integer_atomic_type = JsonValue::from(example());
            non_integer_atomic_type["atomic_types"] = json::array![1, "x"];

            let mut missing_interaction_range = JsonValue::from(example());
            missing_interaction_range.remove("interaction_range");

            let mut negative_interaction_range = JsonValue::from(example());
            negative_interaction_range["interaction_range"] = (-1.0).into();

            let mut missing_length_unit = JsonValue::from(example());
            missing_length_unit.remove("length_unit");

            let mut wrong_dimension_length_unit = JsonValue::from(example());
            wrong_dimension_length_unit["length_unit"] = "eV".into();

            let mut non_array_supported_devices = JsonValue::from(example());
            non_array_supported_devices["supported_devices"] = "cpu".into();

            let mut invalid_device = JsonValue::from(example());
            invalid_device["supported_devices"] = json::array!["cpu", "wat"];

            let mut missing_dtype = JsonValue::from(example());
            missing_dtype.remove("dtype");

            let mut invalid_dtype = JsonValue::from(example());
            invalid_dtype["dtype"] = "float16".into();

            let cases: Vec<(JsonValue, &str)> = vec![
                (JsonValue::from("not an object"),
                    "serialization error: invalid JSON data for ModelCapabilities, expected an object"),
                (wrong_type,
                    "serialization error: 'type' in JSON for ModelCapabilities must be 'metatomic_model_capabilities'"),
                (non_array_outputs,
                    "serialization error: 'outputs' in JSON for ModelCapabilities must be an array"),
                (non_array_atomic_types,
                    "serialization error: 'atomic_types' in JSON for ModelCapabilities must be an array"),
                (non_integer_atomic_type,
                    "serialization error: 'atomic_types' in JSON for ModelCapabilities must be an array of integers"),
                (missing_interaction_range,
                    "serialization error: 'interaction_range' in JSON for ModelCapabilities must be a number"),
                (negative_interaction_range,
                    "serialization error: 'interaction_range' in JSON for ModelCapabilities must be non-negative"),
                (missing_length_unit,
                    "serialization error: 'length_unit' in JSON for ModelCapabilities must be a string"),
                (wrong_dimension_length_unit,
                    "invalid parameter: dimension mismatch in 'length_unit' in JSON for ModelCapabilities: 'eV' has dimension [L^2 T^-2 M] but expected dimension [L]"),
                (non_array_supported_devices,
                    "serialization error: 'supported_devices' in JSON for ModelCapabilities must be an array"),
                (invalid_device,
                    "serialization error: invalid string for device in JSON for ModelCapabilities, expected 'cpu', 'cuda', 'rocm', or 'metal'"),
                (missing_dtype,
                    "serialization error: dtype in JSON for ModelCapabilities must be a string"),
                (invalid_dtype,
                    "serialization error: invalid string for dtype in JSON for ModelCapabilities, expected 'float32' or 'float64'"),
            ];

            for (json, expected) in cases {
                let error = ModelCapabilities::try_from(&json).expect_err("expected an error");
                assert_eq!(error.to_string(), expected);
            }
        }
    }
}
