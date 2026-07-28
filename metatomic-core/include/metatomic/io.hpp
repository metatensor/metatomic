#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <metatomic.h>

#include <metatomic/errors.hpp>
#include <metatomic/system.hpp>

namespace metatomic {
namespace io {

/// Save a system to a file.
///
/// @param path path of the file to create or overwrite
/// @param system system to serialize
inline void save(const std::string& path, const System& system) {
    details::check_status(mta_save(path.c_str(), system.as_mta_system_t()));
}

/// Serialize a system into a byte container.
///
/// `Buffer` must be constructible from a pair of iterators over bytes. The
/// serialization is performed using a `std::vector<uint8_t>` and copied into
/// the requested container type.
///
/// @tparam Buffer byte-container type, such as `std::vector<uint8_t>`
/// @param system system to serialize
/// @return serialized system data
template <typename Buffer>
Buffer save_buffer(const System& system) {
    auto buffer = metatomic::io::save_buffer<std::vector<uint8_t>>(system);
    return Buffer(buffer.begin(), buffer.end());
}

/// Serialize a system into a `std::vector<uint8_t>`.
///
/// The C API grows the vector through a reallocation callback. The returned
/// vector contains exactly the number of bytes produced by the serializer.
///
/// @param system system to serialize
/// @return serialized system data
template <>
inline std::vector<uint8_t> save_buffer<std::vector<uint8_t>>(const System& system) {
    std::vector<uint8_t> buffer;

    auto* ptr = buffer.data();
    auto size = buffer.size();

    auto realloc = [](void* user_data, uint8_t*, uintptr_t new_size) {
        auto* buffer = reinterpret_cast<std::vector<uint8_t>*>(user_data);
        buffer->resize(new_size, '\0');
        return buffer->data();
    };

    details::check_status(mta_save_buffer(&ptr, &size, &buffer, realloc, system.as_mta_system_t()));

    buffer.resize(size, '\0');

    return buffer;
}

/// Load a system from a file.
///
/// @param path path of the serialized system file
/// @param create_array callback used to create arrays during deserialization
/// @return reconstructed system
inline System load(
    const std::string& path,
    mts_create_array_callback_t create_array = metatensor::details::default_create_array
) {
    mta_system_t* ptr = nullptr;
    details::check_status(mta_load(path.c_str(), create_array, &ptr));
    details::check_pointer(ptr);
    return System::unsafe_from_ptr(ptr);
}

/// Load a system from a contiguous byte buffer.
///
/// @param buffer serialized system data
/// @param buffer_count number of bytes available at `buffer`
/// @param create_array callback used to create arrays during deserialization
/// @return reconstructed system
inline System load_buffer(
    const uint8_t* buffer,
    uintptr_t buffer_count,
    mts_create_array_callback_t create_array = metatensor::details::default_create_array
) {
    mta_system_t* ptr = nullptr;
    details::check_status(mta_load_buffer(buffer, buffer_count, create_array, &ptr));
    details::check_pointer(ptr);
    return System::unsafe_from_ptr(ptr);
}

/// Load a system from a byte container.
///
/// The container must provide contiguous storage through `data()` and report
/// its size in bytes through `size()`.
///
/// @tparam Buffer contiguous byte-container type
/// @param buffer serialized system data
/// @param create_array callback used to create arrays during deserialization
/// @return reconstructed system
template <typename Buffer>
System load_buffer(
    const Buffer& buffer,
    mts_create_array_callback_t create_array = metatensor::details::default_create_array
) {
    static_assert(sizeof(typename Buffer::value_type) == sizeof(uint8_t), "`Buffer` must be a container of uint8_t or equivalent");
    return metatomic::io::load_buffer(reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size(), create_array);
}

} // namespace io
} // namespace metatomic
