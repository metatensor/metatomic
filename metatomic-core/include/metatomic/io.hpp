#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <metatomic.h>

#include <errors.hpp>
#include <system.hpp>

namespace metatomic {

namespace io {

inline void save(const std::string& path, const System& system) {
    details::check_status(mta_save(path.c_str(), system.as_mta_system_t()));
}

template <typename Buffer>
Buffer save_buffer(const System& system) {
    auto buffer = io::save_buffer<std::vector<uint8_t>>(system);
    return Buffer(buffer.begin(), buffer.end());
}

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

inline System load(const std::string& path, mts_create_array_callback_t create_array) {
    mta_system_t* ptr = nullptr;
    details::check_status(mta_load(path.c_str(), create_array, &ptr));
    details::check_pointer(ptr);
    return System::unsafe_from_ptr(ptr);
}

inline System load_buffer(
    const uint8_t* buffer,
    size_t buffer_count,
    mts_create_array_callback_t create_array
) {
    mta_system_t* ptr = nullptr;
    details::check_status(mta_load_buffer(buffer, buffer_count, create_array, &ptr));
    details::check_pointer(ptr);
    return System::unsafe_from_ptr(ptr);
}

template <typename Buffer>
System load_buffer(const Buffer& buffer, mts_create_array_callback_t create_array) {
    static_assert(
        sizeof(typename Buffer::value_type) == sizeof(uint8_t),
        "`Buffer` must be a container of uint8_t or equivalent"
    );

    return io::load_buffer(
        reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size(), create_array
    );
}

} // namespace io

} // namespace metatomic
