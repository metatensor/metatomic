#ifndef METATOMIC_TESTS_UTILS_HPP
#define METATOMIC_TESTS_UTILS_HPP

#include <cstdint>
#include <vector>

#include <metatensor.hpp>

/// Helpers to create DLPack tensors for System creation through the C API.
/// These return raw `DLManagedTensorVersioned*` pointers that are transferred
/// to `mta_system_create` (which takes ownership).

template <typename T>
inline DLManagedTensorVersioned* types_tensor(size_t n_atoms) {
    std::vector<T> type_data;
    type_data.reserve(n_atoms);
    for (size_t i = 0; i < n_atoms; i++) {
        type_data.push_back(static_cast<T>(i * 3 + 1));
    }
    auto array = std::make_unique<metatensor::SimpleDataArray<T>>(
        std::vector<uintptr_t>{n_atoms},
        std::move(type_data)
    );
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));
    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    return mts.as_dlpack(cpu, nullptr, version);
}

template <typename T>
inline DLManagedTensorVersioned* positions_tensor(size_t n_atoms) {
    std::vector<T> position_data;
    position_data.reserve(n_atoms * 3);
    for (size_t i = 0; i < n_atoms; i++) {
        position_data.push_back(static_cast<T>(i * 3 + 1));
        position_data.push_back(static_cast<T>(i * 3 + 2));
        position_data.push_back(static_cast<T>(i * 3 + 3));
    }
    auto array = std::make_unique<metatensor::SimpleDataArray<T>>(
        std::vector<uintptr_t>{n_atoms, 3},
        std::move(position_data)
    );
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));
    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    return mts.as_dlpack(cpu, nullptr, version);
}

template <typename T>
inline DLManagedTensorVersioned* cell_tensor() {
    auto array = std::make_unique<metatensor::SimpleDataArray<T>>(
        std::vector<uintptr_t>{3, 3},
        std::vector<T>{
            T(10.0), T(0.0), T(0.0),
            T(0.0), T(0.0), T(0.0),
            T(0.0), T(0.0), T(10.0),
        }
    );
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));
    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    return mts.as_dlpack(cpu, nullptr, version);
}

inline DLManagedTensorVersioned* pbc_tensor() {
    std::vector<uint8_t> pbc_data = {1, 0, 1};
    auto array = std::make_unique<metatensor::SimpleDataArray<bool>>(
        std::vector<uintptr_t>{3},
        std::move(pbc_data)
    );
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));
    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    return mts.as_dlpack(cpu, nullptr, version);
}

#endif // METATOMIC_TESTS_UTILS_HPP
