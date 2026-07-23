#pragma once

#include <cstdint>
#include <vector>

#include <metatensor.hpp>
#include "metatomic.hpp"

/// Build a `types` tensor for `n_atoms` atoms.
///
/// The tensor has dtype int32 and shape `(n_atoms,)`.
inline metatomic::DLPackTensor types_tensor(size_t n_atoms) {
    auto type_data = std::vector<int32_t>();
    type_data.reserve(n_atoms);
    for (size_t i=0; i<n_atoms; i++) {
        type_data.push_back(static_cast<int32_t>(i * 3 + 1));
    }

    auto array = std::make_unique<metatensor::SimpleDataArray<int32_t>>(
        std::vector<uintptr_t>{n_atoms}, std::move(type_data)
    );
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));

    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    return metatomic::DLPackTensor(mts.as_dlpack(cpu, nullptr, version));
}

/// Build a `positions` tensor for `n_atoms` atoms.
///
/// The tensor has dtype float32 and shape `(n_atoms, 3)`.
inline metatomic::DLPackTensor positions_tensor(size_t n_atoms) {
    auto position_data = std::vector<float>();
    position_data.reserve(n_atoms * 3);
    for (size_t i=0; i<n_atoms; i++) {
        position_data.push_back(static_cast<float>(i * 3 + 1));
        position_data.push_back(static_cast<float>(i * 3 + 2));
        position_data.push_back(static_cast<float>(i * 3 + 3));
    }

    auto array = std::make_unique<metatensor::SimpleDataArray<float>>(
        std::vector<uintptr_t>{n_atoms, 3}, std::move(position_data)
    );
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));

    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    return metatomic::DLPackTensor(mts.as_dlpack(cpu, nullptr, version));
}

/// Build a `cell` tensor.
///
/// The tensor has dtype float32 and shape `(3, 3)`. The `y` row is zero to
/// match the non-periodic `y` direction in `pbc`.
inline metatomic::DLPackTensor cell_tensor() {
    auto array = std::make_unique<metatensor::SimpleDataArray<float>>(
        std::vector<uintptr_t>{3, 3},
        std::vector<float>{
            10.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 10.0F,
        }
    );
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));

    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    return metatomic::DLPackTensor(mts.as_dlpack(cpu, nullptr, version));
}

/// Build a `pbc` tensor.
///
/// The tensor has dtype bool and shape `(3,)`.
inline metatomic::DLPackTensor pbc_tensor() {
    // `SimpleDataArray<bool>` does not compile (`std::vector<bool>` has no
    // `data()` method), so we use `uint8_t` and patch the dtype code to
    // `kDLBool`.
    auto array = std::make_unique<metatensor::SimpleDataArray<uint8_t>>(
        std::vector<uintptr_t>{3}, std::vector<uint8_t>{1, 0, 1}
    );
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));

    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    auto* tensor = mts.as_dlpack(cpu, nullptr, version);
    tensor->dl_tensor.dtype.code = DLDataTypeCode::kDLBool;

    return metatomic::DLPackTensor(tensor);
}

/// Build a simple `System` with `n_atoms` atoms.
inline metatomic::System test_system(size_t n_atoms = 4) {
    return metatomic::System(
        "nm",
        types_tensor(n_atoms),
        positions_tensor(n_atoms),
        cell_tensor(),
        pbc_tensor()
    );
}

/// Build a `TensorMap` holding a single scalar value.
///
/// @param value scalar value to store
/// @param property name of the single property in the returned tensor map
inline metatensor::TensorMap scalar_tensor(double value, const std::string& property) {
    auto values = std::make_unique<metatensor::SimpleDataArray<double>>(
        std::vector<uintptr_t>{1, 1}, std::vector<double>{value}
    );

    auto samples = metatensor::Labels({"system"}, {{0}});
    auto properties = metatensor::Labels({property}, {{0}});

    auto block = metatensor::TensorBlock(
        std::move(values), samples, {}, properties
    );

    auto blocks = std::vector<metatensor::TensorBlock>();
    blocks.push_back(std::move(block));

    auto keys = metatensor::Labels({"_"}, {{0}});
    return metatensor::TensorMap(keys, std::move(blocks));
}
