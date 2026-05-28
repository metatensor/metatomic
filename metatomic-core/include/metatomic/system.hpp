#pragma once

#include <string>
#include <vector>

#include <metatomic.h>
#include <metatensor.hpp>

#include "./utils.hpp"

namespace metatomic {

/// A System contains all the information about an atomistic system, and should
/// be used as input of atomistic models.
class System final {
public:
    /// Create a new `System` from DLPack tensors.
    ///
    /// Ownership of all DLPack tensors is transferred to the C API.
    System(
        const std::string& length_unit,
        DLManagedTensorVersioned* types,
        DLManagedTensorVersioned* positions,
        DLManagedTensorVersioned* cell,
        DLManagedTensorVersioned* pbc
    ): system_(nullptr) {
        details::check_status(mta_system_create(
            length_unit.c_str(),
            types,
            positions,
            cell,
            pbc,
            &system_
        ));
        details::check_pointer(system_);
    }

    ~System() {
        if (system_ != nullptr) {
            (void)mta_system_free(system_);
        }
    }

    System(const System&) = delete;
    System& operator=(const System&) = delete;

    System(System&& other) noexcept: system_(nullptr) {
        *this = std::move(other);
    }

    System& operator=(System&& other) noexcept {
        if (system_ != nullptr) {
            (void)mta_system_free(system_);
        }

        system_ = other.system_;
        other.system_ = nullptr;
        return *this;
    }

    /// Get the number of particles in this system.
    size_t size() const {
        uintptr_t result = 0;
        details::check_status(mta_system_size(system_, &result));
        return static_cast<size_t>(result);
    }

    /// Get the length unit used by positions and cell.
    std::string length_unit() const {
        mta_string_t length_unit = nullptr;
        details::check_status(mta_system_get_length_unit(system_, &length_unit));
        return String(length_unit).str();
    }

    /// Get particle types for all particles in the system.
    DLPackTensor types() const {
        return this->data(MTA_SYSTEM_DATA_TYPES);
    }

    /// Get the positions for all particles in the system.
    DLPackTensor positions() const {
        return this->data(MTA_SYSTEM_DATA_POSITIONS);
    }

    /// Get the unit cell/bounding box of the system.
    DLPackTensor cell() const {
        return this->data(MTA_SYSTEM_DATA_CELL);
    }

    /// Get the periodic boundary conditions for the system.
    DLPackTensor pbc() const {
        return this->data(MTA_SYSTEM_DATA_PBC);
    }

    /// Add a new pair list in this system.
    ///
    /// Ownership of `pairs` is transferred to the C API.
    void set_pairs(const std::string& options, mts_block_t* pairs) {
        details::check_status(mta_system_set_pairs(system_, options.c_str(), pairs));
    }

    /// Retrieve a previously stored pair list with the given options.
    const mts_block_t* pairs_raw(const std::string& options) const {
        const mts_block_t* pairs = nullptr;
        details::check_status(mta_system_get_pairs(system_, options.c_str(), &pairs));
        details::check_pointer(pairs);
        return pairs;
    }

    /// Retrieve a previously stored pair list with the given options as a
    /// non-owning metatensor view.
    metatensor::TensorBlock pairs(const std::string& options) const {
        return metatensor::TensorBlock::unsafe_view_from_ptr(
            const_cast<mts_block_t*>(this->pairs_raw(options))
        );
    }

    /// Get the options for all pair lists registered with this `System`.
    std::vector<std::string> pairs_options() const {
        uintptr_t count = 0;
        details::check_status(mta_system_pairs_count(system_, &count));

        auto result = std::vector<std::string>();
        result.reserve(count);
        for (uintptr_t i=0; i<count; i++) {
            mta_string_t options = nullptr;
            details::check_status(mta_system_pairs_options(system_, i, &options));
            result.push_back(String(options).str());
        }

        return result;
    }

    /// Add custom data to this system.
    ///
    /// Ownership of `data` is transferred to the C API.
    void set_data(const std::string& name, mts_tensormap_t* data) {
        details::check_status(mta_system_set_custom_data(system_, name.c_str(), data));
    }

    /// Retrieve custom data stored in this system.
    ///
    /// The returned pointer is borrowed and owned by this `System`.
    const mts_tensormap_t* data_raw(const std::string& name) const {
        const mts_tensormap_t* data = nullptr;
        details::check_status(mta_system_get_custom_data(system_, name.c_str(), &data));
        details::check_pointer(data);
        return data;
    }

    /// Get the names of all custom data registered with this `System`.
    std::vector<std::string> data_names() const {
        uintptr_t count = 0;
        details::check_status(mta_system_data_count(system_, &count));

        auto result = std::vector<std::string>();
        result.reserve(count);
        for (uintptr_t i=0; i<count; i++) {
            mta_string_t name = nullptr;
            details::check_status(mta_system_data_name(system_, i, &name));
            result.push_back(String(name).str());
        }

        return result;
    }

    /// Get the underlying `mta_system_t` pointer.
    mta_system_t* as_mta_system_t() & {
        details::check_pointer(system_);
        return system_;
    }

    /// Get the underlying `mta_system_t` pointer.
    const mta_system_t* as_mta_system_t() const & {
        details::check_pointer(system_);
        return system_;
    }

    mta_system_t* as_mta_system_t() && = delete;

    /// Take ownership of a raw `mta_system_t*`.
    static System unsafe_from_ptr(mta_system_t* system) {
        return System(system);
    }

    /// Release the raw `mta_system_t*` without freeing it.
    mta_system_t* release() {
        auto* system = system_;
        system_ = nullptr;
        return system;
    }

private:
    explicit System(mta_system_t* system): system_(system) {}

    DLPackTensor data(mta_system_data_kind request) const {
        DLManagedTensorVersioned* data = nullptr;
        details::check_status(mta_system_get_data(system_, request, &data));
        details::check_pointer(data);
        return DLPackTensor(data);
    }

    mta_system_t* system_;
};

} // namespace metatomic
