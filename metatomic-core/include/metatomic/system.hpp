#pragma once

#include <string>
#include <utility>
#include <vector>

#include <metatomic.h>
#include <metatensor.hpp>
#include <nlohmann/json.hpp>

#include "./metadata.hpp"
#include "./utils.hpp"

namespace metatomic {

/// A System contains all the information about an atomistic system.
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
    ): system_(nullptr), is_view_(false) {
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
        if (system_ != nullptr && !is_view_) {
            (void)mta_system_free(system_);
        }
    }

    System(const System&) = delete;
    System& operator=(const System&) = delete;

    System(System&& other) noexcept: system_(nullptr), is_view_(true) {
        *this = std::move(other);
    }

    System& operator=(System&& other) noexcept {
        if (this != &other) {
            if (system_ != nullptr && !is_view_) {
                (void)mta_system_free(system_);
            }

            system_ = other.system_;
            is_view_ = other.is_view_;
            other.system_ = nullptr;
            other.is_view_ = true;
        }
        return *this;
    }

    size_t size() const {
        uintptr_t result = 0;
        details::check_status(mta_system_size(system_, &result));
        return static_cast<size_t>(result);
    }

    std::string length_unit() const {
        mta_string_t length_unit = nullptr;
        details::check_status(mta_system_get_length_unit(system_, &length_unit));
        return String(length_unit).str();
    }

    DLPackTensor types() const {
        return this->data(MTA_SYSTEM_DATA_TYPES);
    }

    DLPackTensor positions() const {
        return this->data(MTA_SYSTEM_DATA_POSITIONS);
    }

    DLPackTensor cell() const {
        return this->data(MTA_SYSTEM_DATA_CELL);
    }

    DLPackTensor pbc() const {
        return this->data(MTA_SYSTEM_DATA_PBC);
    }

    /// Add a new pair list in this system.
    ///
    /// Ownership of `pairs` is transferred to the C API.
    void add_pairs(const PairListOptions& options, mts_block_t* pairs) {
        details::check_status(mta_system_add_pairs(system_, options.to_json().c_str(), pairs));
    }

    const mts_block_t* pairs_raw(const PairListOptions& options) const {
        const mts_block_t* pairs = nullptr;
        details::check_status(mta_system_get_pairs(system_, options.to_json().c_str(), &pairs));
        details::check_pointer(pairs);
        return pairs;
    }

    metatensor::TensorBlock pairs(const PairListOptions& options) const {
        return metatensor::TensorBlock::unsafe_view_from_ptr(
            const_cast<mts_block_t*>(this->pairs_raw(options))
        );
    }

    std::vector<PairListOptions> known_pairs() const {
        mta_string_t pairs_options = nullptr;
        details::check_status(mta_system_known_pairs(system_, &pairs_options));

        auto result = std::vector<PairListOptions>();
        for (const auto& options: nlohmann::json::parse(String(pairs_options).str())) {
            result.push_back(PairListOptions::from_json(options.dump()));
        }
        return result;
    }

    /// Add custom data to this system.
    ///
    /// Ownership of `data` is transferred to the C API.
    void add_data(const std::string& name, mts_tensormap_t* data) {
        details::check_status(mta_system_add_custom_data(system_, name.c_str(), data));
    }

    const mts_tensormap_t* data_raw(const std::string& name) const {
        const mts_tensormap_t* data = nullptr;
        details::check_status(mta_system_get_custom_data(system_, name.c_str(), &data));
        details::check_pointer(data);
        return data;
    }

    metatensor::TensorMap data(const std::string& name) const {
        auto* copy = mts_tensormap_copy(this->data_raw(name));
        details::check_pointer(copy);
        return metatensor::TensorMap(copy);
    }

    std::vector<std::string> known_data() const {
        mta_string_t names = nullptr;
        details::check_status(mta_system_known_custom_data(system_, &names));
        return nlohmann::json::parse(String(names).str()).get<std::vector<std::string>>();
    }

    mta_system_t* as_mta_system_t() & {
        details::check_pointer(system_);
        return system_;
    }

    const mta_system_t* as_mta_system_t() const & {
        details::check_pointer(system_);
        return system_;
    }

    mta_system_t* as_mta_system_t() && = delete;

    static System unsafe_from_ptr(mta_system_t* system) {
        return System(system, false);
    }

    static System unsafe_view_from_ptr(const mta_system_t* system) {
        return System(const_cast<mta_system_t*>(system), true);
    }

    mta_system_t* release() {
        auto* system = system_;
        system_ = nullptr;
        return system;
    }

private:
    explicit System(mta_system_t* system, bool is_view): system_(system), is_view_(is_view) {}

    DLPackTensor data(mta_system_data_kind request) const {
        DLManagedTensorVersioned* data = nullptr;
        details::check_status(mta_system_get_data(system_, request, &data));
        details::check_pointer(data);
        return DLPackTensor(data);
    }

    mta_system_t* system_;
    bool is_view_;
};

} // namespace metatomic
