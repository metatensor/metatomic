#pragma once

#include <cstddef>
#include <string>
#include <utility>

#include <metatomic.h>
#include <metatensor.hpp>

#include <metatomic/errors.hpp>
#include <metatomic/utils.hpp>
#include <metatomic/metadata.hpp>

namespace metatomic {
    /// A `System` contains all the information about an atomistic system, and is
    /// used as the input of atomistic models.
    ///
    /// This is a RAII wrapper around the `mta_system_t` type from the C API. It
    /// can either own the underlying system (in which case it is freed with the
    /// `System`), or be a non-owning view into a system owned elsewhere (for
    /// example a system passed to a model by the runtime).
    class System final {
    public:
        /// Create a new `System` from DLPack tensors.
        ///
        /// Ownership of all four tensors is transferred to the new `System`.
        ///
        /// @param length_unit unit of length used by `positions` and `cell`
        /// @param types tensor with shape `(n_atoms,)` of atomic types
        /// @param positions tensor with shape `(n_atoms, 3)` of atomic positions
        /// @param cell tensor with shape `(3, 3)` of the unit cell vectors
        /// @param pbc tensor with shape `(3,)` of periodic boundary conditions
        ///
        /// The dtype and layout required for each tensor are validated by
        /// `mta_system_create`; see the C API documentation for details.
        System(
            const std::string& length_unit,
            DLPackTensor types,
            DLPackTensor positions,
            DLPackTensor cell,
            DLPackTensor pbc
        ) {
            auto status = mta_system_create(
                length_unit.c_str(),
                types.release(),
                positions.release(),
                cell.release(),
                pbc.release(),
                &system_
            );
            details::check_status(status);
            details::check_pointer(system_);
        }

        ~System() {
            if (!is_view_) {
                // `mta_system_free` is a no-op on a null pointer
                mta_system_free(system_);
            }
        }

        /// `System` is not copy-constructible
        System(const System&) = delete;
        /// `System` is not copy-assignable
        System& operator=(const System&) = delete;

        /// `System` is move-constructible
        System(System&& other) noexcept {
            *this = std::move(other);
        }

        /// `System` is move-assignable
        System& operator=(System&& other) noexcept {
            if (!is_view_) {
                mta_system_free(system_);
            }

            system_ = other.system_;
            is_view_ = other.is_view_;

            other.system_ = nullptr;
            other.is_view_ = true;

            return *this;
        }

        /// Get the number of atoms in this system.
        size_t size() const {
            uintptr_t size = 0;
            auto status = mta_system_size(system_, &size);
            details::check_status(status);
            return static_cast<size_t>(size);
        }

        /// Get the unit of length used by the positions and cell of this system.
        std::string length_unit() const {
            mta_string_t length_unit = nullptr;
            auto status = mta_system_get_length_unit(system_, &length_unit);
            details::check_status(status);
            return details::string_from_mta(length_unit);
        }

        /// Get the atomic types of all atoms in this system, as a tensor with
        /// shape `(n_atoms,)`.
        ///
        /// @see `data` for the meaning of the returned tensor.
        DLPackTensor types() const {
            return this->data(MTA_SYSTEM_DATA_TYPES);
        }

        /// Get the positions of all atoms in this system, as a tensor with shape
        /// `(n_atoms, 3)`.
        ///
        /// @see `data` for the meaning of the returned tensor.
        DLPackTensor positions() const {
            return this->data(MTA_SYSTEM_DATA_POSITIONS);
        }

        /// Get the unit cell of this system, as a tensor with shape `(3, 3)`.
        ///
        /// @see `data` for the meaning of the returned tensor.
        DLPackTensor cell() const {
            return this->data(MTA_SYSTEM_DATA_CELL);
        }

        /// Get the periodic boundary conditions of this system, as a tensor with
        /// shape `(3,)`.
        ///
        /// @see `data` for the meaning of the returned tensor.
        DLPackTensor pbc() const {
            return this->data(MTA_SYSTEM_DATA_PBC);
        }

        /// Add a pair list (i.e. neighbor list) to this system.
        ///
        /// Ownership of `pairs` is transferred to this `System`.
        ///
        /// @param options options describing the pair list
        /// @param pairs pairs data, stored as a metatensor block
        void add_pairs(const PairListOptions& options, metatensor::TensorBlock pairs) {
            nlohmann::json j = options;
            this->add_pairs(j.dump(), std::move(pairs));
        }

        /// Add a pair list (i.e. neighbor list) to this system.
        ///
        /// Ownership of `pairs` is transferred to this `System`.
        ///
        /// @param options_json JSON-serialized `PairListOptions` describing the
        ///     pair list
        /// @param pairs pairs data, stored as a metatensor block
        void add_pairs(const std::string& options_json, metatensor::TensorBlock pairs) {
            auto status = mta_system_add_pairs(system_, options_json.c_str(), pairs.release());
            details::check_status(status);
        }

        /// Get a previously stored pair list matching the given `options_json`.
        ///
        /// The returned block is a non-owning view into data owned by this
        /// `System`, and is only valid for as long as this `System` is alive.
        ///
        /// @param options options identifying the pair list to retrieve
        metatensor::TensorBlock pairs(const PairListOptions& options) const {
            nlohmann::json j = options;
            return this->pairs(j.dump());
        }

        /// Get a previously stored pair list matching the given `options_json`.
        ///
        /// The returned block is a non-owning view into data owned by this
        /// `System`, and is only valid for as long as this `System` is alive.
        ///
        /// @param options_json JSON-serialized `PairListOptions` identifying
        ///     the pair list to retrieve
        metatensor::TensorBlock pairs(const std::string& options_json) const {
            const mts_block_t* pairs = nullptr;
            auto status = mta_system_get_pairs(system_, options_json.c_str(), &pairs);
            details::check_status(status);
            details::check_pointer(pairs);
            return metatensor::TensorBlock::unsafe_view_from_ptr(const_cast<mts_block_t*>(pairs));
        }

        /// Get the options of all pair lists registered with this `System`
        std::vector<PairListOptions> known_pairs() const {
            mta_string_t options = nullptr;
            auto status = mta_system_known_pairs(system_, &options);
            details::check_status(status);
            nlohmann::json j = nlohmann::json::parse(mta_string_view(options));
            mta_string_free(options);
            return j.get<std::vector<PairListOptions>>();
        }

        /// Get the options of all pair lists registered with this `System`, as
        /// a JSON-serialized array of `PairListOptions`.
        std::vector<std::string> known_pairs_json() const {
            mta_string_t options = nullptr;
            auto status = mta_system_known_pairs(system_, &options);
            details::check_status(status);
            nlohmann::json j = nlohmann::json::parse(mta_string_view(options));
            mta_string_free(options);
            return j.get<std::vector<std::string>>();
        }

        /// Add custom data to this system, stored under the given `name`.
        ///
        /// Ownership of `data` is transferred to this `System`.
        ///
        /// @param name name used to identify the custom data
        /// @param data custom data, stored as a metatensor tensor map
        void add_custom_data(const std::string& name, metatensor::TensorMap data) {
            auto status = mta_system_add_custom_data(system_, name.c_str(), data.release());
            details::check_status(status);
        }

        /// Get the custom data previously stored under the given `name`.
        ///
        /// The returned tensor map is a non-owning view into data owned by this
        /// `System`, and is only valid for as long as this `System` is alive.
        ///
        /// @param name name of the custom data to retrieve
        metatensor::TensorMap custom_data(const std::string& name) const {
            const mts_tensormap_t* data = nullptr;
            auto status = mta_system_get_custom_data(system_, name.c_str(), &data);
            details::check_status(status);
            details::check_pointer(data);
            return metatensor::TensorMap::unsafe_view_from_ptr(const_cast<mts_tensormap_t*>(data));
        }

        /// Get the names of all custom data registered with this `System`
        std::vector<std::string> known_custom_data() const {
            mta_string_t names = nullptr;
            auto status = mta_system_known_custom_data(system_, &names);
            details::check_status(status);
            nlohmann::json j = nlohmann::json::parse(mta_string_view(names));
            mta_string_free(names);
            return j.get<std::vector<std::string>>();
        }

        /// Get the raw `mta_system_t` pointer backing this `System`.
        ///
        /// The `System` keeps ownership of the pointer, which is only valid for
        /// as long as this `System` is alive.
        mta_system_t* as_mta_system_t() & {
            return system_;
        }

        /// Get the raw `mta_system_t` pointer backing this `System`.
        ///
        /// The `System` keeps ownership of the pointer, which is only valid for
        /// as long as this `System` is alive.
        const mta_system_t* as_mta_system_t() const & {
            return system_;
        }

        /// Getting the raw pointer from a temporary `System` is forbidden, as it
        /// would immediately dangle.
        mta_system_t* as_mta_system_t() && = delete;

        /// Create an owning `System` from a raw `mta_system_t` pointer, taking
        /// ownership of it. The system will be freed when the `System` is
        /// destroyed.
        ///
        /// This is an advanced function, and the caller is responsible for
        /// ensuring that `system` was allocated by the C API and is not used
        /// anywhere else.
        static System unsafe_from_ptr(mta_system_t* system) {
            return System(system, /*is_view*/ false);
        }

        /// Create a non-owning `System` view from a raw `mta_system_t` pointer.
        /// The system will *not* be freed when the `System` is destroyed, and
        /// must outlive it.
        ///
        /// This is an advanced function, mainly useful to wrap the systems given
        /// to a model by the runtime.
        static System unsafe_view_from_ptr(const mta_system_t* system) {
            return System(const_cast<mta_system_t*>(system), /*is_view*/ true);
        }

        /// Release the raw `mta_system_t` pointer from this `System` without
        /// freeing it, transferring ownership back to the caller.
        mta_system_t* release() {
            this->check_not_view("release");
            auto* system = system_;
            system_ = nullptr;
            is_view_ = true;
            return system;
        }

    private:
        /// Wrap an existing `mta_system_t` pointer, see `unsafe_from_ptr` and
        /// `unsafe_view_from_ptr`.
        explicit System(mta_system_t* system, bool is_view):
            system_(system), is_view_(is_view) {}

        void check_not_view(const char* method_name) const {
            if (is_view_) {
                throw Error(
                    "can not call System::" + std::string(method_name) +
                    " on this system since it is a view of a system owned elsewhere."
                );
            }
        }

        /// Get one of the always-present data tensors of this system.
        ///
        /// The returned `DLPackTensor` is a view sharing its data with the
        /// system, which is kept alive for as long as the view exists.
        DLPackTensor data(mta_system_data_kind request) const {
            DLManagedTensorVersioned* data = nullptr;
            auto status = mta_system_get_data(system_, request, &data);
            details::check_status(status);
            details::check_pointer(data);
            return DLPackTensor(data);
        }

        mta_system_t* system_ = nullptr;
        bool is_view_ = false;
    };
} // namespace metatomic
