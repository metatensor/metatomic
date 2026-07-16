#pragma once

#include <memory>
#include <string>

#include <metatomic.h>
#include <metatomic/errors.hpp>

namespace metatomic {
    /// RAII wrapper around a DLPack `DLManagedTensorVersioned*`.
    ///
    /// This owns the managed tensor and calls its deleter when the wrapper is
    /// destroyed. It can be used to move ownership of DLPack tensors across the
    /// metatomic C++ API.
    class DLPackTensor final {
    public:
        /// Create an empty wrapper, not owning any tensor.
        DLPackTensor() = default;

        /// Take ownership of an existing DLPack managed tensor.
        explicit DLPackTensor(DLManagedTensorVersioned* tensor): tensor_(tensor) {}

        /// The managed tensor is freed through its own deleter on destruction.
        ~DLPackTensor() = default;

        /// `DLPackTensor` is not copy-constructible
        DLPackTensor(const DLPackTensor&) = delete;
        /// `DLPackTensor` is not copy-assignable
        DLPackTensor& operator=(const DLPackTensor&) = delete;

        /// `DLPackTensor` is move-constructible
        DLPackTensor(DLPackTensor&&) noexcept = default;
        /// `DLPackTensor` is move-assignable
        DLPackTensor& operator=(DLPackTensor&&) noexcept = default;

        /// Check whether this wrapper currently owns a tensor.
        explicit operator bool() const {
            return static_cast<bool>(tensor_);
        }

        /// Access the underlying `DLManagedTensorVersioned` without transferring
        /// ownership. The pointer stays owned by this `DLPackTensor`.
        DLManagedTensorVersioned* operator->() const {
            return tensor_.get();
        }

        /// Get the underlying `DLManagedTensorVersioned` pointer. It stays owned
        /// by this `DLPackTensor`, and is only valid for as long as it is alive.
        DLManagedTensorVersioned* as_dlpack() const {
            return tensor_.get();
        }

        /// Release the underlying `DLManagedTensorVersioned` without calling its
        /// deleter, transferring ownership back to the caller.
        DLManagedTensorVersioned* release() {
            return tensor_.release();
        }

    private:
        /// Deleter implementing the DLPack ownership protocol: invoke the managed
        /// tensor's own `deleter` callback if it has one.
        struct Deleter {
            void operator()(DLManagedTensorVersioned* tensor) const noexcept {
                if (tensor->deleter != nullptr) {
                    tensor->deleter(tensor);
                }
            }
        };

        std::unique_ptr<DLManagedTensorVersioned, Deleter> tensor_;
    };

    namespace details {
        /// Take ownership of an `mta_string_t` returned by the C API, copy its
        /// contents into an owned `std::string`, and free the C string.
        ///
        /// The `unique_ptr` guard frees the C string on return, including if the
        /// copy into the `std::string` throws. A null `mta_string_t` (as produced
        /// by an empty output) yields an empty string.
        inline std::string string_from_mta(mta_string_t string) {
            struct Deleter {
                void operator()(mta_string_t ptr) const noexcept {
                    mta_string_free(ptr);
                }
            };
            std::unique_ptr<std::remove_pointer_t<mta_string_t>, Deleter> owned(string);

            if (string == nullptr) {
                return std::string();
            }
            return std::string(mta_string_view(string));
        }
    } // namespace details

    /// Get the multiplicative conversion factor to use to convert from
    /// `from_unit` to `to_unit`. Both units are parsed as expressions
    /// (e.g. `kJ / mol / A^2`, `(eV * u)^(1/2)`) and their dimensions must
    /// match.
    ///
    /// @verbatim embed:rst:leading-slashes
    ///
    /// .. seealso::
    ///
    ///     The general documentation for :ref:`units`, with the expression
    ///     syntax and list of supported base units.
    ///
    /// @endverbatim
    ///
    /// @param from_unit the unit to convert from
    /// @param to_unit the unit to convert to
    inline double unit_conversion_factor(
        const std::string& from_unit,
        const std::string& to_unit
    ) {
        double conversion = 0.0;

        auto status = mta_unit_conversion_factor(from_unit.c_str(), to_unit.c_str(), &conversion);
        details::check_status(status);

        return conversion;
    }
} // namespace metatomic
