#pragma once

#include <cstring>
#include <cstdio>

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <metatomic.h>

namespace metatomic {

/// Exception class used for all errors in metatomic.
class Error: public std::runtime_error {
public:
    /// Create a new Error with the given `message`.
    explicit Error(const std::string& message): std::runtime_error(message) {}
};

/// Key/value pair used when loading models from plugins.
struct KeyValuePair {
    std::string key;
    std::string value;
};

/// RAII wrapper around a `DLManagedTensorVersioned*`.
///
/// This owns the DLPack managed tensor object, and calls its deleter when the
/// wrapper is destroyed.
class DLPackTensor final {
public:
    /// Create an empty wrapper.
    DLPackTensor(): tensor_(nullptr) {}

    /// Take ownership of an existing DLPack managed tensor.
    explicit DLPackTensor(DLManagedTensorVersioned* tensor): tensor_(tensor) {}

    ~DLPackTensor() {
        if (tensor_ != nullptr && tensor_->deleter != nullptr) {
            tensor_->deleter(tensor_);
        }
    }

    DLPackTensor(const DLPackTensor&) = delete;
    DLPackTensor& operator=(const DLPackTensor&) = delete;

    DLPackTensor(DLPackTensor&& other) noexcept: DLPackTensor() {
        *this = std::move(other);
    }

    DLPackTensor& operator=(DLPackTensor&& other) noexcept {
        if (tensor_ != nullptr && tensor_->deleter != nullptr) {
            tensor_->deleter(tensor_);
        }

        tensor_ = other.tensor_;
        other.tensor_ = nullptr;
        return *this;
    }

    /// Check if this wrapper contains a tensor.
    explicit operator bool() const {
        return tensor_ != nullptr;
    }

    /// Get the underlying DLPack managed tensor.
    DLManagedTensorVersioned* get() const {
        return tensor_;
    }

    /// Get the underlying DLPack managed tensor.
    DLManagedTensorVersioned* as_dlpack() const {
        return tensor_;
    }

    /// Release the DLPack managed tensor without calling its deleter.
    DLManagedTensorVersioned* release() {
        auto* tensor = tensor_;
        tensor_ = nullptr;
        return tensor;
    }

private:
    DLManagedTensorVersioned* tensor_;
};

namespace details {
    /// Check if a return status from the C API indicates an error, and throw a
    /// `metatomic::Error` with the last error message if this is the case.
    inline void check_status(mta_status_t status) {
        if (status == MTA_SUCCESS) {
            return;
        }

        const char* message = nullptr;
        const char* origin = nullptr;
        void* data = nullptr;
        (void)mta_last_error(&message, &origin, &data);

        if (origin != nullptr && std::strcmp(origin, "C++ exception") == 0 && data != nullptr) {
            std::rethrow_exception(*static_cast<std::exception_ptr*>(data));
        }

        throw Error(message == nullptr ? "unknown error" : message);
    }

    /// Call the given `function`, catching any C++ exception and translating it
    /// to a metatomic error code.
    ///
    /// This is required to prevent callbacks unwinding through the C API.
    template<typename Function, typename ...Args>
    inline mta_status_t catch_exceptions(Function function, Args ...args) {
        try {
            function(std::move(args)...);
            return MTA_SUCCESS;
        } catch (...) {
            auto* exception_ptr = new std::exception_ptr(std::current_exception());

            const char* message = nullptr;
            try {
                std::rethrow_exception(*exception_ptr);
            } catch (const std::exception& e) {
                message = e.what();
            } catch (...) {
                message = "C++ code threw an exception that was not a std::exception";
            }

            auto status = mta_set_last_error(
                message,
                "C++ exception",
                exception_ptr,
                [](void* ptr) { delete static_cast<std::exception_ptr*>(ptr); }
            );

            if (status != MTA_SUCCESS) {
                std::fprintf(
                    stderr,
                    "INTERNAL ERROR: unable to set last error after C++ callback failure (status: %d). ",
                    static_cast<int>(status)
                );
                if (message != nullptr) {
                    std::fprintf(stderr, "C++ error was: %s\n", message);
                } else {
                    std::fprintf(stderr, "Unknown C++ error\n");
                }
                delete exception_ptr;
            }

            return MTA_ERROR_OTHER;
        }
    }

    /// Check if a pointer allocated by the C API is null.
    inline void check_pointer(const void* pointer) {
        if (pointer != nullptr) {
            return;
        }

        const char* message = nullptr;
        const char* origin = nullptr;
        void* data = nullptr;
        (void)mta_last_error(&message, &origin, &data);

        if (origin != nullptr && std::strcmp(origin, "C++ exception") == 0 && data != nullptr) {
            std::rethrow_exception(*static_cast<std::exception_ptr*>(data));
        }

        throw Error(message == nullptr ? "received a null pointer from the metatomic C API" : message);
    }

    inline std::vector<mta_kv_pair_t> to_c_options(const std::vector<KeyValuePair>& options) {
        auto c_options = std::vector<mta_kv_pair_t>();
        c_options.reserve(options.size());

        for (const auto& option: options) {
            c_options.push_back(mta_kv_pair_t{option.key.c_str(), option.value.c_str()});
        }

        return c_options;
    }

    inline std::vector<KeyValuePair> from_c_options(const mta_kv_pair_t* options, uintptr_t count) {
        auto result = std::vector<KeyValuePair>();
        result.reserve(count);

        if (count != 0) {
            check_pointer(options);
        }

        for (uintptr_t i=0; i<count; i++) {
            result.push_back(KeyValuePair{
                options[i].key == nullptr ? "" : options[i].key,
                options[i].value == nullptr ? "" : options[i].value,
            });
        }

        return result;
    }
} // namespace details

/// RAII wrapper for `mta_string_t`.
class String final {
public:
    /// Create an empty string wrapper.
    String(): string_(nullptr) {}

    /// Create a new string managed by metatomic.
    explicit String(const std::string& string): string_(mta_string_create(string.c_str())) {
        details::check_pointer(string_);
    }

    /// Take ownership of an existing `mta_string_t`.
    explicit String(mta_string_t string): string_(string) {}

    ~String() {
        if (string_ != nullptr) {
            mta_string_free(string_);
        }
    }

    String(const String&) = delete;
    String& operator=(const String&) = delete;

    String(String&& other) noexcept: String() {
        *this = std::move(other);
    }

    String& operator=(String&& other) noexcept {
        if (string_ != nullptr) {
            mta_string_free(string_);
        }

        string_ = other.string_;
        other.string_ = nullptr;
        return *this;
    }

    /// Get the underlying C pointer.
    mta_string_t as_mta_string_t() const {
        return string_;
    }

    /// View this string as a null-terminated C string.
    const char* c_str() const {
        if (string_ == nullptr) {
            return "";
        }

        auto* ptr = mta_string_view(string_);
        details::check_pointer(ptr);
        return ptr;
    }

    /// Copy this string into a C++ string.
    std::string str() const {
        return std::string(this->c_str());
    }

private:
    mta_string_t string_;
};

/// Get the runtime version of metatomic as a string.
inline std::string version() {
    auto* raw = mta_version();
    details::check_pointer(raw);
    return std::string(raw);
}

/// Get the conversion factor from `from_unit` to `to_unit`.
inline double unit_conversion_factor(const std::string& from_unit, const std::string& to_unit) {
    double conversion = 0.0;
    details::check_status(mta_unit_conversion_factor(from_unit.c_str(), to_unit.c_str(), &conversion));
    return conversion;
}

/// Format model metadata JSON for display.
inline std::string format_metadata(const std::string& metadata) {
    mta_string_t printed = nullptr;
    details::check_status(mta_format_metadata(metadata.c_str(), &printed));
    return String(printed).str();
}

} // namespace metatomic
