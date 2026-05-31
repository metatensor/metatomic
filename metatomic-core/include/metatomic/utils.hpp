#pragma once

#include <cstring>
#include <cstdio>

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

#include <metatomic.h>

namespace metatomic {

/// Exception class used for all errors in metatomic.
class Error: public std::runtime_error {
public:
    explicit Error(const std::string& message): std::runtime_error(message) {}
};

/// RAII wrapper around a `DLManagedTensorVersioned*`.
class DLPackTensor final {
public:
    DLPackTensor(): tensor_(nullptr) {}

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
        if (this != &other) {
            if (tensor_ != nullptr && tensor_->deleter != nullptr) {
                tensor_->deleter(tensor_);
            }

            tensor_ = other.tensor_;
            other.tensor_ = nullptr;
        }
        return *this;
    }

    explicit operator bool() const {
        return tensor_ != nullptr;
    }

    DLManagedTensorVersioned* get() const {
        return tensor_;
    }

    DLManagedTensorVersioned* as_dlpack() const {
        return tensor_;
    }

    DLManagedTensorVersioned* release() {
        auto* tensor = tensor_;
        tensor_ = nullptr;
        return tensor;
    }

private:
    DLManagedTensorVersioned* tensor_;
};

namespace details {
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

        throw Error(message == nullptr ? "unknown metatomic error" : message);
    }

    template<typename Function>
    inline mta_status_t catch_exceptions(Function function) {
        try {
            function();
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
                std::fprintf(stderr, "INTERNAL ERROR: unable to set last error after C++ callback failure\n");
                delete exception_ptr;
            }

            return MTA_CALLBACK_ERROR;
        }
    }

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
} // namespace details

/// RAII wrapper for `mta_string_t`.
class String final {
public:
    String(): string_(nullptr) {}

    explicit String(const std::string& string): string_(mta_string_create(string.c_str())) {
        details::check_pointer(string_);
    }

    explicit String(mta_string_t string): string_(string) {}

    ~String() {
        mta_string_free(string_);
    }

    String(const String&) = delete;
    String& operator=(const String&) = delete;

    String(String&& other) noexcept: String() {
        *this = std::move(other);
    }

    String& operator=(String&& other) noexcept {
        if (this != &other) {
            mta_string_free(string_);
            string_ = other.string_;
            other.string_ = nullptr;
        }
        return *this;
    }

    const char* c_str() const {
        if (string_ == nullptr) {
            return "";
        }

        auto* ptr = mta_string_view(string_);
        details::check_pointer(ptr);
        return ptr;
    }

    std::string str() const {
        return std::string(this->c_str());
    }

private:
    mta_string_t string_;
};

inline std::string version() {
    auto* raw = mta_version();
    details::check_pointer(raw);
    return std::string(raw);
}

inline double unit_conversion_factor(const std::string& from_unit, const std::string& to_unit) {
    double conversion = 0.0;
    details::check_status(mta_unit_conversion_factor(from_unit.c_str(), to_unit.c_str(), &conversion));
    return conversion;
}

inline std::string format_metadata(const std::string& metadata) {
    mta_string_t printed;
    printed = nullptr;
    details::check_status(mta_format_metadata(metadata.c_str(), &printed));
    return String(printed).str();
}

} // namespace metatomic
