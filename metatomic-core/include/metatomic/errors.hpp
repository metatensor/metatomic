#pragma once

#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include <metatomic.h>

namespace metatomic {

    /// Exception class used for all errors in metatomic
    class Error: public std::runtime_error {
    public:
        /// Create a new MetatomicError with the given `message`
        Error(const std::string& message): std::runtime_error(message) {}
    };

    namespace details {
        /// Check if a return status from the C API indicates an error, and if it is
        /// the case, throw an exception of type `metatomic::Error` with the last
        /// error message from the library.
        inline void check_status(mta_status_t status) {
            if (status == MTA_SUCCESS) {
                return;
            } else if (status == MTA_CALLBACK_ERROR) {
                const char* message = nullptr;
                const char* origin = nullptr;
                void* data = nullptr;
                mta_last_error(&message, &origin, &data);
                if (origin != nullptr &&std::strcmp(origin, "C++ exception") == 0 && data != nullptr) {
                    std::rethrow_exception(*static_cast<std::exception_ptr*>(data));
                } else {
                    throw Error(message == nullptr ? "unknown error" : message);
                }
            } else {
                const char* message = nullptr;
                mta_last_error(&message, nullptr, nullptr);
                throw Error(message == nullptr ? "unknown error" : message);
            }
        }

        /// Call the given `function` with the given `args` (the function should
        /// return an `mta_status_t`), catching any C++ exception, and translating
        /// them to native metatomic error code.
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
                    [](void *ptr) { delete static_cast<std::exception_ptr*>(ptr); }
                );

                if (status != MTA_SUCCESS) {
                    // If we failed to set the error, we are in a very bad state,
                    // but we should still try to report the original error
                    // message if possible.
                    std::fprintf(stderr, "INTERNAL ERROR: unable to set last error after C++ callback failure (status: %d). ", status);
                    if (message != nullptr) {
                        fprintf(stderr, "C++ error was: %s\n", message);
                    } else {
                        fprintf(stderr, "Unknown C++ error\n");
                    }
                    delete exception_ptr;
                }

                return MTA_CALLBACK_ERROR;
            }
        }

        /// Check if a pointer allocated by the C API is null, and if it is the
        /// case, throw an exception of type `metatomic::Error` with the last
        /// error message from the library.
        inline void check_pointer(const void* pointer) {
            if (pointer == nullptr) {
                const char* message = nullptr;
                const char* origin = nullptr;
                void* data = nullptr;
                mta_last_error(&message, &origin, &data);
                if (std::strcmp(origin, "C++ exception") == 0 && data != nullptr) {
                    std::rethrow_exception(*static_cast<std::exception_ptr*>(data));
                } else {
                    throw Error(message);
                }
            }
        }
    } // namespace details

} // namespace metatomic
