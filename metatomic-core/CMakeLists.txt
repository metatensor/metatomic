# This file defines the CMake build system for the C and C++ API of metatomic.
#
# This API is implemented in Rust, in the metatomic-core crate, but Rust users
# of the API should use the metatomic crate instead, wrapping metatomic-core in
# an easier to use, idiomatic Rust API.
cmake_minimum_required(VERSION 3.22)

if (POLICY CMP0135)
    # Use download time as timestamp when extracting files from archives.
    cmake_policy(SET CMP0135 NEW)
endif()

# Is metatomic the main project configured by the user? Or is this being used
# as a submodule/subdirectory?
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    set(METATOMIC_MAIN_PROJECT ON)
else()
    set(METATOMIC_MAIN_PROJECT OFF)
endif()

if(${METATOMIC_MAIN_PROJECT} AND NOT "${CACHED_LAST_CMAKE_VERSION}" VERSION_EQUAL ${CMAKE_VERSION})
    # We use CACHED_LAST_CMAKE_VERSION to only print the cmake version
    # once in the configuration log
    set(CACHED_LAST_CMAKE_VERSION ${CMAKE_VERSION} CACHE INTERNAL "Last version of cmake used to configure")
    message(STATUS "Running CMake version ${CMAKE_VERSION}")
endif()

if (POLICY CMP0077)
    # use variables to set OPTIONS
    cmake_policy(SET CMP0077 NEW)
endif()

file(STRINGS "Cargo.toml" CARGO_TOML_CONTENT)
foreach(line ${CARGO_TOML_CONTENT})
    string(REGEX REPLACE "^version = \"(.*)\"" "\\1" METATOMIC_VERSION ${line})
    if (NOT ${CMAKE_MATCH_COUNT} EQUAL 0)
        # stop on the first regex match, this should be the right version
        break()
    endif()
endforeach()

include(cmake/dev-versions.cmake)
create_development_version("${METATOMIC_VERSION}" METATOMIC_FULL_VERSION "metatomic-core-v")
message(STATUS "Building metatomic-core v${METATOMIC_FULL_VERSION}")

# strip any -dev/-rc suffix on the version since project(VERSION) does not support it
string(REGEX REPLACE "([0-9]*)\\.([0-9]*)\\.([0-9]*).*" "\\1.\\2.\\3" METATOMIC_VERSION ${METATOMIC_FULL_VERSION})
project(metatomic
    VERSION ${METATOMIC_VERSION}
    LANGUAGES C CXX # we need to declare a language to access CMAKE_SIZEOF_VOID_P later
)
set(PROJECT_VERSION ${METATOMIC_FULL_VERSION})


# We follow the standard CMake convention of using BUILD_SHARED_LIBS to provide
# either a shared or static library as a default target. But since cargo always
# builds both versions by default, we also install both versions by default.
# `METATOMIC_INSTALL_BOTH_STATIC_SHARED=OFF` allow to disable this behavior, and
# only install the file corresponding to `BUILD_SHARED_LIBS=ON/OFF`.
#
# BUILD_SHARED_LIBS controls the `metatomic` cmake target, making it an alias of
# either `metatomic::static` or `metatomic::shared`. This is mainly relevant
# when using metatomic from another cmake project, either as a submodule or from
# an installed library (see cmake/metatomic-config.cmake)
option(BUILD_SHARED_LIBS "Use a shared library by default instead of a static one" ON)
option(METATOMIC_INSTALL_BOTH_STATIC_SHARED "Install both shared and static libraries" ON)

set(RUST_BUILD_TARGET "${RUST_BUILD_TARGET}" CACHE STRING "Cross-compilation target for rust code. Leave empty to build for the host")
set(EXTRA_RUST_FLAGS "${EXTRA_RUST_FLAGS}" CACHE STRING "Flags used to build rust code")

include(GNUInstallDirs)

if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
    message(STATUS "Setting build type to 'release' as none was specified.")
    set(CMAKE_BUILD_TYPE "release"
        CACHE STRING
        "Choose the type of build, options are: debug or release"
    FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug)
endif()

if(${METATOMIC_MAIN_PROJECT} AND NOT "${CACHED_LAST_CMAKE_BUILD_TYPE}" STREQUAL "${CMAKE_BUILD_TYPE}")
    set(CACHED_LAST_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE INTERNAL "Last build type used in configuration")
    message(STATUS "Building metatomic in ${CMAKE_BUILD_TYPE} mode")
endif()


function(check_compatible_versions _actual_ _requested_)
    if(${_actual_} MATCHES "^([0-9]+)\\.([0-9]+)")
        set(_actual_major_ "${CMAKE_MATCH_1}")
        set(_actual_minor_ "${CMAKE_MATCH_2}")
    else()
        message(FATAL_ERROR "Failed to parse actual version: ${_actual_}")
    endif()

    if(${_requested_} MATCHES "^([0-9]+)\\.([0-9]+)")
        set(_requested_major_ "${CMAKE_MATCH_1}")
        set(_requested_minor_ "${CMAKE_MATCH_2}")
    else()
        message(FATAL_ERROR "Failed to parse requested version: ${_requested_}")
    endif()

    if (${_requested_major_} EQUAL 0 AND ${_actual_minor_} EQUAL ${_requested_minor_})
        # major version is 0 and same minor version, everything is fine
    elseif (${_actual_major_} EQUAL ${_requested_major_})
        # same major version, everything is fine
    else()
        # not compatible
        message(FATAL_ERROR "Incompatible versions: we need ${_requested_}, but we got ${_actual_}")
    endif()
endfunction()


set(REQUIRED_METATENSOR_VERSION "0.2.0")
# Either metatensor is built as part of the same CMake project, or we try to
# find the corresponding CMake package
if (TARGET metatensor)
    get_target_property(METATENSOR_BUILD_VERSION metatensor BUILD_VERSION)
    check_compatible_versions(${METATENSOR_BUILD_VERSION} ${REQUIRED_METATENSOR_VERSION})
else()
    find_package(metatensor ${REQUIRED_METATENSOR_VERSION} CONFIG REQUIRED)
endif()


find_program(CARGO_EXE "cargo" DOC "path to cargo (Rust build system)")
if (NOT CARGO_EXE)
    message(FATAL_ERROR
        "could not find cargo, please make sure the Rust compiler is installed \
        (see https://www.rust-lang.org/tools/install) or set CARGO_EXE"
    )
endif()

execute_process(
    COMMAND ${CARGO_EXE} "--version" "--verbose"
    RESULT_VARIABLE CARGO_STATUS
    OUTPUT_VARIABLE CARGO_VERSION_RAW
)

if(CARGO_STATUS AND NOT CARGO_STATUS EQUAL 0)
    message(FATAL_ERROR
        "could not run cargo, please make sure the Rust compiler is installed \
        (see https://www.rust-lang.org/tools/install)"
    )
endif()

set(REQUIRED_RUST_VERSION "1.74.0")
if (CARGO_VERSION_RAW MATCHES "cargo ([0-9]+\\.[0-9]+\\.[0-9]+).*")
    set(CARGO_VERSION "${CMAKE_MATCH_1}")
else()
    message(FATAL_ERROR "failed to determine cargo version, output was: ${CARGO_VERSION_RAW}")
endif()

if (${CARGO_VERSION} VERSION_LESS ${REQUIRED_RUST_VERSION})
    message(FATAL_ERROR
        "your Rust installation is too old (you have version ${CARGO_VERSION}), \
        at least ${REQUIRED_RUST_VERSION} is required"
    )
else()
    if(NOT "${CACHED_LAST_CARGO_VERSION}" STREQUAL ${CARGO_VERSION})
        set(CACHED_LAST_CARGO_VERSION ${CARGO_VERSION} CACHE INTERNAL "Last version of cargo used in configuration")
        message(STATUS "Using cargo version ${CARGO_VERSION} at ${CARGO_EXE}")
        set(CARGO_VERSION_CHANGED TRUE)
    endif()
endif()

# ============================================================================ #
# determine Cargo flags

set(CARGO_BUILD_ARG "")

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.lock)
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--locked")
endif()

# TODO: support multiple configuration generators (MSVC, ...)
string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
if ("${BUILD_TYPE}" STREQUAL "debug")
    set(CARGO_BUILD_TYPE "debug")
elseif("${BUILD_TYPE}" STREQUAL "release")
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--release")
    set(CARGO_BUILD_TYPE "release")
elseif("${BUILD_TYPE}" STREQUAL "relwithdebinfo")
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--release")
    set(CARGO_BUILD_TYPE "release")
else()
    message(FATAL_ERROR "unsuported build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CARGO_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/target)
set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--target-dir=${CARGO_TARGET_DIR}")

if (CARGO_VERSION_RAW MATCHES "host: ([a-zA-Z0-9_\\-]*)\n")
    set(RUST_HOST_TARGET "${CMAKE_MATCH_1}")
    if (RUST_HOST_TARGET MATCHES "([a-zA-Z0-9_]*)\\-")
        set(RUST_HOST_ARCH "${CMAKE_MATCH_1}")
    else()
        message(FATAL_ERROR "failed to determine host CPU arch, target was: ${RUST_HOST_TARGET}")
    endif()
else()
    message(FATAL_ERROR "failed to determine host target, output was: ${CARGO_VERSION_RAW}")
endif()

if (WIN32)
    # on Windows, we need to use the same ABI in both CMake and cargo. If the
    # user did not explicitly request a target, we can try to set it ourself,
    # otherwise we just check that it matches what we expect.
    if (MSVC)
        if ("${RUST_BUILD_TARGET}" STREQUAL "")
            set(RUST_BUILD_TARGET "${RUST_HOST_ARCH}-pc-windows-msvc")
            message(STATUS "Setting rust target to ${RUST_BUILD_TARGET}")
        elseif(NOT "${RUST_BUILD_TARGET}" MATCHES "-pc-windows-msvc")
            message(FATAL_ERROR "CMake is building with MSVC but the Rust target is ${RUST_BUILD_TARGET}")
        endif()
    endif()

    if (MINGW)
        if ("${RUST_BUILD_TARGET}" STREQUAL "")
            set(RUST_BUILD_TARGET "${RUST_HOST_ARCH}-pc-windows-gnu")
            message(STATUS "Setting rust target to ${RUST_BUILD_TARGET}")
        elseif(NOT "${RUST_BUILD_TARGET}" MATCHES "-pc-windows-gnu")
            message(FATAL_ERROR "CMake is building with MinGW but the Rust target is ${RUST_BUILD_TARGET}")
        endif()
    endif()
endif()

# Handle cross compilation with RUST_BUILD_TARGET
if ("${RUST_BUILD_TARGET}" STREQUAL "")
    if (${METATOMIC_MAIN_PROJECT})
        message(STATUS "Compiling to host (${RUST_HOST_TARGET})")
    endif()

    set(CARGO_OUTPUT_DIR "${CARGO_TARGET_DIR}/${CARGO_BUILD_TYPE}")
    set(RUST_BUILD_TARGET ${RUST_HOST_TARGET})
else()
    if (${METATOMIC_MAIN_PROJECT})
        message(STATUS "Cross-compiling to ${RUST_BUILD_TARGET}")
    endif()

    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--target=${RUST_BUILD_TARGET}")
    set(CARGO_OUTPUT_DIR "${CARGO_TARGET_DIR}/${RUST_BUILD_TARGET}/${CARGO_BUILD_TYPE}")
endif()

# Get the list of libraries linked by default by cargo/rustc to add when linking
# to metatomic::static
if (CARGO_VERSION_CHANGED)
    include(cmake/tempdir.cmake)
    get_tempdir(TMPDIR)

    # Adapted from https://github.com/corrosion-rs/corrosion/blob/dc1e4e5/cmake/FindRust.cmake
    execute_process(
        COMMAND "${CARGO_EXE}" new --lib _cargo_required_libs
        WORKING_DIRECTORY "${TMPDIR}"
        RESULT_VARIABLE cargo_new_result
        ERROR_QUIET
    )

    if (cargo_new_result)
        message(FATAL_ERROR "could not create empty project to find default static libs: ${cargo_new_result}")
    endif()

    file(APPEND "${TMPDIR}/_cargo_required_libs/Cargo.toml" "[lib]\ncrate-type=[\"staticlib\"]")

    execute_process(
        COMMAND ${CARGO_EXE} rustc --color never --target=${RUST_BUILD_TARGET} -- --print=native-static-libs
        WORKING_DIRECTORY "${TMPDIR}/_cargo_required_libs"
        RESULT_VARIABLE cargo_static_libs_result
        ERROR_VARIABLE cargo_static_libs_stderr
    )

    # clean up the files
    file(REMOVE_RECURSE "${TMPDIR}")

    if (cargo_static_libs_result)
        message(FATAL_ERROR
            "could not extract default static libs (status ${cargo_static_libs_result}), stderr:\n${cargo_static_libs_stderr}"
        )
    endif()

    # The pattern starts with `native-static-libs:` and goes to the end of the line.
    if (cargo_static_libs_stderr MATCHES "native-static-libs: ([^\r\n]+)\r?\n")
        string(REPLACE " " ";" "libs_list" "${CMAKE_MATCH_1}")
        set(stripped_lib_list "")
        foreach(lib ${libs_list})
            # Strip leading `-l` (unix) and potential .lib suffix (windows)
            string(REGEX REPLACE "^-l" "" "stripped_lib" "${lib}")
            string(REGEX REPLACE "\.lib$" "" "stripped_lib" "${stripped_lib}")
            list(APPEND stripped_lib_list "${stripped_lib}")
        endforeach()

        # Special case `msvcrt` to link with the debug version in Debug mode.
        list(TRANSFORM stripped_lib_list REPLACE "^msvcrt$" "\$<\$<CONFIG:Debug>:msvcrtd>")
        # Don't try to pass a linker *flag* where CMake expects libraries
        list(REMOVE_ITEM stripped_lib_list "/defaultlib:msvcrt")

        if (APPLE)
            # Prevent warnings about duplicated `System` in linked libraries
            # from Apple's `ld`
            list(REMOVE_ITEM stripped_lib_list "System")
        endif()

        list(REMOVE_DUPLICATES stripped_lib_list)
        set(CARGO_DEFAULT_LIBRARIES "${stripped_lib_list}" CACHE INTERNAL "list of implicitly linked libraries")

        if (${METATOMIC_MAIN_PROJECT})
            message(STATUS "Cargo default link libraries are: ${CARGO_DEFAULT_LIBRARIES}")
        endif()
    else()
        message(FATAL_ERROR "could not find default static libs: `native-static-libs` not found in: `${cargo_static_libs_stderr}`")
    endif()
endif()

file(GLOB_RECURSE ALL_RUST_SOURCES
    ${PROJECT_SOURCE_DIR}/Cargo.toml
    ${PROJECT_SOURCE_DIR}/src/**.rs
)

add_library(metatomic::shared SHARED IMPORTED GLOBAL)
set(METATOMIC_SHARED_LOCATION "${CARGO_OUTPUT_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}metatomic${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(METATOMIC_IMPLIB_LOCATION "${METATOMIC_SHARED_LOCATION}.lib")

if (MINGW)
    # `rustc` does not follow the usual naming scheme for DLL with mingw (it
    # would typically be 'libmetatomic.dll')
    set(METATOMIC_SHARED_LOCATION "${CARGO_OUTPUT_DIR}/metatomic.dll")
    set(METATOMIC_IMPLIB_LOCATION "${CARGO_OUTPUT_DIR}/libmetatomic.dll.a")
endif()

add_library(metatomic::static STATIC IMPORTED GLOBAL)
set(METATOMIC_STATIC_LOCATION "${CARGO_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}metatomic${CMAKE_STATIC_LIBRARY_SUFFIX}")

get_filename_component(METATOMIC_SHARED_LIB_NAME ${METATOMIC_SHARED_LOCATION} NAME)
get_filename_component(METATOMIC_IMPLIB_NAME     ${METATOMIC_IMPLIB_LOCATION} NAME)
get_filename_component(METATOMIC_STATIC_LIB_NAME ${METATOMIC_STATIC_LOCATION} NAME)

# We need to add some metadata to the shared library to enable linking to it
# without using an absolute path.
if (UNIX)
    if (APPLE)
        # set the install name to `@rpath/libmetatomic.dylib`
        set(CARGO_RUSTC_ARGS "-Clink-arg=-Wl,-install_name,@rpath/${METATOMIC_SHARED_LIB_NAME}")
        set_target_properties(metatomic::shared PROPERTIES
            IMPORTED_SONAME @rpath/${METATOMIC_SHARED_LIB_NAME}
        )
    else() # LINUX
        # set the SONAME to libmetatomic.so
        set(CARGO_RUSTC_ARGS "-Clink-arg=-Wl,-soname,${METATOMIC_SHARED_LIB_NAME}")
        set_target_properties(metatomic::shared PROPERTIES
            IMPORTED_SONAME ${METATOMIC_SHARED_LIB_NAME}
        )
    endif()
else()
    set(CARGO_RUSTC_ARGS "")
endif()

if (NOT "${EXTRA_RUST_FLAGS}" STREQUAL "")
    set(CARGO_RUSTC_ARGS "${CARGO_RUSTC_ARGS};${EXTRA_RUST_FLAGS}")
endif()

# Set environment variables for cargo build
set(CARGO_ENV "METATOMIC_FULL_VERSION=${METATOMIC_FULL_VERSION}")
if (NOT "${CMAKE_OSX_DEPLOYMENT_TARGET}" STREQUAL "")
    list(APPEND CARGO_ENV "MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()
if (NOT "$ENV{RUSTC_WRAPPER}" STREQUAL "")
    list(APPEND CARGO_ENV "RUSTC_WRAPPER=$ENV{RUSTC_WRAPPER}")
endif()

if (METATOMIC_INSTALL_BOTH_STATIC_SHARED)
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--crate-type=cdylib;--crate-type=staticlib")
    set(CARGO_OUTPUTS ${METATOMIC_SHARED_LOCATION} ${METATOMIC_STATIC_LOCATION})
    if (WIN32)
        list(APPEND CARGO_OUTPUTS ${METATOMIC_IMPLIB_LOCATION})
        set(FILE_CREATED_MESSAGE "${METATOMIC_SHARED_LIB_NAME}, ${METATOMIC_STATIC_LIB_NAME}, and ${METATOMIC_IMPLIB_NAME}")
    else()
        set(FILE_CREATED_MESSAGE "${METATOMIC_SHARED_LIB_NAME} and ${METATOMIC_STATIC_LIB_NAME}")
    endif()
else()
    if (BUILD_SHARED_LIBS)
        set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--crate-type=cdylib")
        set(CARGO_OUTPUTS ${METATOMIC_SHARED_LOCATION})
        if (WIN32)
            list(APPEND CARGO_OUTPUTS ${METATOMIC_IMPLIB_LOCATION})
            set(FILE_CREATED_MESSAGE "${METATOMIC_SHARED_LIB_NAME} and ${METATOMIC_IMPLIB_NAME}")
        else()
            set(FILE_CREATED_MESSAGE "${METATOMIC_SHARED_LIB_NAME}")
        endif()
    else()
        set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--crate-type=staticlib")
        set(CARGO_OUTPUTS ${METATOMIC_STATIC_LOCATION})
        set(FILE_CREATED_MESSAGE "${METATOMIC_STATIC_LIB_NAME}")
    endif()
endif()

add_custom_command(
    OUTPUT ${CARGO_OUTPUTS}
    COMMAND ${CMAKE_COMMAND} -E env ${CARGO_ENV}
        cargo rustc ${CARGO_BUILD_ARG} -- ${CARGO_RUSTC_ARGS}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS ${ALL_RUST_SOURCES}
    COMMENT "Building ${FILE_CREATED_MESSAGE} with cargo"
    VERBATIM
)
add_custom_target(cargo-build-metatomic ALL DEPENDS ${CARGO_OUTPUTS})

# Auto-generate a header containing the version number as #define
set(_path_ "${CMAKE_CURRENT_BINARY_DIR}/generated-version.h")
file(WRITE ${_path_} "#pragma once\n\n")
file(APPEND ${_path_} "/** Full version of metatomic as a string */\n")
file(APPEND ${_path_} "#define METATOMIC_VERSION \"${METATOMIC_FULL_VERSION}\"\n\n")
file(APPEND ${_path_} "/** Major version number of metatomic as an integer */\n")
file(APPEND ${_path_} "#define METATOMIC_VERSION_MAJOR ${PROJECT_VERSION_MAJOR}\n\n")
file(APPEND ${_path_} "/** Minor version number of metatomic as an integer */\n")
file(APPEND ${_path_} "#define METATOMIC_VERSION_MINOR ${PROJECT_VERSION_MINOR}\n\n")
file(APPEND ${_path_} "/** Patch version number of metatomic as an integer */\n")
file(APPEND ${_path_} "#define METATOMIC_VERSION_PATCH ${PROJECT_VERSION_PATCH}\n")

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include/metatomic)
set(_destination_ "${CMAKE_CURRENT_BINARY_DIR}/include/metatomic/version.h")
file(COPY_FILE ${_path_} ${_destination_} ONLY_IF_DIFFERENT)

add_dependencies(metatomic::shared cargo-build-metatomic)
add_dependencies(metatomic::static cargo-build-metatomic)

set_target_properties(metatomic::shared PROPERTIES
    IMPORTED_LOCATION ${METATOMIC_SHARED_LOCATION}
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_BINARY_DIR}/include"
    BUILD_VERSION "${METATOMIC_FULL_VERSION}"
)
target_compile_features(metatomic::shared INTERFACE cxx_std_17)

if (WIN32)
    set_target_properties(metatomic::shared PROPERTIES
        IMPORTED_IMPLIB ${METATOMIC_IMPLIB_LOCATION}
    )
endif()

set_target_properties(metatomic::static PROPERTIES
    IMPORTED_LOCATION ${METATOMIC_STATIC_LOCATION}
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_BINARY_DIR}/include"
    INTERFACE_LINK_LIBRARIES "${CARGO_DEFAULT_LIBRARIES}"
    BUILD_VERSION "${METATOMIC_FULL_VERSION}"
)
target_compile_features(metatomic::static INTERFACE cxx_std_17)

if (TARGET metatensor::static)
    target_link_libraries(metatomic::static INTERFACE metatensor::static)
else()
    target_link_libraries(metatomic::static INTERFACE metatensor)
endif()

if (TARGET metatensor::shared)
    target_link_libraries(metatomic::shared INTERFACE metatensor::shared)
else()
    target_link_libraries(metatomic::shared INTERFACE metatensor)
endif()

include(FetchContent)

# JSON library from https://github.com/nlohmann/json
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(nlohmann_json)

target_link_libraries(metatomic::shared INTERFACE nlohmann_json::nlohmann_json)
target_link_libraries(metatomic::static INTERFACE nlohmann_json::nlohmann_json)

if (BUILD_SHARED_LIBS)
    add_library(metatomic ALIAS metatomic::shared)
else()
    add_library(metatomic ALIAS metatomic::static)
endif()

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/metatomic-config.in.cmake
    ${PROJECT_BINARY_DIR}/metatomic-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metatomic
)
write_basic_package_version_file(
    metatomic-config-version.cmake
    VERSION ${METATOMIC_FULL_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(FILES "include/metatomic.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "include/metatomic.hpp" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY "include/metatomic" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/metatomic/version.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/metatomic)

if (METATOMIC_INSTALL_BOTH_STATIC_SHARED OR BUILD_SHARED_LIBS)
    if (WIN32)
        # DLL files should go in <prefix>/bin
        install(
            FILES ${METATOMIC_SHARED_LOCATION}
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_READ WORLD_EXECUTE
        )
        # .lib files should go in <prefix>/lib
        install(FILES ${METATOMIC_IMPLIB_LOCATION} DESTINATION ${CMAKE_INSTALL_LIBDIR})
    else()
        install(
            FILES ${METATOMIC_SHARED_LOCATION}
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_READ WORLD_EXECUTE
        )
    endif()
endif()

if (METATOMIC_INSTALL_BOTH_STATIC_SHARED OR NOT BUILD_SHARED_LIBS)
    install(FILES ${METATOMIC_STATIC_LOCATION} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

install(FILES
    ${PROJECT_BINARY_DIR}/metatomic-config-version.cmake
    ${PROJECT_BINARY_DIR}/metatomic-config.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metatomic
)
