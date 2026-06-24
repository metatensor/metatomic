@PACKAGE_INIT@

cmake_minimum_required(VERSION 3.22)

include(CMakeFindDependencyMacro)
include(FindPackageHandleStandardArgs)

if(metatomic_FOUND)
    return()
endif()

enable_language(CXX)

# use the same version for metatensor-core as the main CMakeLists.txt
set(REQUIRED_METATENSOR_VERSION @REQUIRED_METATENSOR_VERSION@)
find_package(metatensor ${REQUIRED_METATENSOR_VERSION} CONFIG REQUIRED)

# Find nlohmann_json dependency using the installed module
include(${CMAKE_CURRENT_LIST_DIR}/nlohmann_json.cmake)

get_filename_component(METATOMIC_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/@PACKAGE_RELATIVE_PATH@" ABSOLUTE)

if (WIN32)
    set(METATOMIC_SHARED_LOCATION ${METATOMIC_PREFIX_DIR}/@CMAKE_INSTALL_BINDIR@/@METATOMIC_SHARED_LIB_NAME@)
    set(METATOMIC_IMPLIB_LOCATION ${METATOMIC_PREFIX_DIR}/@CMAKE_INSTALL_LIBDIR@/@METATOMIC_IMPLIB_NAME@)
else()
    set(METATOMIC_SHARED_LOCATION ${METATOMIC_PREFIX_DIR}/@CMAKE_INSTALL_LIBDIR@/@METATOMIC_SHARED_LIB_NAME@)
endif()

set(METATOMIC_STATIC_LOCATION ${METATOMIC_PREFIX_DIR}/@CMAKE_INSTALL_LIBDIR@/@METATOMIC_STATIC_LIB_NAME@)
set(METATOMIC_INCLUDE ${METATOMIC_PREFIX_DIR}/@CMAKE_INSTALL_INCLUDEDIR@/)

if (NOT EXISTS ${METATOMIC_INCLUDE}/metatomic.h OR NOT EXISTS ${METATOMIC_INCLUDE}/metatomic.hpp)
    message(FATAL_ERROR "could not find metatomic headers in '${METATOMIC_INCLUDE}', please re-install metatomic")
endif()


# Shared library target
if (@METATOMIC_INSTALL_BOTH_STATIC_SHARED@ OR @BUILD_SHARED_LIBS@)
    if (NOT EXISTS ${METATOMIC_SHARED_LOCATION})
        message(FATAL_ERROR "could not find metatomic library at '${METATOMIC_SHARED_LOCATION}', please re-install metatomic")
    endif()

    add_library(metatomic::shared SHARED IMPORTED)
    set_target_properties(metatomic::shared PROPERTIES
        IMPORTED_LOCATION ${METATOMIC_SHARED_LOCATION}
        INTERFACE_INCLUDE_DIRECTORIES ${METATOMIC_INCLUDE}
        BUILD_VERSION "@METATOMIC_FULL_VERSION@"
    )

    target_compile_features(metatomic::shared INTERFACE cxx_std_17)
    target_link_libraries(metatomic::shared INTERFACE metatensor nlohmann_json::nlohmann_json)

    if (WIN32)
        if (NOT EXISTS ${METATOMIC_IMPLIB_LOCATION})
            message(FATAL_ERROR "could not find metatomic library at '${METATOMIC_IMPLIB_LOCATION}', please re-install metatomic")
        endif()

        set_target_properties(metatomic::shared PROPERTIES
            IMPORTED_IMPLIB ${METATOMIC_IMPLIB_LOCATION}
        )
    endif()
endif()


# Static library target
if (@METATOMIC_INSTALL_BOTH_STATIC_SHARED@ OR NOT @BUILD_SHARED_LIBS@)
    if (NOT EXISTS ${METATOMIC_STATIC_LOCATION})
        message(FATAL_ERROR "could not find metatomic library at '${METATOMIC_STATIC_LOCATION}', please re-install metatomic")
    endif()

    add_library(metatomic::static STATIC IMPORTED)
    set_target_properties(metatomic::static PROPERTIES
        IMPORTED_LOCATION ${METATOMIC_STATIC_LOCATION}
        INTERFACE_INCLUDE_DIRECTORIES ${METATOMIC_INCLUDE}
        INTERFACE_LINK_LIBRARIES "@CARGO_DEFAULT_LIBRARIES@"
        BUILD_VERSION "@METATOMIC_FULL_VERSION@"
    )

    target_compile_features(metatomic::static INTERFACE cxx_std_17)
    target_link_libraries(metatomic::static INTERFACE metatensor nlohmann_json::nlohmann_json)
endif()

# Export either the shared or static library as the metatomic target
if (@BUILD_SHARED_LIBS@)
    add_library(metatomic ALIAS metatomic::shared)
else()
    add_library(metatomic ALIAS metatomic::static)
endif()


if (@BUILD_SHARED_LIBS@)
    find_package_handle_standard_args(metatomic DEFAULT_MSG METATOMIC_SHARED_LOCATION METATOMIC_INCLUDE)
else()
    find_package_handle_standard_args(metatomic DEFAULT_MSG METATOMIC_STATIC_LOCATION METATOMIC_INCLUDE)
endif()
