# This file allow the python module in metatomic-core to either use an
# externally-provided version of the shared metatomic library; or to build the
# code from source and bundle the shared library inside the wheel.
#
# The first case is used when distributing the code in conda (since we have a
# separate libmetatomic package), the second one is used everywhere else (for
# local development builds and for the PyPI distribution).

cmake_minimum_required(VERSION 3.22)
project(metatomic-python NONE)

option(METATOMIC_CORE_PYTHON_USE_EXTERNAL_LIB "Force the usage of an external version of metatomic-core" OFF)
set(METATOMIC_CORE_SOURCE_DIR "" CACHE PATH "Path to the sources of metatomic-core")

file(REMOVE ${CMAKE_INSTALL_PREFIX}/_external.py)

set(REQUIRED_METATOMIC_VERSION "0.1.0")
if(${METATOMIC_CORE_PYTHON_USE_EXTERNAL_LIB})
    # when building a source checkout, update version to include git information
    # this will not apply when building a sdist
    if (EXISTS ${CMAKE_SOURCE_DIR}/../../metatomic-core/cmake/dev-versions.cmake)
        include(${CMAKE_SOURCE_DIR}/../../metatomic-core/cmake/dev-versions.cmake)
        create_development_version("${REQUIRED_METATOMIC_VERSION}" REQUIRED_METATOMIC_VERSION "metatomic-core-v")
        # strip any -dev/-rc suffix on the version since find_package does not support it
        string(REGEX REPLACE "([0-9]*)\\.([0-9]*)\\.([0-9]*).*" "\\1.\\2.\\3" REQUIRED_METATOMIC_VERSION ${REQUIRED_METATOMIC_VERSION})
    endif()

    find_package(metatomic ${REQUIRED_METATOMIC_VERSION} REQUIRED)

    get_target_property(METATOMIC_LOCATION metatomic::shared LOCATION)
    message(STATUS "Using external metatomic-core v${metatomic_VERSION} at ${METATOMIC_LOCATION}")

    # Get the prefix to use as cmake_prefix_path when trying to load this
    # version of the library again
    get_filename_component(METATOMIC_PREFIX "${METATOMIC_LOCATION}" DIRECTORY)
    get_filename_component(METATOMIC_PREFIX "${METATOMIC_PREFIX}" DIRECTORY)

    file(WRITE ${CMAKE_INSTALL_PREFIX}/_external.py
        "EXTERNAL_METATOMIC_PATH = \"${METATOMIC_LOCATION}\"\n\n"
    )
    file(APPEND ${CMAKE_INSTALL_PREFIX}/_external.py
        "EXTERNAL_METATOMIC_PREFIX = \"${METATOMIC_PREFIX}\"\n"
    )

    install(CODE "message(STATUS \"nothing to install\")")
else()
    if ("${METATOMIC_CORE_SOURCE_DIR}" STREQUAL "")
        message(FATAL_ERROR
            "Missing METATOMIC_CORE_SOURCE_DIR, please specify where to \
            find the source code for metatomic-core"
        )
    endif()

    message(STATUS "Using internal metatomic-core from ${METATOMIC_CORE_SOURCE_DIR}")

    set(BUILD_SHARED_LIBS ON)
    set(METATOMIC_INSTALL_BOTH_STATIC_SHARED OFF)
    # strip dynamic library for smaller wheels to download/install
    set(EXTRA_RUST_FLAGS "-Cstrip=symbols")

    add_subdirectory("${METATOMIC_CORE_SOURCE_DIR}" metatomic-core)
endif()
