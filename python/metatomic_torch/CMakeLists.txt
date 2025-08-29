# This file allow the python module in metatomic-torch to either use an
# externally-provided version of the shared metatomic_torch library; or to build
# the code from source and bundle the shared library inside the wheel.
#
# The first case is used when distributing the code in conda (since we have a
# separate libmetatomic_torch package), the second one is used everywhere else
# (for local development builds and for the PyPI distribution).

cmake_minimum_required(VERSION 3.16)
project(metatomic-torch-python CXX)

option(METATOMIC_TORCH_PYTHON_USE_EXTERNAL_LIB "Force the usage of an external version of metatomic-torch" OFF)
set(METATOMIC_TORCH_SOURCE_DIR "" CACHE PATH "Path to the sources of metatomic-torch")

file(REMOVE ${CMAKE_INSTALL_PREFIX}/_external.py)

set(REQUIRED_METATOMIC_TORCH_VERSION "0.1.0")
if(${METATOMIC_TORCH_PYTHON_USE_EXTERNAL_LIB})
    # when building a source checkout, update version to include git information
    # this will not apply when building a sdist
    if (EXISTS ${CMAKE_SOURCE_DIR}/../../metatomic-torch/cmake/dev-versions.cmake)
        include(${CMAKE_SOURCE_DIR}/../../metatomic-torch/cmake/dev-versions.cmake)
        create_development_version(
            "${REQUIRED_METATOMIC_TORCH_VERSION}"
            REQUIRED_METATOMIC_TORCH_VERSION
            "metatomic-torch-v"
        )
    endif()

    # strip any -dev/-rc suffix on the version since find_package does not support it
    string(
        REGEX REPLACE "([0-9]*)\\.([0-9]*)\\.([0-9]*).*" "\\1.\\2.\\3"
        REQUIRED_METATOMIC_TORCH_VERSION
        ${REQUIRED_METATOMIC_TORCH_VERSION}
    )
    find_package(metatomic_torch ${REQUIRED_METATOMIC_TORCH_VERSION} REQUIRED)

    get_target_property(METATOMIC_TORCH_LOCATION metatomic_torch LOCATION)
    message(STATUS "Using external metatomic-torch v${metatomic_torch_VERSION} at ${METATOMIC_TORCH_LOCATION}")

    # Get the prefix to use as cmake_prefix_path when trying to load this
    # version of the library again
    get_filename_component(METATOMIC_TORCH_PREFIX "${METATOMIC_TORCH_LOCATION}" DIRECTORY)
    get_filename_component(METATOMIC_TORCH_PREFIX "${METATOMIC_TORCH_PREFIX}" DIRECTORY)

    file(WRITE ${CMAKE_INSTALL_PREFIX}/_external.py
        "EXTERNAL_METATOMIC_TORCH_PATH = \"${METATOMIC_TORCH_LOCATION}\"\n\n"
    )
    file(APPEND ${CMAKE_INSTALL_PREFIX}/_external.py
        "EXTERNAL_METATOMIC_TORCH_PREFIX = \"${METATOMIC_TORCH_PREFIX}\"\n"
    )

    install(CODE "message(STATUS \"nothing to install\")")
else()
    if ("${METATOMIC_TORCH_SOURCE_DIR}" STREQUAL "")
        message(FATAL_ERROR
            "Missing METATOMIC_TORCH_SOURCE_DIR, please specify where to \
            find the source code for metatomic-torch"
        )
    endif()

    message(STATUS "Using internal metatomic-torch from ${METATOMIC_TORCH_SOURCE_DIR}")

    add_subdirectory("${METATOMIC_TORCH_SOURCE_DIR}" metatomic-torch)


    if (LINUX OR APPLE)
        if (LINUX)
            set(rpath_origin "$ORIGIN")
        elseif(APPLE)
            set(rpath_origin "@loader_path")
        endif()
        find_package(Torch)

        set(metatomic_install_rpath "${CMAKE_INSTALL_RPATH}")
        # when loading the libraries from a Python installation:
        # - $ORIGIN/../../../../torch/lib is where libtorch.so will be
        # - $ORIGIN/../../../../metatensor/lib is where libmetatensor.so will be
        # - $ORIGIN/../../../../metatensor/torch/torch-x.y/lib is where libmetatensor_torch.so will be
        set(metatomic_install_rpath "${metatomic_install_rpath};${rpath_origin}/../../../../torch/lib")
        set(metatomic_install_rpath "${metatomic_install_rpath};${rpath_origin}/../../../../metatensor/lib")
        set(metatomic_install_rpath "${metatomic_install_rpath};${rpath_origin}/../../../../metatensor/torch/torch-${Torch_VERSION_MAJOR}.${Torch_VERSION_MINOR}/lib")

        set_target_properties(
            metatomic_torch PROPERTIES INSTALL_RPATH "${metatomic_install_rpath}"
        )
    endif()
endif()
