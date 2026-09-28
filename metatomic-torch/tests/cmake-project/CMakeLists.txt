cmake_minimum_required(VERSION 3.22)

message(STATUS "Running with CMake version ${CMAKE_VERSION}")

project(metatomic-torch-test-cmake-project CXX)

option(USE_CMAKE_SUBDIRECTORY OFF)

if (USE_CMAKE_SUBDIRECTORY)
    message(STATUS "Using metatomic-torch with add_subdirectory")
    # build metatomic and metatomic_torch as part of this project
    add_subdirectory(../../../metatomic-core metatomic_core)
    add_subdirectory(../../ metatomic_torch)

    find_package(metatensor_torch)
    # load metatensor, metatensor-torch and metatomic from the build path
    set(CMAKE_BUILD_RPATH "$<TARGET_FILE_DIR:torch>;$<TARGET_FILE_DIR:metatensor::shared>;$<TARGET_FILE_DIR:metatensor_torch>;$<TARGET_FILE_DIR:metatomic::shared>")
else()
    message(STATUS "Using metatomic-torch with find_package")
    # If building a dev version, we also need to update the
    # REQUIRED_METATOMIC_TORCH_VERSION in the same way we update the
    # metatomic-torch version
    include(../../cmake/dev-versions.cmake)
    set(REQUIRED_METATOMIC_TORCH_VERSION "0.1.15")
    create_development_version("${REQUIRED_METATOMIC_TORCH_VERSION}" METATOMIC_TORCH_FULL_VERSION "metatomic-torch-v")
    string(REGEX REPLACE "([0-9]*)\\.([0-9]*).*" "\\1.\\2" REQUIRED_METATOMIC_TORCH_VERSION ${METATOMIC_TORCH_FULL_VERSION})

    # find metatomic_torch with find_package
    find_package(metatomic_torch ${REQUIRED_METATOMIC_TORCH_VERSION} CONFIG REQUIRED)
endif()

add_executable(torch-main main.cpp)
target_link_libraries(torch-main metatomic_torch)

enable_testing()
add_test(NAME torch-main COMMAND torch-main)

if(WIN32)
    # We need to set the path to allow access to the various DLL
    set_property(TEST torch-main PROPERTY
        ENVIRONMENT_MODIFICATION
            "PATH=path_list_append:$<TARGET_FILE_DIR:torch>"
            "PATH=path_list_append:$<TARGET_FILE_DIR:metatensor::shared>"
            "PATH=path_list_append:$<TARGET_FILE_DIR:metatensor_torch>"
            "PATH=path_list_append:$<TARGET_FILE_DIR:metatomic::shared>"
            "PATH=path_list_append:$<TARGET_FILE_DIR:metatomic_torch>"
    )
endif()
