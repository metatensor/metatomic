include(CMakeFindDependencyMacro)

# use the same version for metatensor-torch as the main CMakeLists.txt
set(REQUIRED_METATENSOR_TORCH_VERSION @REQUIRED_METATENSOR_TORCH_VERSION@)
find_package(metatensor_torch ${REQUIRED_METATENSOR_TORCH_VERSION} CONFIG REQUIRED)

# We can only load metatomic_torch with the same minor version of Torch that
# was used to compile it (and is stored in BUILD_TORCH_VERSION)
set(BUILD_TORCH_VERSION @Torch_VERSION@)
set(BUILD_TORCH_MAJOR @Torch_VERSION_MAJOR@)
set(BUILD_TORCH_MINOR @Torch_VERSION_MINOR@)

find_package(Torch "${BUILD_TORCH_MAJOR}.${BUILD_TORCH_MINOR}" REQUIRED)

if (NOT "${BUILD_TORCH_MAJOR}" STREQUAL "${Torch_VERSION_MAJOR}")
    message(FATAL_ERROR "found incompatible torch version: metatomic-torch was built against v${BUILD_TORCH_VERSION} but we found v${Torch_VERSION}")
endif()

if (NOT "${BUILD_TORCH_MINOR}" STREQUAL "${Torch_VERSION_MINOR}")
    message(FATAL_ERROR "found incompatible torch version: metatomic-torch was built against v${BUILD_TORCH_VERSION} but we found v${Torch_VERSION}")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/metatomic_torch-targets.cmake)
