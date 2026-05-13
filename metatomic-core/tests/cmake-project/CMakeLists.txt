cmake_minimum_required(VERSION 3.22)

message(STATUS "Running with CMake version ${CMAKE_VERSION}")

project(metatomic-test-cmake-project C CXX)

option(USE_CMAKE_SUBDIRECTORY OFF)

if (MINGW)
    # CI can't find libsdc++, so we statically link it
    set(CMAKE_EXE_LINKER_FLAGS "-static-libstdc++")
endif()


if (USE_CMAKE_SUBDIRECTORY)
    message(STATUS "Using metatomic with add_subdirectory")
    # build metatomic as part of this project
    add_subdirectory(../../ metatomic)

    # load metatomic from the build path
    set(CMAKE_BUILD_RPATH "$<TARGET_FILE_DIR:metatomic::shared>")
else()
    message(STATUS "Using metatomic with find_package")
    # If building a dev version, we also need to update the REQUIRED_METATOMIC_VERSION
    # in the same way we update the metatomic-torch version
    include(../../cmake/dev-versions.cmake)
    set(REQUIRED_METATOMIC_VERSION "0.1.0")
    create_development_version("${REQUIRED_METATOMIC_VERSION}" METATOMIC_CORE_FULL_VERSION "metatomic-core-v")
    string(REGEX REPLACE "([0-9]*)\\.([0-9]*).*" "\\1.\\2" REQUIRED_METATOMIC_VERSION ${METATOMIC_CORE_FULL_VERSION})

    find_package(metatomic ${REQUIRED_METATOMIC_VERSION} REQUIRED)

    if(TARGET metatomic::shared)
        get_target_property(mta_build_version metatomic::shared BUILD_VERSION)
        if (NOT ${mta_build_version} STREQUAL ${METATOMIC_CORE_FULL_VERSION})
            message(FATAL_ERROR "Invalid BUILD_VERSION for metatomic::shared, expected ${METATOMIC_CORE_FULL_VERSION} but got ${mta_build_version}")
        endif()
    endif()

    if(TARGET metatomic::static)
        get_target_property(mta_build_version metatomic::static BUILD_VERSION)
        if (NOT ${mta_build_version} STREQUAL ${METATOMIC_CORE_FULL_VERSION})
            message(FATAL_ERROR "Invalid BUILD_VERSION for metatomic::static, expected ${METATOMIC_CORE_FULL_VERSION} but got ${mta_build_version}")
        endif()
    endif()
endif()

enable_testing()


if(TARGET metatomic::shared)
    add_executable(c-main src/main.c)
    target_link_libraries(c-main metatomic::shared)

    add_executable(cxx-main src/main.cpp)
    target_link_libraries(cxx-main metatomic::shared)

    add_test(NAME c-main COMMAND c-main)
    add_test(NAME cxx-main COMMAND cxx-main)

    if(WIN32)
        # We need to set the path to allow access to metatomic.dll
        STRING(REPLACE ";" "\\;" PATH_STRING "$ENV{PATH}")
        set_tests_properties(c-main PROPERTIES
            ENVIRONMENT "PATH=${PATH_STRING}\;$<TARGET_FILE_DIR:metatomic::shared>"
        )

        set_tests_properties(cxx-main PROPERTIES
            ENVIRONMENT "PATH=${PATH_STRING}\;$<TARGET_FILE_DIR:metatomic::shared>"
        )
    endif()
endif()


if(TARGET metatomic::static)
    add_executable(c-main-static src/main.c)
    target_link_libraries(c-main-static metatomic::static)

    add_executable(cxx-main-static src/main.cpp)
    target_link_libraries(cxx-main-static metatomic::static)

    add_test(NAME c-main-static COMMAND c-main-static)
    add_test(NAME cxx-main-static COMMAND cxx-main-static)
endif()
