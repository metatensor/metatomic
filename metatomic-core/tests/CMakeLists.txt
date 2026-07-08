cmake_minimum_required(VERSION 3.22)
project(metatomic-tests)

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: debug or release"
        FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug)
    endif()
endif()

if (MINGW)
    # CI can't find libsdc++, so we statically link it
    set(CMAKE_EXE_LINKER_FLAGS "-static-libstdc++")
endif()

add_subdirectory(../ metatomic)
get_target_property(METATOMIC_IMPORTED_LOCATION metatomic::shared IMPORTED_LOCATION)
get_filename_component(METATOMIC_DIR ${METATOMIC_IMPORTED_LOCATION} DIRECTORY)

add_subdirectory(external)

find_program(VALGRIND valgrind)
if (VALGRIND)
    if (NOT "$ENV{METATOMIC_DISABLE_VALGRIND}" EQUAL "1")
        message(STATUS "Running tests using valgrind")
        set(TEST_COMMAND
            "${VALGRIND}" "--tool=memcheck" "--dsymutil=yes" "--error-exitcode=125"
            "--leak-check=full" "--show-leak-kinds=definite,indirect,possible" "--track-origins=yes"
            "--gen-suppressions=all"
        )
    endif()
else()
    set(TEST_COMMAND "")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat-pedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-weak-vtables")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-float-equal")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-prototypes")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shadow-uncaptured-local")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-padded")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unsafe-buffer-usage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-poison-system-directories")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-allocator-wrappers")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default")
endif()


enable_testing()
add_subdirectory(test-plugins)

if (TARGET metatensor::shared)
    get_target_property(METATENSOR_LOCATION metatensor::shared IMPORTED_LOCATION)
    get_filename_component(METATENSOR_DIR ${METATENSOR_LOCATION} DIRECTORY)
elseif (TARGET metatensor)
    get_target_property(METATENSOR_LOCATION metatensor LOCATION)
    get_filename_component(METATENSOR_DIR ${METATENSOR_LOCATION} DIRECTORY)
else()
    set(METATENSOR_DIR "")
endif()

function(metatomic_add_test source target)
    add_executable(${target} ${source})
    target_link_libraries(${target} metatomic catch)

    set_target_properties(${target} PROPERTIES
        # Ensure that the binaries find the right shared library.
        #
        # Without this, when configuring with cmake before the library is built,
        # cmake does not find the library on the filesystem and does not add the
        # RPATH to executables linking to it
        BUILD_RPATH ${METATOMIC_DIR}
        NO_SYSTEM_FROM_IMPORTED ON
    )

    target_compile_definitions(${target} PRIVATE PLUGIN_DIR="$<TARGET_FILE_DIR:test-c-plugin>")

    add_test(
        NAME ${target}
        COMMAND ${TEST_COMMAND} $<TARGET_FILE:${target}>
    )

    if(WIN32)
        # We need to set the path to allow access to metatomic.dll and
        # metatensor.dll. This does a similar job to the BUILD_RPATH above.
        STRING(REPLACE ";" "\\;" PATH_STRING "$ENV{PATH}")
        set_tests_properties(${target} PROPERTIES
            ENVIRONMENT "PATH=${PATH_STRING}\;$<TARGET_FILE_DIR:metatomic>\;${METATENSOR_DIR}"
        )
    endif()
endfunction()

file(GLOB ALL_TESTS *.cpp)
foreach(_file_ ${ALL_TESTS})
    get_filename_component(_name_ ${_file_} NAME_WE)
    metatomic_add_test(${_file_} ${_name_})
endforeach()

file(GLOB ALL_CPP_TESTS cxx/*.cpp)
foreach(_file_ ${ALL_CPP_TESTS})
    get_filename_component(_name_ ${_file_} NAME_WE)
    metatomic_add_test(${_file_} "cxx-${_name_}")
endforeach()
