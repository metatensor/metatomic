# Plugin loading models defined in Python scripts. This plugin does not link to
# libpython, and can be used with any Python version.

enable_language(CXX)
include(GNUInstallDirs)

if (NOT TARGET metatomic::shared)
    message(FATAL_ERROR "metatomic::shared target must be defined before adding python-plugin")
endif()

# Embed find-python.py inside the plugin, as an array of bytes
set(FIND_PYTHON_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/find-python.py")
set(FIND_PYTHON_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated/find-python.py.inc")
add_custom_command(
    OUTPUT ${FIND_PYTHON_GENERATED}
    COMMAND ${CMAKE_COMMAND}
        -DINPUT_FILE=${FIND_PYTHON_SOURCE}
        -DOUTPUT_FILE=${FIND_PYTHON_GENERATED}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/make_hex_array.cmake
    DEPENDS ${FIND_PYTHON_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/make_hex_array.cmake
    COMMENT "Generating ${FIND_PYTHON_GENERATED}"
)

add_library(metatomic-python-plugin SHARED python_plugin.cpp ${FIND_PYTHON_GENERATED})
target_link_libraries(metatomic-python-plugin PRIVATE metatomic::shared)
target_include_directories(metatomic-python-plugin PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_features(metatomic-python-plugin PRIVATE cxx_std_17)

# Only export `mta_plugin_init` from the plugin
set_target_properties(metatomic-python-plugin PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# This plugin is loaded inside arbitrary engines, which might use a different
# version of the C++ standard library. We link statically to the C++ standard
# library to avoid any conflict, while still using the shared C runtime (which
# is shared with Python, and required to use the same environment variables).
#
# On macOS, libc++ is part of the OS and can not be linked statically.
if (MSVC)
    # Use the static C++ runtime and vcruntime, but the shared UCRT.
    set_target_properties(metatomic-python-plugin PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )
    target_link_options(metatomic-python-plugin PRIVATE
        "/NODEFAULTLIB:libucrt$<$<CONFIG:Debug>:d>.lib"
        "/DEFAULTLIB:ucrt$<$<CONFIG:Debug>:d>.lib"
    )
elseif (MINGW OR (UNIX AND NOT APPLE))
    include(CheckLinkerFlag)
    check_linker_flag(CXX "-static-libstdc++" METATOMIC_PYTHON_PLUGIN_STATIC_LIBSTDCXX)
    check_linker_flag(CXX "-static-libgcc" METATOMIC_PYTHON_PLUGIN_STATIC_LIBGCC)

    if (METATOMIC_PYTHON_PLUGIN_STATIC_LIBSTDCXX)
        target_link_options(metatomic-python-plugin PRIVATE -static-libstdc++)

        if (NOT MINGW)
            # don't re-export the symbols from the static C++ standard library
            check_linker_flag(CXX "LINKER:--exclude-libs,ALL" METATOMIC_PYTHON_PLUGIN_EXCLUDE_LIBS)
            if (METATOMIC_PYTHON_PLUGIN_EXCLUDE_LIBS)
                target_link_options(metatomic-python-plugin PRIVATE "LINKER:--exclude-libs,ALL")
            endif()
        endif()
    else()
        message(WARNING
            "can not link statically to the C++ standard library, the Python "
            "plugin will depend on the C++ standard library of the system"
        )
    endif()

    if (METATOMIC_PYTHON_PLUGIN_STATIC_LIBGCC)
        target_link_options(metatomic-python-plugin PRIVATE -static-libgcc)
    endif()
endif()

if (WIN32)
    target_link_libraries(metatomic-python-plugin PRIVATE psapi)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(metatomic-python-plugin PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
endif()

# The plugin only uses libmetatomic, don't link to libraries coming from the
# interface of metatomic::shared (i.e. libmetatensor)
if (APPLE)
    target_link_options(metatomic-python-plugin PRIVATE "LINKER:-dead_strip_dylibs")
elseif (UNIX)
    target_link_options(metatomic-python-plugin PRIVATE "LINKER:--as-needed")
endif()

# Use the same name on every platform to simplify loading the plugin
set_target_properties(metatomic-python-plugin PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

if (WIN32)
    set(PYTHON_PLUGIN_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
else()
    set(PYTHON_PLUGIN_INSTALL_DIR "${CMAKE_INSTALL_LIBEXECDIR}/metatomic")
endif()

if (APPLE OR UNIX)
    # Find libmetatomic, either next to the plugin (in the wheel) or in the
    # external installation. This might need to change if we ever want to
    # distribute the plugin as a standalone package.
    if (METATOMIC_CORE_PYTHON_USE_EXTERNAL_LIB)
        get_target_property(_metatomic_location metatomic::shared LOCATION)
        get_filename_component(_metatomic_lib_dir "${_metatomic_location}" DIRECTORY)
        set(_python_plugin_rpath "${_metatomic_lib_dir}")
    else()
        get_filename_component(_plugin_abs
            "${CMAKE_INSTALL_PREFIX}/${PYTHON_PLUGIN_INSTALL_DIR}" ABSOLUTE
        )
        get_filename_component(_lib_abs
            "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" ABSOLUTE
        )
        file(RELATIVE_PATH _rpath_to_lib "${_plugin_abs}" "${_lib_abs}")
        if (APPLE)
            set(_python_plugin_rpath "@loader_path/${_rpath_to_lib}")
        else()
            set(_python_plugin_rpath "$ORIGIN/${_rpath_to_lib}")
        endif()
    endif()

    set_target_properties(metatomic-python-plugin PROPERTIES
        INSTALL_RPATH "${_python_plugin_rpath}"
    )
endif()

install(
    TARGETS metatomic-python-plugin
    LIBRARY DESTINATION ${PYTHON_PLUGIN_INSTALL_DIR}
    RUNTIME DESTINATION ${PYTHON_PLUGIN_INSTALL_DIR}
)
