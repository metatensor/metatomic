# `LegacyModel` is an implementation detail of the plugin: this library is
# linked into the plugin and the C++ tests, and is not installed. It compiles
# its own copy of the DLPack conversions, which libmetatomic_torch does not
# export.
add_library(metatomic-torch-legacy STATIC
    legacy.cpp
    ../src/internal/dlpack.cpp
)
target_link_libraries(metatomic-torch-legacy PUBLIC metatomic_torch)
target_include_directories(metatomic-torch-legacy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(metatomic-torch-legacy PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

add_library(metatomic-torch-plugin SHARED plugin.cpp)
target_link_libraries(metatomic-torch-plugin PRIVATE metatomic-torch-legacy)

# `mta_plugin_init` is the only symbol the plugin has to export, and it is
# declared with `MTA_EXPORT`. Hiding everything else keeps `LegacyModel` out of
# the dynamic symbol table.
set_target_properties(metatomic-torch-plugin PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Use the same name on every platform to simplify loading the plugin
set_target_properties(metatomic-torch-plugin PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# same directory as the plugins installed by metatomic-core
if (WIN32)
    set(METATOMIC_TORCH_PLUGIN_SUBDIR "${CMAKE_INSTALL_BINDIR}")
else()
    set(METATOMIC_TORCH_PLUGIN_SUBDIR "${CMAKE_INSTALL_LIBEXECDIR}/metatomic")
endif()

if (UNIX)
    # Resolve libmetatomic_torch relative to the plugin install directory
    get_filename_component(_plugin_abs
        "${CMAKE_INSTALL_PREFIX}/${METATOMIC_TORCH_PLUGIN_SUBDIR}" ABSOLUTE
    )
    get_filename_component(_lib_abs
        "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" ABSOLUTE
    )
    file(RELATIVE_PATH _rpath_to_lib "${_plugin_abs}" "${_lib_abs}")
    if (_rpath_to_lib STREQUAL "")
        set(_rpath_to_lib ".")
    endif()

    if (APPLE)
        set(METATOMIC_TORCH_PLUGIN_RPATH "@loader_path/${_rpath_to_lib}")
    else()
        set(METATOMIC_TORCH_PLUGIN_RPATH "$ORIGIN/${_rpath_to_lib}")
    endif()

    set_target_properties(metatomic-torch-plugin PROPERTIES
        INSTALL_RPATH "${METATOMIC_TORCH_PLUGIN_RPATH}"
    )
endif()

install(
    TARGETS metatomic-torch-plugin
    LIBRARY DESTINATION ${METATOMIC_TORCH_PLUGIN_SUBDIR}
    RUNTIME DESTINATION ${METATOMIC_TORCH_PLUGIN_SUBDIR}
)
