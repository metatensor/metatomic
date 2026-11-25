add_library(metatomic-torch-legacy-plugin SHARED
    plugin.cpp
    legacy_model.cpp
)
target_link_libraries(metatomic-torch-legacy-plugin PUBLIC metatomic_torch)

set_target_properties(metatomic-torch-legacy-plugin PROPERTIES
    # `mta_plugin_init` is the only symbol the plugin has to export, and it is
    # declared with `MTA_EXPORT`
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    # Use the same name on every platform to simplify loading the plugin
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

    set_target_properties(metatomic-torch-legacy-plugin PROPERTIES
        INSTALL_RPATH "${METATOMIC_TORCH_PLUGIN_RPATH}"
    )
endif()

install(
    TARGETS metatomic-torch-legacy-plugin
    LIBRARY DESTINATION ${METATOMIC_TORCH_PLUGIN_SUBDIR}
    RUNTIME DESTINATION ${METATOMIC_TORCH_PLUGIN_SUBDIR}
)
