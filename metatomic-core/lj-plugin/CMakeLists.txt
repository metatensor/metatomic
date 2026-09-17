add_library(lj-plugin SHARED lennard_jones.cpp)
target_link_libraries(lj-plugin PRIVATE metatomic)
target_compile_features(lj-plugin PRIVATE cxx_std_17)

# Use the same name on every platform to simplify loading the plugin
set_target_properties(lj-plugin PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

if(NOT METATOMIC_PLUGINS_SUBDIR)
    message(FATAL_ERROR "METATOMIC_PLUGINS_SUBDIR must be set before adding lj-plugin")
endif()

if(APPLE OR UNIX)
    # Resolve libmetatomic relative to the plugin install directory, whatever
    # CMAKE_INSTALL_{LIB,LIBEXEC}DIR the user configured.
    get_filename_component(_lj_plugin_abs
        "${CMAKE_INSTALL_PREFIX}/${METATOMIC_PLUGINS_SUBDIR}" ABSOLUTE
    )
    get_filename_component(_lj_lib_abs
        "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" ABSOLUTE
    )
    file(RELATIVE_PATH _lj_rpath_to_lib "${_lj_plugin_abs}" "${_lj_lib_abs}")
    if(_lj_rpath_to_lib STREQUAL "")
        set(_lj_rpath_to_lib ".")
    endif()

    if(APPLE)
        set_target_properties(lj-plugin PROPERTIES
            INSTALL_RPATH "@loader_path/${_lj_rpath_to_lib}"
        )
    else()
        set_target_properties(lj-plugin PROPERTIES
            INSTALL_RPATH "$ORIGIN/${_lj_rpath_to_lib}"
        )
    endif()
endif()

if(MINGW)
    # tests/CMakeLists.txt statically links libstdc++ into executables, do the same here
    # so nobody has to search for libstdc++.dll (which is a pain to find on CI)
    target_link_options(lj-plugin PRIVATE -static-libstdc++)
endif()

install(
    TARGETS lj-plugin
    LIBRARY DESTINATION ${METATOMIC_PLUGINS_SUBDIR}
    RUNTIME DESTINATION ${METATOMIC_PLUGINS_SUBDIR}
)
