add_library(lj-plugin SHARED lennard_jones.cpp)
target_link_libraries(lj-plugin PRIVATE metatomic)
target_compile_features(lj-plugin PRIVATE cxx_std_17)
# Imported metatomic uses SYSTEM includes; keep header errors visible here.
set_target_properties(lj-plugin PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)

# Use the same name on every platform to simplify loading the plugin
set_target_properties(lj-plugin PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# plugin lives in libexec/metatomic/; libmetatomic is in lib/ (or lib/<triplet>)
if(APPLE)
    set_target_properties(lj-plugin PROPERTIES
        INSTALL_RPATH "@loader_path/../../${CMAKE_INSTALL_LIBDIR}"
    )
elseif(UNIX)
    set_target_properties(lj-plugin PROPERTIES
        INSTALL_RPATH "$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}"
    )
endif()

if(MINGW)
    # tests/CMakeLists.txt only statically links libstdc++ into executables.
    # This plugin is a shared library, so MinGW CI still cannot find
    # libstdc++.dll at load time unless the plugin itself is linked statically.
    target_link_options(lj-plugin PRIVATE -static-libstdc++)
endif()

# Windows has no rpath, so sit next to metatomic.dll. Unix: libexec + rpath above.
if(WIN32)
    set(_lj_plugin_dest ${CMAKE_INSTALL_BINDIR})
else()
    set(_lj_plugin_dest ${CMAKE_INSTALL_LIBEXECDIR}/metatomic)
endif()

install(
    TARGETS lj-plugin
    LIBRARY DESTINATION ${_lj_plugin_dest}
    RUNTIME DESTINATION ${_lj_plugin_dest}
)
