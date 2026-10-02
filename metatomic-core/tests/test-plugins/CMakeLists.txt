# create test plugins with a consistent name across platforms
# .so suffix is used for all the platforms

add_library(test-c-plugin SHARED plugin.c)
target_link_libraries(test-c-plugin metatomic)
set_target_properties(test-c-plugin PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

add_library(bad-abi-plugin SHARED bad-abi.c)
target_link_libraries(bad-abi-plugin metatomic)
set_target_properties(bad-abi-plugin PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

add_library(test-cxx-plugin SHARED cxx-plugin.cpp)
target_link_libraries(test-cxx-plugin metatomic)
set_target_properties(test-cxx-plugin PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)
