.. _c-api-core:

C API reference
===============


The C API of metatomic is defined in the ``metatomic.h`` header file. To build
code using this API, you'll need to include this header and link to
``libmetatomic.so`` / ``libmetatomic.dylib`` / ``metatomic.dll``. It is
recommended to link to the shared library version of metatomic to allow multiple
programs/plugins to all use the same library.

We provide CMake configuration files that can be used to link to the metatomic
library:

.. code-block:: cmake

    add_library(my_library SHARED my_library.c)
    add_executable(my_executable my_executable.c)

    find_package(metatomic REQUIRED)

    # this takes care of setting the include path and linking to the library.
    target_link_libraries(my_library PUBLIC metatomic)
    target_link_libraries(my_executable PUBLIC metatomic)

In metatomic, models are defined using :c:type:`mta_model_t`, systems are
represented by :c:type:`mta_system_t`, and plugins through
:c:type:`mta_plugin_t`. See the links below for more information on each of
these.

.. toctree::
    :maxdepth: 1

    system
    model
    plugin
    misc

.. toctree::
    :maxdepth: 1
    :hidden:

    CHANGELOG.md
