Plugin system
=============

In metatomic, models are provided by plugins. A plugin can provide a single
purpose model (i.e. Lennard-Jones for a Cu-H system), a familly of models all
sharing similar architecture (i.e. any EAM parametrization), or even any model
implemented in a given programming language (such as the Python plugin for
models defined as Python scripts).

A plugin is created with the :c:type:`MTA_REGISTER_PLUGIN` macro constructing a
:c:type:`mta_plugin_t`, see :ref:`the corresponding tutorial
<c-tutorial-plugin>` for more information.

.. doxygenstruct:: mta_plugin_t
    :members:

.. doxygendefine:: MTA_REGISTER_PLUGIN

--------------------------------------------------------------------------------

Plugins are typically distributed as shared libraries and should be loaded by
simulation engines with :c:func:`mta_load_plugin`. Loading a plugin makes all the
models it supports available in :c:func:`mta_load_model`.

.. doxygenfunction:: mta_load_plugin
