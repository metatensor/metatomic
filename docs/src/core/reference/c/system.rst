System
======

.. doxygentypedef:: mta_system_t

The following functions operate on :c:type:`mta_system_t`:

- :c:func:`mta_system_create`: create a new system from types, positions, cell, and PBC data
- :c:func:`mta_system_free`: free a system handle
- :c:func:`mta_system_size`: get the number of atoms in a system
- :c:func:`mta_system_get_data`: get a borrowed DLPack tensor for some system data
- :c:func:`mta_system_get_length_unit`: get the length unit of a system
- :c:func:`mta_system_add_pairs`: add a pair list to a system
- :c:func:`mta_system_get_pairs`: get a borrowed view of a pair list from a system
- :c:func:`mta_system_known_pairs`: get all pair list options known by a system
- :c:func:`mta_system_add_custom_data`: add custom data to a system
- :c:func:`mta_system_get_custom_data`: get a borrowed view of custom data by name
- :c:func:`mta_system_known_custom_data`: get all custom data names known by a system

- :c:func:`mta_save`: save a system to a file
- :c:func:`mta_save_buffer`: save a system to a buffer
- :c:func:`mta_load`: load a system from a file
- :c:func:`mta_load_buffer`: load a system from a buffer

--------------------------------------------------------------------------------

.. doxygenfunction:: mta_system_create

.. doxygenfunction:: mta_system_free

.. doxygenfunction:: mta_system_size

.. doxygenfunction:: mta_system_get_data

.. doxygenfunction:: mta_system_get_length_unit

.. doxygenfunction:: mta_system_add_pairs

.. doxygenfunction:: mta_system_get_pairs

.. doxygenfunction:: mta_system_known_pairs

.. doxygenfunction:: mta_system_add_custom_data

.. doxygenfunction:: mta_system_get_custom_data

.. doxygenfunction:: mta_system_known_custom_data

.. doxygenfunction:: mta_save

.. doxygenfunction:: mta_save_buffer

.. doxygenfunction:: mta_load

.. doxygenfunction:: mta_load_buffer
