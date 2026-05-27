Miscellaneous
=============

Version number
^^^^^^^^^^^^^^

.. doxygenfunction:: mta_version

.. c:macro:: METATOMIC_VERSION

    Macro containing the compile-time version of metatomic, as a string

.. c:macro:: METATOMIC_VERSION_MAJOR

    Macro containing the compile-time **major** version number of metatomic, as
    an integer

.. c:macro:: METATOMIC_VERSION_MINOR

    Macro containing the compile-time **minor** version number of metatomic, as
    an integer

.. c:macro:: METATOMIC_VERSION_PATCH

    Macro containing the compile-time **patch** version number of metatomic, as
    an integer


Error handling
^^^^^^^^^^^^^^

.. doxygenfunction:: mta_last_error

.. doxygenfunction:: mta_set_last_error

.. doxygenenum:: mta_status_t


String manipulation
^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: mta_string_t

.. doxygenfunction:: mta_string_create

.. doxygenfunction:: mta_string_free

.. doxygenfunction:: mta_string_view

.. doxygenfunction:: mta_format_metadata


Unit conversion
^^^^^^^^^^^^^^^

.. doxygenfunction:: mta_unit_conversion_factor
