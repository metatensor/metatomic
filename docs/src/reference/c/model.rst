Model
=====

Models in metatomic are represented by a :c:type:`mta_model_t`, which implements
a `virtual table`_, storing together the data of the model and functions that
operate on this data to implement the model interface.

.. _virtual table: https://en.wikipedia.org/wiki/Virtual_method_table

Models are typically provided by a plugin (see :c:type:`mta_plugin_t`), and
should be loaded with :c:func:`mta_load_model` and executed through
:c:func:`mta_execute_model`. See :ref:`this tutorial <c-tutorial-model>` for an
example model implemented in C.

--------------------------------------------------------------------------------

.. doxygenfunction:: mta_load_model

.. doxygenfunction:: mta_execute_model

--------------------------------------------------------------------------------

.. doxygenstruct:: mta_model_t
    :members:
