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

.. _c-streams:

Streams
^^^^^^^

On devices with asynchronous execution (GPUs), :c:func:`mta_execute_model` takes
a ``stream`` parameter, used to order the work done by metatomic and by the
model with the work done by the caller. See :c:func:`mta_execute_model` for the
possible values of this parameter, and the guarantees given by metatomic.

:c:func:`mta_stream_synchronize` can be used to wait for all the work submitted
to a stream to finish, for example in models that can not submit their work
directly to the caller's stream, or before calling :c:func:`mta_system_create`
and :c:func:`mta_system_add_pairs`, which require all the work producing their
data to be complete.

.. doxygenfunction:: mta_stream_synchronize

--------------------------------------------------------------------------------

.. doxygenstruct:: mta_model_t
    :members:
