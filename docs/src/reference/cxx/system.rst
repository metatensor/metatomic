System
======

.. doxygenclass:: metatomic::System
   :members:

.. doxygenclass:: metatomic::PairListOptions
   :members:

Serialization
-------------

Systems can be saved to a file or serialized into an in-memory byte buffer.
Loading a system requires an array-creation callback, which allocates the
arrays of the reconstructed system.

.. doxygenfunction:: metatomic::io::save

.. doxygenfunction:: metatomic::io::save_buffer

.. doxygenfunction:: metatomic::io::load

.. doxygenfunction:: metatomic::io::load_buffer(const uint8_t* buffer, uintptr_t buffer_count, mts_create_array_callback_t create_array)

.. doxygenfunction:: metatomic::io::load_buffer(const Buffer& buffer, mts_create_array_callback_t create_array)
