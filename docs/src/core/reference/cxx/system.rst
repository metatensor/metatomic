System
======

.. doxygenclass:: metatomic::System
   :members:

.. doxygenclass:: metatomic::PairListOptions
   :members:

The following functions save and load :cpp:class:`metatomic::System` objects:

- :cpp:func:`metatomic::io::save`: save a system to a file
- :cpp:func:`metatomic::io::save_buffer`: save a system to a buffer
- :cpp:func:`metatomic::io::load`: load a system from a file
- :cpp:func:`metatomic::io::load_buffer`: load a system from a buffer

--------------------------------------------------------------------------------

.. doxygenfunction:: metatomic::io::save(const std::string& path, const System& system)

.. doxygenfunction:: metatomic::io::save_buffer(const System& system)

.. doxygenfunction:: metatomic::io::load(const std::string& path, mts_create_array_callback_t create_array)

.. doxygenfunction:: metatomic::io::load_buffer(const uint8_t* buffer, uintptr_t buffer_count, mts_create_array_callback_t create_array)

.. doxygenfunction:: metatomic::io::load_buffer(const Buffer& buffer, mts_create_array_callback_t create_array)
