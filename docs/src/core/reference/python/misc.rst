Miscellaneous
=============

.. currentmodule:: metatomic

Error handling
^^^^^^^^^^^^^^

.. autoclass:: MetatomicError

Utilities
^^^^^^^^^

.. autodata:: metatomic.utils.cmake_prefix_path

Testing
^^^^^^^

Helpers for discovering plugins shipped with metatomic for engine integration
tests.

.. autofunction:: metatomic.testing.plugins_directory
.. autofunction:: metatomic.testing.plugin_path

A Lennard-Jones potential is also available as a NumPy model, which can either
be used directly from Python, or loaded by engines through the Python plugin.
In addition to the energy, this model provides most of the outputs engines might
want to support.

.. autoclass:: metatomic.testing.LennardJonesModel
