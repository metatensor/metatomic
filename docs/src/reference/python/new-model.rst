Defining new models
===================

.. currentmodule:: metatomic

Custom Python models can be defined by subclassing :py:class:`BaseModel` and
implementing its abstract methods. Most of the time, it is easier to subclass
one of the following classes, which provide a more convenient interface to
define models based on a specific framework:

- :py:class:`NumpyModel` for models that are implemented with `NumPy`_;
- :py:class:`TorchModel` for models that are implemented with `PyTorch`_.

The systems given to a model can share their data with the engine, which
modifies it between calls. A model that keeps some of this data from one call
to the next (for example the previous positions) must make its own copy, see
:py:class:`System`.


.. _NumPy: https://numpy.org/
.. _PyTorch: https://pytorch.org/

.. autoclass:: NumpyModel
   :members:
   :inherited-members:

.. autoclass:: TorchModel
   :members:
   :inherited-members: Module

--------------------------------------------------------------------------------

.. autoclass:: BaseModel
   :members:
