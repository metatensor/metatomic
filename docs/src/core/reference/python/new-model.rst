Defining new models
===================

.. currentmodule:: metatomic

Custom Python models can be defined by subclassing :py:class:`BaseModel` and
implementing its abstract methods. Most of the time, it is easier to subclass
one of the following classes, which provide a more convenient interface to
define models based on a specific framework:

- :py:class:`NumpyModel` for models that are implemented with `NumPy`_;
- :py:class:`TorchModel` for models that are implemented with `PyTorch`_.


.. _NumPy: https://numpy.org/
.. _PyTorch: https://pytorch.org/

.. autoclass:: NumpyModel
   :inherited-members:

.. autoclass:: TorchModel
   :members:
   :inherited-members: Module

--------------------------------------------------------------------------------

.. autoclass:: BaseModel
   :members:
