Defining new models
===================

.. currentmodule:: metatomic

Custom Python models can be defined by subclassing :py:class:`BaseModel` and
implementing its abstract methods. Most of the time, it is easier to subclass
one of the following classes, which provide a more convenient interface to
define models based on a specific framework:

- :py:class:`NumpyModel` for models that are implemented with `NumPy`_.


.. _NumPy: https://numpy.org/

.. autoclass:: NumpyModel
   :inherited-members:

--------------------------------------------------------------------------------

.. autoclass:: BaseModel
   :members:
