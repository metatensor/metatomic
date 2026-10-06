Models
======

.. currentmodule:: metatomic

Custom Python models can be defined by subclassing :py:class:`BaseModel` and
implementing its abstract methods. Models defined in other languages and loaded
from a plugin through :py:func:`load_model` are wrapped in an
:py:class:`ExternalModel`, which exposes the same interface.

All models should be executed with :py:func:`execute_model`, which takes care of
unit conversions and can check the inputs and outputs of the model for
consistency.

.. autoclass:: BaseModel
   :members:

.. autoclass:: ExternalModel

.. autofunction:: execute_model
