Executing models
================

.. currentmodule:: metatomic

In metatomic, models can be defined in Python or in other programing languages.
It migth be required to load a plugin with :py:func:`load_plugin` to access a
specific set of models. Models should then be loaded with :py:func:`load_model`,
which returns an :py:class:`ExternalModel` object. If a model is defined in
Python, it will be a subclass of :py:class:`BaseModel`, and can be used in the
same way as an :py:class:`ExternalModel`.

All models, Python or not, should be executed with :py:func:`execute_model`
which takes care of unit conversions and can check the inputs and outputs of the
model for consistency.


.. autofunction:: load_plugin

.. autofunction:: load_model

.. autofunction:: execute_model

.. autoclass:: ExternalModel
