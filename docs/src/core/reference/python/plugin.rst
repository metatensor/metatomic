Plugin system
=============

.. currentmodule:: metatomic

Plugins are shared libraries that know how to load one or more kinds of models.
A plugin must be loaded with :py:func:`load_plugin` before models can be loaded
from it with :py:func:`load_model`.

.. autofunction:: load_plugin

.. autofunction:: load_model
