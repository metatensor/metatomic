Information about models
------------------------

.. py:currentmodule:: metatomic.torch

Here are the classes that are used to store and use information about the
atomistic models.

- :py:class:`ModelMetadata` stores metadata about the model: name, authors,
  references, *etc.*
- :py:class:`ModelCapabilities` stores information about what a model can do.
  Part of this is the full set of outputs the model can produce, stored in
  :py:class:`ModelOutput`;
- :py:class:`ModelEvaluationOptions` is used by the simulation engine to request
  the model to do things. This is handled by
  :py:class:`AtomisticModel`, which transforms the arguments then given
  to :py:meth:`ModelInterface.forward`.

--------------------------------------------------------------------------------

.. autoclass:: metatomic.torch.ModelMetadata
    :members:

.. autoclass:: metatomic.torch.ModelOutput
    :members:

.. autoclass:: metatomic.torch.ModelCapabilities
    :members:

.. autoclass:: metatomic.torch.ModelEvaluationOptions
    :members:
