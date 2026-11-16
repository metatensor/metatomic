Metadata
========

.. currentmodule:: metatomic_torch

The following classes are used to store metadata about a model.
:py:class:`Quantity` represent a physical quantity used as input or output of a
model; :py:class:`ModelCapabilities` stores information about what a model can
do; and :py:class:`ModelMetadata` stores meta-information about a model: list of
authors, references, *etc.*

.. autoclass:: Quantity
   :members:
   :special-members: __eq__, __ne__

.. autoclass:: ModelCapabilities
   :members:
   :special-members: __eq__, __ne__

.. autoclass:: ModelMetadata
   :members:
