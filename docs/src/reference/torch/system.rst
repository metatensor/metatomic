Systems
=======

.. autoclass:: metatomic_torch.System
    :members:

.. autoclass:: metatomic_torch.PairListOptions
    :members:
    :special-members: __eq__, __ne__

.. autofunction:: metatomic_torch.systems_to_torch

.. autoclass:: metatomic_torch.IntoSystem

.. autofunction:: metatomic_torch.register_autograd_neighbors

Serialization
-------------

Below are functions to load and save metatomic systems to disk. The
serialization format is based on numpy's ``.npz``.

.. autofunction:: metatomic_torch.save

.. autofunction:: metatomic_torch.load_system

.. autofunction:: metatomic_torch.save_buffer

.. autofunction:: metatomic_torch.load_system_buffer
