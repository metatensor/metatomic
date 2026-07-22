Systems
=======

.. autoclass:: metatomic.torch.System
    :members:

.. autoclass:: metatomic.torch.NeighborListOptions
    :members:
    :special-members: __eq__, __ne__

.. autofunction:: metatomic.torch.systems_to_torch

.. autoclass:: metatomic.torch.systems_to_torch.IntoSystem

.. autofunction:: metatomic.torch.register_autograd_neighbors

Serialization
-------------

Below are functions to load and save metatomic systems to disk. The
serialization format is based on numpy's ``.npz``.

.. autofunction:: metatomic.torch.save

.. autofunction:: metatomic.torch.load_system

.. autofunction:: metatomic.torch.save_buffer

.. autofunction:: metatomic.torch.load_system_buffer
