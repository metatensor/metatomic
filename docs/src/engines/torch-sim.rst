.. _engine-torch-sim:

torch-sim
=========

.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatomic supported?
   * - https://torchsim.github.io/torch-sim/
     - Via the ``metatomic-torchsim`` package

How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

Install the integration package from PyPI:

.. code-block:: bash

   pip install metatomic-torchsim

For the full TorchSim documentation, see https://torchsim.github.io/torch-sim/.

Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

Only the :ref:`energy <energy-output>` output is supported. Forces and stresses
are derived via autograd.

How to use the code
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import ase.build
   import torch_sim as ts
   from metatomic_torchsim import MetatomicModel

   model = MetatomicModel("model.pt", device="cpu")

   atoms = ase.build.bulk("Si", "diamond", a=5.43, cubic=True)
   sim_state = ts.initialize_state(atoms, device=model.device, dtype=model.dtype)

   results = model(sim_state)
   print(results["energy"])   # shape [1]
   print(results["forces"])   # shape [n_atoms, 3]
   print(results["stress"])   # shape [1, 3, 3]

API documentation
-----------------

.. autoclass:: metatomic_torchsim.MetatomicModel
    :show-inheritance:
    :members:
