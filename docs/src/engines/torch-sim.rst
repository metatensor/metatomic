.. _engine-torch-sim:

torch-sim
=========

.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatomic supported?
   * - https://radical-ai.github.io/torch-sim/
     - Via the ``metatomic-torchsim`` package

How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

Install the integration package from PyPI:

.. code-block:: bash

   pip install metatomic-torchsim

This pulls in ``torch-sim-atomistic`` and ``metatomic-torch`` as dependencies.

For the full TorchSim documentation, see
https://radical-ai.github.io/torch-sim/.

Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

Only the :ref:`energy <energy-output>` output is supported. Forces and stresses
are derived via autograd.

How to use the code
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import ase.build
   import torch_sim as ts
   from metatomic.torchsim import MetatomicModel

   model = MetatomicModel("model.pt", device="cpu")

   atoms = ase.build.bulk("Si", "diamond", a=5.43, cubic=True)
   sim_state = ts.io.atoms_to_state([atoms], model.device, model.dtype)

   results = model(sim_state)
   print(results["energy"])   # shape [1]
   print(results["forces"])   # shape [n_atoms, 3]
   print(results["stress"])   # shape [1, 3, 3]

For more details, see the `metatomic-torchsim documentation
<https://docs.metatensor.org/metatomic/latest/torchsim/>`_.
