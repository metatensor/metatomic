.. _torchsim-getting-started:

Getting started
===============

This tutorial walks through running a short NVE molecular dynamics
simulation with a metatomic model and TorchSim.

Prerequisites
-------------

Install the package and its dependencies:

.. code-block:: bash

   pip install metatomic-torchsim

Load the model
--------------

.. code-block:: python

   from metatomic_torchsim import MetatomicModel

   model = MetatomicModel("path/to/model.pt", device="cpu")

The wrapper detects the model's dtype and supported devices
automatically. Pass ``device="cuda"`` to run on GPU.

Build a simulation state
------------------------

TorchSim works with ``SimState`` objects. Convert ASE ``Atoms`` using
``torch_sim.io.atoms_to_state``:

.. code-block:: python

   import ase.build
   import torch_sim as ts

   atoms = ase.build.bulk("Si", "diamond", a=5.43, cubic=True)
   sim_state = ts.io.atoms_to_state([atoms], model.device, model.dtype)

Evaluate the model
------------------

Call the model on the simulation state to get energies, forces, and
stresses:

.. code-block:: python

   results = model(sim_state)

   print("Energy:", results["energy"])    # shape [1]
   print("Forces:", results["forces"])    # shape [n_atoms, 3]
   print("Stress:", results["stress"])    # shape [1, 3, 3]

Run NVE dynamics
----------------

Use TorchSim's Velocity Verlet integrator:

.. code-block:: python

   from torch_sim.integrators import VelocityVerletIntegrator

   integrator = VelocityVerletIntegrator(
       model=model,
       state=sim_state,
       dt=1.0,  # femtoseconds
   )

   for step in range(100):
       sim_state = integrator.step(sim_state)
       if step % 10 == 0:
           energy = model(sim_state)["energy"].item()
           print(f"Step {step:3d}  E = {energy:.4f} eV")

The total energy should remain approximately constant in an NVE
simulation, which serves as a basic sanity check for your model.

Next steps
----------

- :ref:`torchsim-model-loading` covers all supported input formats
- :ref:`torchsim-batched` explains running multiple systems at once
- :ref:`torchsim-architecture` describes the internals
