.. _torchsim-batched:

Batched simulations
===================

TorchSim supports batching multiple systems into a single ``SimState``
for efficient parallel evaluation on GPU. ``MetatomicModel`` handles
this transparently.

Creating a batched state
------------------------

Pass a list of ASE ``Atoms`` objects to ``initialize_state``:

.. code-block:: python

   import ase.build
   import torch_sim as ts
   from metatomic_torchsim import MetatomicModel

   model = MetatomicModel("model.pt", device="cpu")

   atoms_list = [
       ase.build.bulk("Cu", "fcc", a=3.6, cubic=True),
       ase.build.bulk("Ni", "fcc", a=3.52, cubic=True),
       ase.build.bulk("Al", "fcc", a=4.05, cubic=True),
   ]

   sim_state = ts.initialize_state(atoms_list, device=model.device, dtype=model.dtype)

Evaluating the batch
--------------------

A single forward call evaluates all systems:

.. code-block:: python

   results = model(sim_state)

The output shapes reflect the batch:

- ``results["energy"]`` has shape ``[n_systems]`` (one energy per system)
- ``results["forces"]`` has shape ``[n_total_atoms, 3]`` (all atoms
  concatenated)
- ``results["stress"]`` has shape ``[n_systems, 3, 3]`` (one 3x3 tensor
  per system)

How system_idx works
--------------------

``SimState`` tracks which atom belongs to which system via the
``system_idx`` tensor. For three 4-atom systems, ``system_idx`` looks
like::

   [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

``MetatomicModel.forward`` uses this to split the batched positions and
types into per-system ``System`` objects before calling the underlying
model.

Batch consistency
-----------------

Energies computed in a batch match those computed individually. This is
guaranteed because each system gets its own neighbor list and
independent evaluation.

Performance considerations
--------------------------

Batching is most beneficial on GPU, where the neighbor list computation
and model forward pass can run in parallel across systems. On CPU, the
speedup comes from reduced Python overhead (one call instead of N).

For very large systems or many small ones, adjust the batch size to fit
in GPU memory. TorchSim does not impose a maximum batch size, but each
system gets its own neighbor list, so memory scales with the sum of
per-system sizes.
