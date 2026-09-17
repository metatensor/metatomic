.. _engine-openmm:

OpenMM
======

.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatomic supported?
   * - https://openmm.org/
     - Via `OpenMM-ML <https://github.com/openmm/openmm-ml>`_ (``openmmml``)


Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`energy <energy-quantity>` output is supported. Conservative forces are
obtained by differentiating the energy with respect to positions (autograd).
Periodic cells, mixed ML/MM regions, and model-requested neighbor lists are
handled. Per-system :ref:`charge <charge-quantity>` and
:ref:`spin_multiplicity <spin-multiplicity-quantity>` can be passed when the
model requests them.

See :ref:`model-dataflow` for the engine-side evaluation protocol this
integration follows.

How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

Install OpenMM-ML together with the metatomic Torch bindings:

.. code-block:: bash

   pip install openmmml metatomic-torch

For the ASE route below, also install ``metatomic-ase`` and ``ase``.

How to use the code
^^^^^^^^^^^^^^^^^^^

There are two ways to run a metatomic model inside OpenMM.

ASE calculator (correctness path)
---------------------------------

OpenMM-ML already accepts any ASE calculator. Pair it with
:class:`metatomic_ase.MetatomicCalculator`:

.. code-block:: python

   from metatomic_ase import MetatomicCalculator
   from openmmml import MLPotential

   calculator = MetatomicCalculator(
       "exported-model.pt",
       device="cuda",
       do_gradients_with_energy=True,
   )

   potential = MLPotential("ase")
   system = potential.createSystem(
       topology,
       calculator=calculator,
   )

This also works for mixed ML/MM:

.. code-block:: python

   system = potential.createMixedSystem(
       topology,
       mm_system,
       ml_atoms,
       calculator=calculator,
       info={
           "charge": 0,
           "spin_multiplicity": 1,
       },
   )

This path reuses the full ASE integration: exported-model loading, capability
and atomic-type checks, unit conversion, model-requested neighbor lists
(including CUDA lists through vesin / nvalchemi), conservative forces, periodic
cells, and extra inputs such as charge and spin.

Native ``MLPotential("metatomic")`` backend
-------------------------------------------

The first-class backend talks to :mod:`metatomic.torch` directly (no ASE
``Atoms`` object on the evaluation path):

.. code-block:: python

   from openmmml import MLPotential

   potential = MLPotential(
       "metatomic",
       modelPath="exported-model.pt",
       device="cuda",
       extensionsDirectory="./extensions",
       checkConsistency=False,
   )
   system = potential.createSystem(topology)

Optional ``createSystem()`` / ``createMixedSystem()`` arguments:

- ``charge``: total charge (default ``0``)
- ``multiplicity`` / ``spin_multiplicity``: spin multiplicity (default ``1``)
- ``info``: dict with the same keys, matching the ASE calculator convention

If the model requests extra inputs other than charge and spin multiplicity, use
the ASE path above.

The native backend returns energy and conservative forces only, which is enough
for NVT/NVE. It does not provide a virial, so NPT is not supported. CUDA
neighbor lists through nvalchemi are available on the ASE path, not on
``MLPotential("metatomic")``.
