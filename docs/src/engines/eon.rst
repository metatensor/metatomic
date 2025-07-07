.. _engine-eon:

eOn
===

.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatomic supported?
   * - https://theochemui.github.io/eOn/
     - In the official version


Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

Only the :ref:`energy <energy-output>` output is supported in eOn, as a custom
``Potential``. This allows running methods, including:

- Saddle search methods
  - Single ended (dimer method, GPR accelerated dimer)
  - Double ended (Nudged Elastic Band with energy weighted strings)
- Adaptive Kinetic Monte Carlo (aKMC) for long time scale simulations

With the engine integration, it is possible to run these with interatomic
potentials in ``metatomic`` format; distributing the calculation on GPUs as
well.
