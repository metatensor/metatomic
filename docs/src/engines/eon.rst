.. _engine-eon:

eOn
===

.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatomic supported?
   * - https://eondocs.org/
     - In the official Github version


Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

The eOn interface primarily utilizes the :ref:`energy <energy-output>` output to
compute forces via autograd and drive molecular dynamics or saddle point
searches.

Additionally, the interface supports the :ref:`energy_uncertainty
<energy-uncertainty-output>` output. When enabled, the client checks per-atom
uncertainties against a user-defined threshold and flags or terminates
calculations that enter unreliable regions of the potential energy surface.

This allows running methods including:

- **Saddle search methods:**
  - Single ended (dimer method, GPR accelerated dimer)
  - Double ended (Nudged Elastic Band with energy weighted strings, OCI-NEB)
- **Long timescale simulations:**
  - Adaptive Kinetic Monte Carlo (aKMC)
  - Parallel Replica Dynamics

How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

Please refer to the latest `eOn documentation`_ for installation instructions.

.. _eOn documentation: https://eondocs.org/install/metatomic.html

How to use the code
^^^^^^^^^^^^^^^^^^^

.. note::

   This guide assumes the existence of an exported model for use in simulations.
   Please see :ref:`this tutorial <atomistic-tutorial-export>` to learn how to
   manually create and export a model, or use a tool like `metatrain`_ to create
   a model based on existing architectures and custom datasets.

   .. _metatrain: https://github.com/metatensor/metatrain

The metatomic interface in eOn provides a custom Metatomic Potential that can be
used in combination with any existing eOn runs, both server (aKMC) or client
(dimer, NEB). The relevant configuration is:

Basic Configuration
"""""""""""""""""""

To enable the potential, specify ``metatomic`` in the ``[Potential]`` block and
provide the model path:

.. code-block:: ini

    [Potential]
    potential = metatomic

    [Metatomic]
    model_path = # $FULL_PATH/pet-mad-full-best.pt

Advanced Configuration
""""""""""""""""""""""

The interface exposes additional parameters to control uncertainty
quantification and model variants. Complete details of the input file
specification reside in the `corresponding reference documentation`_.

.. _corresponding reference documentation: https://eondocs.org/user_guide/metatomic_pot
