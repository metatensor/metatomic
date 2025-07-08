.. _engine-eon:

eOn
===

.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatomic supported?
   * - https://theochemui.github.io/eOn/
     - In the official Github version


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

How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

Please refer to latest `eOn documentation`_ about how to install it.

.. _eOn documentation: https://theochemui.github.io/eOn/install/metatomic.html

How to use the code
^^^^^^^^^^^^^^^^^^^

.. note::

  Here we assume you already have an exported model that you want to use in your
  simulations. Please see :ref:`this tutorial <atomistic-tutorial-export>` to
  learn how to manually create and export a model; or use a tool like
  `metatrain`_ to create a model based on existing architectures and your own
  dataset.

  .. _metatrain: https://github.com/metatensor/metatrain

The metatomic interface in eOn provides a custom Metatomic Potential that can be
used in combination with any existing eOn runs, both server (aKMC) or client
(dimer, NEB). The relevant configuration is:

.. code-block:: ini

    [Potential]
    potential = metatomic

    [Metatomic]
    model_path = # $FULL_PATH/pet-mad-full-best.pt

Where it is more robust to use the complete model path, especially for the adaptive kinetic monte carlo runs. Complete details of the input file specification are present in the `corresponding reference documentation`_.

.. _corresponding reference documentation: https://theochemui.github.io/eOn/user_guide/index.html
