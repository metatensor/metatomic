.. _atomistic-tutorials:

Tutorials
=========

The first set of tutorials on this page is about existing integrations between
``metatomic`` and simulation engines. You can also find more example in the
:ref:`engines` section. These tutorials are intended for users who want to use
existing metatomic models with existing simulation engines.

.. toctree::
    :maxdepth: 1
    :hidden:

    ase/index
    torchsim/index


.. grid::

    .. grid-item-card:: ASE tutorials
        :link: ase-tutorials
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0
        :img-top: /../static/images/logo-ase.*
        :class-img-top: mta-card-img-top

        How to use ``metatomic`` with the Atomic Simulation Environment (ASE).

    .. grid-item-card:: Torch-Sim tutorials
        :link: torchsim-tutorials
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0
        :img-top: /../static/images/logo-radical-ai.*
        :class-img-top: mta-card-img-top

        How to use existing ``metatomic`` models with Torch-Sim, a simulation
        engine for batched molecular dynamics simulations, based on PyTorch.


--------------------------------------------------------------------------------

The second set of tutorials on this page is intended for developers who want to
use ``metatomic`` to either create new models, or run exising models in new
simulation engine. These tutorials require existing knowledge of the
corresponding programming languages and machine learning frameworks.

.. toctree::
    :maxdepth: 1
    :hidden:

    c/index
    torch/index

.. grid::

    .. grid-item-card:: C API tutorials
        :link: c-tutorials
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0
        :img-top: /../static/images/logo-c.*
        :class-img-top: mta-card-img-top

        How to use the C API of ``metatomic`` both to create custom atomistic
        models; and to load and run existing atomistic models from simulation
        engines.

    .. grid-item-card:: PyTorch tutorials
        :link: torch-tutorials
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0
        :img-top: /../static/images/logo-torch.*
        :class-img-top: mta-card-img-top

        How to use the PyTorch API of ``metatomic`` to define custom atomistic
        models.
