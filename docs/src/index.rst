.. image:: /../static/images/metatomic-horizontal.png
   :class: only-light sd-mb-4
   :width: 600px

.. image:: /../static/images/metatomic-horizontal-dark.png
   :class: only-dark sd-mb-4
   :width: 600px

``metatomic`` is a library that defines a common interface between atomistic
machine learning models, and atomistic simulation engines. Our main goal is to
define and train models once, and then be able to re-use them across many
different simulation engines (such as LAMMPS, PLUMED, *etc.*). We strive to
achieve this goal without imposing any structure on the model itself, and to
allow any model architecture to be used.

This library focuses on exporting and importing fully working, already trained
models. If you want to train existing architectures with new data or re-use
existing trained models, look into the metatrain_ project instead.

.. _metatrain: https://github.com/lab-cosmo/metatrain

.. grid::

    .. grid-item-card:: ⚛️ Overview
        :link: atomistic-overview
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Why should you use ``metatomic`` to define and export your model? What is
        the point of the interface? How can you use models that follow the
        interface in your own simulation code?

        All these questions and more will be answered in this overview!

    .. grid-item-card:: 💡 Tutorials
        :link: atomistic-tutorials
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Learn how to define your own models using ``metatomic``, and how to use
        these models to run calculations in various engines.

    .. grid-item-card:: 📋 Standard models outputs
        :link: atomistic-models-outputs
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Understand the different outputs a model can have, and what the metadata
        should be provided for standardized outputs, such as the potential energy.

    .. grid-item-card:: ⚙️ Simulation engines
        :link: engines
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Explore the various simulation software that can use metatomic models,
        and what each one of them can do, from running molecular dynamics
        simulations to interactive dataset exploration.

    .. grid-item-card:: |Python-16x16| Python API reference
        :link: python-api-torch
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Read the documentation for all the classes and functions related to
        atomistic models in Python.

        +++
        Documentation for version |metatomic-torch-version|

    .. grid-item-card:: |Cxx-16x16| C++ API reference
        :link: cxx-api-torch
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Read the documentation for all the classes and functions related to
        atomistic models in C++.

        +++
        Documentation for version |metatomic-torch-version|

.. toctree::
    :maxdepth: 2
    :hidden:

    overview
    installation
    torch/index
    outputs/index
    engines/index
    examples/index
