Metatensor
==========

Metatensor is a specialized data storage format for all your atomistic machine
learning needs, and more. Think numpy ``ndarray`` or pytorch ``Tensor`` equipped
with extra metadata for atomic — and other particles — systems.

.. |C| image:: ../static/images/logo-c.*
    :width: 32px
    :height: 32px
    :alt: C

.. |Cxx| image:: ../static/images/logo-cxx.*
    :width: 32px
    :height: 32px
    :alt: C++

.. |Rust| image:: ../static/images/logo-rust.*
    :width: 32px
    :height: 32px
    :alt: Rust

.. |Python| image:: ../static/images/logo-python.*
    :width: 32px
    :height: 32px
    :alt: Python


.. grid::

    .. grid-item-card:: 🚀 Getting started
        :link: getting-started
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Install the right version of metatensor for your programming language!
        The core of this library is written in Rust and we provide API for C,
        C++, and Python.

        +++
        |Python| |Cxx| |C| |Rust|

    .. grid-item-card:: 💡 What is metatensor
        :link: about
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Learn about the core goals of metatensor, and what the library is about:

        - an exchange format for ML data;
        - a prototyping tool for new models;
        - an interface for atomistic simulations.

    .. grid-item-card:: 🛠️ Core classes
        :link: core-classes
        :link-type: ref
        :columns: 12 12 4 4
        :margin: 0 3 0 0

        Explore the core types of metatensor: :py:class:`TensorMap`,
        :py:class:`TensorBlock` and :py:class:`Labels`, and discover how to used
        them.

        +++
        |Python| |Cxx| |C| |Rust|


    .. grid-item-card:: 📈 Operations
        :link: metatensor-operations
        :link-type: ref
        :columns: 12 12 4 4
        :margin: 0 3 0 0

        Use `operations` to manipulate the core types of metatensor and write
        new algorithms operating on metatensor's sparse data.

        +++
        |Python|


    .. grid-item-card:: 🔥 TorchScript interface
        :link: metatensor-torch
        :link-type: ref
        :columns: 12 12 4 4
        :margin: 0 3 0 0

        Learn about the TorchScript version of metatensor, used to export and
        execute custom models with non-Python software.

        +++
        |Python| |Cxx|


    .. grid-item-card:: 🧑‍💻 Defining new models
        :link: metatensor-learn
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Use the utility class with the same API as torch or scikit-learn to
        define full models working with metatensor!

        +++
        |Python|


    .. grid-item-card:: ⚛️ Running atomistic simulations
        :link: atomistic-models
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Learn about the facilities provided to define atomistic models, and use
        them to run molecular dynamics simulations and more!

        +++
        |Python| |Cxx|



.. toctree::
   :maxdepth: 2
   :hidden:

   about
   installation
   core/index
   operations/index
   torch/index
   learn/index
   atomistic/index
   devdoc/index
