.. _engine-gromacs:

GROMACS
=======

.. list-table::
    :header-rows: 1

    * - Official website
      - How is metatomic supported?
    * - https://www.gromacs.org
      - In a separate `fork <https://github.com/metatensor/gromacs>`_

Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`energy <energy-output>` is supported in the custom metatomic module in
GROMACS. The module allows running molecular dynamics simulations on the full system or
on a subgroup (ML/MM) with interatomic potentials in the metatomic format.

How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

Building from sources
---------------------

The code is available in a custom fork of GROMACS, and you can get it with

.. code-block:: bash

    git clone https://github.com/metatensor/gromacs gromacs-metatomic
    cd gromacs-metatomic

You'll need to provide some of the code dependencies yourself. There are
multiple ways to go about it, here we detail a fully manual installation, an
installation using ``conda`` and an installation using ``pip``.

Option 1: dependencies from ``conda``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All the dependencies of the code are available on ``conda``, you can install
them with

.. code-block:: bash

    # create an environment (you can also re-use an existing one)
    conda create -n gromacs-metatomic
    conda activate gromacs-metatomic

    conda install -c metatensor -c conda-forge libmetatomic-torch

    # Store this information to configure cmake down the line
    CMAKE_PREFIX_PATH="$CONDA_PREFIX"


Option 2: dependencies from ``pip``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All the dependencies of the code are also available on ``PyPI``, you can install
them with

.. code-block:: bash

    # (optional) create an environment with your preferred method
    ...

    python -m pip install metatomic-torch

    # on linux, if you don't have a GPU available, you should force the use of
    # CPU-only torch instead
    python -m pip install --extra-index-url=https://download.pytorch.org/whl/cpu metatomic-torch

    # Get the information to configure cmake down the line
    TORCH_PREFIX=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
    MTS_PREFIX=$(python -c "import metatensor; print(metatensor.utils.cmake_prefix_path)")
    MTS_TORCH_PREFIX=$(python -c "import metatensor.torch; print(metatensor.torch.utils.cmake_prefix_path)")
    MTA_TORCH_PREFIX=$(python -c "import metatomic.torch; print(metatomic.torch.utils.cmake_prefix_path)")

    CMAKE_PREFIX_PATH="$TORCH_PREFIX;$MTS_PREFIX;$MTS_TORCH_PREFIX;$MTA_TORCH_PREFIX"


Option 3: manual dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You'll need to build or download the the C++ version of ``libtorch``. You can
download it from https://pytorch.org/get-started/locally/, using the C++
language selector. Once you have it downloaded, extract the archive somewhere,
and record the path:

.. code-block:: bash

    # point this to the path where you extracted the C++ libtorch
    TORCH_PREFIX=<path/to/torch/installation>

For the other dependencies, you'll either need to install them yourself
following the links below, or let ``cmake`` download and build the latest
compatible versions:

- :external+metatensor:ref:`metatensor <install-c>`
- :external+metatensor:ref:`metatensor-torch <install-torch-cxx>`
- :ref:`metatomic-torch <install-torch-cxx>`

If you want to provide these yourself, you'll need to also record the
corresponding installation paths:

.. code-block:: bash

    MTS_PREFIX=<path/to/metatensor/installation>
    MTS_TORCH_PREFIX=<path/to/metatensor/torch/installation>
    MTA_TORCH_PREFIX=<path/to/metatomic/torch/installation>

And finally you can store this information to configure cmake down the line:

.. code-block:: bash

    CMAKE_PREFIX_PATH="$TORCH_PREFIX;$MTS_PREFIX;$MTS_TORCH_PREFIX;$MTA_TORCH_PREFIX"

Building the code
~~~~~~~~~~~~~~~~~

After installing the dependencies with one of the options above, you can
configure the build with:

.. code-block:: bash

    mkdir build && cd build

    # you can add more options here.
    cmake .. \
        -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
        -DTorch_DIR=$TORCH_PREFIX \
        -DGMX_METATOMIC=AUTO

    cmake --build . --parallel 4 # or `make -jX`

    # optionally install the code on your machine. You can also directly use
    # the `gmx` binary in `gromacs-metatomic/build/bin/gmx` without installation
    cmake --build . --target install # or `make install`

By default, ``cmake`` will try to find the ``metatensor`` and ``metatomic``
libraries on your system and use them. If it can not find the libraries, it will
download and build them as part of the main GROMACS build. You can control this
behavior by adding ``-DDOWNLOAD_METATENSOR=ON`` and ``-DDOWNLOAD_METATOMIC=ON``
to the ``cmake`` options to always force a download; or prevent any download by
setting these options to ``OFF``.

How to use the code
^^^^^^^^^^^^^^^^^^^

.. note::

    Here we assume you already have an exported model that you want to use in
    your simulations. Please see :ref:`this tutorial
    <atomistic-tutorial-export>` to learn how to manually create and export a
    model; or use a tool like `metatrain`_ to create a model based on existing
    architectures and your own dataset.

    .. _metatrain: https://github.com/metatensor/metatrain

After building and optionally installing the code, you can now use metatomic
module in your GROMACS molecular dynamics parameter (MDP) files! Below are the
reference options

  .. parsed-literal::

      **metatomic-active** yes or no
        set this to yes to activate the metatomic potential, or no to disable it.
      **metatomic-input-group**
        name of the input group to use for the metatomic potential. To couple the whole
        system use "System".
      **metatomic-model**
        path to the file containing the exported metatomic model.
      **metatomic-extensions**
        path to a directory containing TorchScript extensions as shared libraries. If
        the model uses extensions, we will try to load them from this directory first
      **metatomic-device** cpu or cuda
        device to use to run the model. If not given, the best available device will be
        used.
      **metatomic-check-consistency** yes or no
        if yes, the code will check that the model is consistent with the system
        topology at the start of the simulation. This can help catch errors due to
        mismatched atom ordering between the model and the system, but it comes at a
        performance cost.
      **metatomic-variant** no or <variant>
        specifies which variant of the model outputs should be uses for making
        predictions. Defaults to no variant.

.. note::

    The device can also be overridden at runtime by setting the environment
    variable ``GMX_METATOMIC_DEVICE`` to a value.

Sample input file
-----------------

Below is a example input file for an ML/MM simulation of an alanin dipeptide in water,
using a metatomic model for the peptide and a classical force field for the water
molecules. For a detailed example we refer to the `chapter in the atomistic cookbook
<TODO>`_.

.. code-block:: ini

    ; Run control
    integrator              = md
    dt                      = 0.0005
    nsteps                  = 500

    ; Output control
    nstxout                 = 10
    nstvout                 = 10
    nstenergy               = 10
    nstcalcenergy           = 10
    nstlog                  = 10

    ; Neighborsearching
    cutoff-scheme           = Verlet
    pbc                     = xyz

    ; Electrostatics
    coulombtype             = PME

    ; Temperature coupling
    tcoupl                  = v-rescale
    tc-grps                 = water protein ; two coupling groups - more accurate
    tau-t                   = 2.0   2.0 ; time constant, in ps
    ref-t                   = 300   300

    ; Metatomic interface
    metatomic-active        = yes
    metatomic-input-group   = protein  ; the group on which ML forces are applied
    metatomic-modelfile     = model.pt  ; path to the model file
    metatomic-device        = cpu  ; device to run the model on (cpu or cuda)
