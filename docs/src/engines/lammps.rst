.. _engine-lammps:

LAMMPS
======

.. list-table::
    :header-rows: 1

    * - Official website
      - How is metatomic supported?
    * - https://lammps.org
      - In a separate `fork <https://github.com/metatensor/lammps>`_


Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`energy <energy-quantity>`, non-conservative :ref:`forces
<non-conservative-force-quantity>` and :ref:`stress
<non-conservative-stress-quantity>` outputs are supported in LAMMPS, as a custom
``pair_style``. This allows running molecular dynamics simulations with
interatomic potentials in the metatomic format; distributing the simulation over
multiple nodes and potentially multiple GPUs.

How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

Getting a pre-built binary with ``conda``
-----------------------------------------

The easiest way to install a version of ``lammps`` which can use metatomic
models is to use the build we provide through conda. We recommend that you use
`miniforge`_ as your conda provider.

First you'll need to pick an MPI implementation, from ``openmpi``, ``mpich`` or
``nompi`` (which does not have MPI enabled). If you'd like to use the MPI
library from your system (for example when running on supercomputers with
specific MPI tuning), please follow these instructions:
https://conda-forge.org/docs/user/tipsandtricks/#using-external-message-passing-interface-mpi-libraries

You can then install LAMMPS with:

.. code-block:: bash

    # for example with nompi
    conda install -c metatensor -c conda-forge "lammps-metatomic=*=*nompi*"

    # or with openmpi
    conda install -c metatensor -c conda-forge "lammps-metatomic=*=*openmpi*"

This version of LAMMPS will be able to run the models on CPU or GPU, but will
run the time integration of the trajectory on CPU. If you also want to run the
time integration on GPU, you'll need to install the `kokkos-enabled`_ build of
LAMMPS. This build currently only exists for CUDA GPUs, and each build only
supports a single GPU architecture.

.. _kokkos-enabled: https://docs.lammps.org/Speed_kokkos.html

To get the correct KOKKOS build for your GPU, you'll first need to determine its
compute capability. For this, you can run the following command with a GPU
present (i.e. on the compute node of supercomputers):

.. code-block:: bash

    nvidia-smi --query-gpu=compute_cap --format=csv,noheader

Alternatively, you can also get the compute capability from `NVIDIA's
documentation <https://developer.nvidia.com/cuda-gpus>`_

We currenly build the code for the following compute capabilities:

- ``VOLTA70``
- ``AMPERE80``
- ``AMPERE86``
- ``ADA89``
- ``HOPPER90``

For example, if you have a **NVIDIA A100 GPU**, its compute capability is
``8.0`` (i.e. ``AMPERE80``).

Now that you know the compute capability, you can install the correct kokkos
build (here as well, you can pick between different MPI implementations).

.. code-block:: bash

    conda install -c metatensor -c conda-forge "lammps-metatomic=*=cuda*AMPERE80*nompi*"

.. warning::

    Be aware that some HPC clusters may be set up without NVIDIA drivers
    installed on the head/login node. This will result in conda not detecting
    the system configuration of the compute nodes (which probably have GPUs if
    you are in this section) and will not install correct torch and cuda
    libraries. To fix it, you should run the install from a GPU node (check that
    after running ``conda info`` on the node your installing from, if ``__cuda`` is
    in the ``virtual packages`` section).

    You can also trick conda into installing cuda enabled versions on a
    login node without NVIDIA drivers, by setting the environment variable
    ``CONDA_OVERRIDE_CUDA`` to the correct CUDA version:

    .. code-block:: bash

        CONDA_OVERRIDE_CUDA=12.4 conda install -c metatensor -c conda-forge "lammps-metatomic=*=cuda*AMPERE80*nompi*"

.. note::

    If you get the following error

    .. code-block::

        Kokkos::Cuda::initialize ERROR: likely mismatch of architecture

    you are likely using the wrong KOKKOS build. Please double check that the
    compute capabilities of your GPU match the build you used.

.. _miniforge: https://github.com/conda-forge/miniforge

Building from sources
---------------------

The code is available in a custom fork of LAMMPS, and you can get it with

.. code-block:: bash

    git clone https://github.com/metatensor/lammps lammps-metatomic
    cd lammps-metatomic

You'll need to provide some of the code dependencies yourself. There are
multiple ways to go about it, here we detail a fully manual installation, an
installation using ``conda`` and an installation using ``pip``.

Option 1: dependencies from ``conda``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All the dependencies of the code are available on ``conda``, you can install
them with

.. code-block:: bash

    # create an environment (you can also re-use an existing one)
    conda create -n lammps-metatomic
    conda activate lammps-metatomic

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

    # on linux, if you don't have the cuda toolkit installed, you should force the use
    # of CPU-only torch instead
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

    # you can add more options here to enable other packages.
    cmake -DPKG_ML-METATOMIC=ON \
          -DLAMMPS_INSTALL_RPATH=ON \
          -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
          ../cmake

    cmake --build . --parallel 4 # or `make -jX`

    # optionally install the code on your machine. You can also directly use
    # the `lmp` binary in `lammps-metatomic/build/lmp` without installation
    cmake --build . --target install # or `make install`

By default, ``cmake`` will try to find the ``metatensor`` and ``metatomic``
libraries on your system and use them. If it can not find the libraries, it will
download and build them as part of the main LAMMPS build. You can control this
behavior by adding ``-DDOWNLOAD_METATENSOR=ON`` and ``-DDOWNLOAD_METATOMIC=ON``
to the ``cmake`` options to always force a download; or prevent any download by
setting these options to ``OFF``.


To enable KOKKOS and use a GPU for the time integration, you'll need to add the
following flags to the ``cmake`` configuration, and then continue the build in the
same way as above.

.. code-block:: bash

    cmake [other flags] \
          -DPKG_KOKKOS=ON \
          -DKokkos_ENABLE_CUDA=ON \
          -DKokkos_ENABLE_OPENMP=ON \
          -DKokkos_ARCH_<ARCH>=ON \ # replace <ARCH> with the correct GPU architecture
          -DCMAKE_CXX_COMPILER="$PWD/../lib/kokkos/bin/nvcc_wrapper" \
          ../cmake

See `the main lammps documentation
<https://docs.lammps.org/Build_extras.html#kokkos>`_ to get more information
about configuring a kokkos build.

How to use the code
^^^^^^^^^^^^^^^^^^^

``pair_style metatomic``
------------------------

``pair_style metatomic`` is a custom LAMMPS pair style that allows using a
metatomic model as an interatomic potential. Below is the reference
documentation for this pair style, following a structure similar to the official
LAMMPS documentation.

.. note::

    Here we assume you already have an exported model that you want to use in
    your simulations. Please see :ref:`this tutorial
    <atomistic-tutorial-export>` to learn how to manually create and export a
    model; or use a tool like `metatrain`_ to create a model based on existing
    architectures and your own dataset.

    .. _metatrain: https://github.com/metatensor/metatrain

.. code-block:: LAMMPS

    pair_style metatomic model_path ... keyword values ...

* ``model_path`` = path to the file containing the exported metatomic model
* ``keyword`` = **device** or **extensions** or **check_consistency** or
* **non_conservative** or **scale** or **uncertainty_threshold** or **variant** or
* **variant/energy** or **variant/energy_uncertainty** or
* **variant/non_conservative_forces** or **variant/non_conservative_stress**

  .. parsed-literal::

      **device** values = device_name
        device_name = name of the Torch device to use for the calculations
      **extensions** values = directory
        directory = path to a directory containing TorchScript extensions as shared
        libraries. If the model uses extensions, we will try to load them from this
        directory first
      **non_conservative** values = on or off or forces or stress
        controls which outputs are read directly from the model rather than computed
        via autograd on the energy:

        - ``off`` (default): conservative mode; forces and stress are both derived
          from the gradient of the energy.
        - ``on``: both forces and stress are read directly from the model's
          non-conservative outputs, typically affording a speedup factor between 2
          and 3. We recommend using this in combination with RESPA to obtain
          physically correct observables (see https://arxiv.org/abs/2412.11569 for
          more information, and
          https://atomistic-cookbook.org/examples/pet-mad-nc/pet-mad-nc.html for an
          example of how to set up the RESPA run).
        - ``forces``: forces are read directly from the model's
          ``non_conservative_forces`` output; stress is still obtained via autograd.
        - ``stress``: stress is read directly from the model's
          ``non_conservative_stress`` output; forces are still obtained via autograd.
      **scale** values = float
        multiplies the contribution of the potential by a scaling factor. Defaults to 1.
      **check_consistency** values = on or off
        set this to on/off to enable/disable internal consistency checks, verifying both
        the data passed by LAMMPS to the model, and the data returned by the model to
        LAMMPS.
      **uncertainty_threshold** values = float or off
        sets a threshold on the maximum allowed energy uncertainty for the model
        predictions. If the model returns an uncertainty larger than this
        threshold for any of the atoms in the system, the simulation will issue
        a warning. Default to 100 meV/atom, set this to ``off`` to disable
        uncertainty warnings.
      **variant** values = string or off
        specifies which variant of the model outputs should be uses for making
        predictions. If set to a value the same variant will be used for the energy the
        energy uncertainty, the non conservative forces and stresses. You set a specific
        variant for each of them with commands given below. Defaults to no variant.
      **variant/energy** values = string or off
        specifies which variant of the model outputs should be uses for making energy
        predictions. Overrides the value given to the ``variant`` keyword. Defaults to
        no variant.
      **variant/energy_uncertainty** values = string or off
        specifies which variant of the model outputs should be uses for making energy
        uncertainty predictions. Overrides the value given to the ``variant`` keyword.
        Defaults to no variant.
      **variant/non_conservative_forces** values = string or off
        specifies which variant of the model outputs should be uses for making
        non-conservative forces predictions. Overrides the value given to the
        ``variant`` keyword. Defaults to no variant.
      **variant/non_conservative_stress** values = string or off
        specifies which variant of the model outputs should be uses for making
        non-conservative stress predictions. Overrides the value given to the
        ``variant`` keyword. Defaults to no variant.

Multiple versions of the pair style are available, to give the ability to mix
multiple metatomic potentials in the same simulation. In addition to
``pair_style metatomic``, you can also use ``pair_style metatomic_1``,
``pair_style metatomic_2``, up to ``pair_style metatomic_9``.

Examples
~~~~~~~~

.. code-block:: LAMMPS

    pair_style metatomic exported-model.pt device cuda extensions /home/user/torch-extensions/
    pair_style metatomic soap-gap.pt check_consistency on
    pair_coeff * * 6 8 1

    pair_style pair_style hybrid/overlay &
        metatomic_1 model_1.pt device cpu non_conservative on &
        metatomic_2 model_2.pt device cuda variant pbe0

    pair_coeff * * metatomic_1 6 17 1 7
    pair_coeff * * metatomic_2 6 17 1 7

Description
~~~~~~~~~~~

Pair style ``metatomic`` provides access to models following :ref:`metatomic
models <model-interface>` interface; and enables using such models as
interatomic potentials to drive a LAMMPS simulation. The models can be fully
defined and trained by the user using Python code, or be existing pre-trained
models. The interface can be used with any type of machine learning model, as
long as the implementation of the model is compatible with TorchScript.

The only required argument for ``pair_style metatomic`` is the path to the model
file, which should be an exported metatomic model.

Optionally, users can define which torch ``device`` (e.g. cpu, cuda, cuda:0,
*etc.*) should be used to run the model. If this is not given, the code will run
on the best available device. If the model uses custom TorchScript operators
defined in a TorchScript extension, the shared library defining these extensions
will be searched in the ``extensions`` path, and loaded before trying to load
the model itself. Finally, ``check_consistency`` can be set to ``on`` or ``off``
to enable (or disable) additional internal consistency checks in the
data being passed from LAMMPS to the model and back.

A single ``pair_coeff`` command should be used with the ``metatomic`` style,
specifying the mapping from LAMMPS types to the atomic types the model can
handle. The first 2 arguments must be \* \* so as to span all LAMMPS atom types.
This is followed by a list of N arguments that specify the mapping of
metatomic's atomic types to LAMMPS types, where N is the number of LAMMPS atom
types.

Sample input file
~~~~~~~~~~~~~~~~~

Below is an example input file that creates an FCC crystal of Nickel, and use a
metatomic model to run NPT simulations. You can save this file to ``input.in``
and run the simulation with ``lmp -in input.in``.

.. code-block:: LAMMPS

    units metal
    boundary p p p

    # create the simulation system without reading external data file
    atom_style atomic
    lattice fcc 3.6
    region box block 0 4 0 4 0 4
    create_box 1 box
    create_atoms 1 box

    labelmap atom 1 Ni
    mass Ni 58.693

    # define the interaction style to use the model in the "nickel-model.pt" file
    pair_style metatomic nickel-model.pt device cuda
    pair_coeff * * 28

    # simulation settings
    timestep 0.001 # 1fs timestep
    fix 1 all npt temp 243 243 $(100 * dt) iso 0 0 $(1000 * dt) drag 1.0

    # output setup
    thermo 10

    # run the simulation for 10000 steps
    run 10000

Here is the same input file, using the KOKKOS version of the ``pair_style``. You
can save this file to ``input-kokkos.in``, and run it with ``lmp -in
input-kokkos.in -suffix kk -k on g 1``. See the `lammps-kokkos`_ documentation
for more information about kokkos options.

.. _lammps-kokkos: https://docs.lammps.org/Speed_kokkos.html

.. code-block:: LAMMPS

    package kokkos newton on neigh half

    units metal
    boundary p p p

    # create the simulation system without reading external data file
    atom_style atomic/kk
    lattice fcc 3.6
    region box block 0 4 0 4 0 4
    create_box 1 box
    create_atoms 1 box

    mass 1 58.693

    # the model will automatically run on the same device as the kokkos code
    pair_style metatomic/kk nickel-model.pt
    pair_coeff * * 28

    # simulation settings
    timestep 0.001 # 1fs timestep
    fix 1 all npt temp 243 243 $(100 * dt) iso 0 0 $(1000 * dt) drag 1.0

    # output setup
    thermo 10

    run_style verlet/kk
    # run the simulation for 10000 steps
    run 10000

``fix metatomic``
-----------------

``fix metatomic`` is a custom LAMMPS fix that allows using a metatomic model to
apply operations on the system. A typical use case is to use the `FlashMD`_
models to perform long stride MD simulations.

.. _FlashMD: https://docs.lammps.org/Speed_kokkos.html

.. code-block:: LAMMPS

    fix ID group_ID metatomic model_path ... keyword values ...

* ``model_path`` = path to the file containing the exported metatomic model
* ``keyword`` = **device** or **extensions** or **check_consistency** or **types**

    .. parsed-literal::

      **device** values = device_name
        device_name = name of the Torch device to use for the calculations
      **extensions** values = directory
        directory = path to a directory containing TorchScript extensions as
        shared libraries. If the model uses extensions, we will try to load them
        from this directory first
      **check_consistency** values = on or off
        set this to on/off to enable/disable internal consistency checks,
        verifying both the data passed by LAMMPS to the model, and the data
        returned by the model to LAMMPS.
      **types** values = list of types
        list of types = list of atom types that will be passed to the model,
        mapping the LAMMPS atom types to the model atom types. Each type is
        separated by a space. The first type in the list corresponds to the
        first LAMMPS atom type in the system, the second type in the list
        corresponds to the second LAMMPS atom type, and so on.

Examples
~~~~~~~~

.. code-block:: LAMMPS

    # NVT-langevin
    fix 0 all metatomic flashmd-16fs.pt types 13 device cuda
    fix 1 all langevin 700.0 700.0 0.1 12345

    # NVT-CSVR
    fix 0 all metatomic flashmd-16fs.pt types 13 device cuda
    fix 1 all temp/csvr 700.0 700.0 0.1 12345

    # NPT
    fix 0 all metatomic flashmd-16fs.pt types 13 device cuda
    fix 1 all langevin 700.0 700.0 0.1 12345  # or CSVR
    fix 2 all press/langevin iso 1.0 1.0 1.0 temp 700.0 700.0 67890

Description
~~~~~~~~~~~

Fix ``metatomic`` provides access to models following :ref:`metatomic models
<model-interface>` interface; and enables using such models to apply operations
on the system, e.g., using a metatomic model to predict the system state after a
certain time and update the system accordingly. The models can be defined and
trained by the user using Python code, or be existing pre-trained models. So
far, only the `FlashMD`_ models have been tested, which can be used to perform
long stride MD simulations.

The required arguments for ``fix metatomic`` are the path to the model file,
which should be an exported metatomic model, and the list of atom types mapping
the LAMMPS atom types to the model atom types. The list of types should be
provided in the same order as the LAMMPS atom types, and each type should be
separated by a space.

Optionally, users can define which torch ``device`` (e.g. cpu, cuda, cuda:0,
*etc.*) should be used to run the model. If this is not given, the code will run
on the best available device. If the model uses custom TorchScript operators
defined in a TorchScript extension, the shared library defining these extensions
will be searched in the ``extensions`` path, and loaded before trying to load
the model itself. Finally, ``check_consistency`` can be set to ``on`` or ``off``
to enable (or disable) additional internal consistency checks in the data being
passed from LAMMPS to the model and back.

Sample input file
~~~~~~~~~~~~~~~~~

Below is an example input file that creates an FCC crystal of Aluminum, and use
a FlashMD model to run NVT simulations with a stride of 16 fs. You can save this
file to ``input.in`` and run the simulation with ``lmp -in input.in``.

.. code-block:: LAMMPS

    units metal
    atom_style atomic
    boundary p p p

    lattice fcc 4.05
    region box block 0 3 0 3 0 3
    create_box 1 box
    create_atoms 1 box

    mass 1 26.9815386

    velocity all create 800.0 12345 mom yes rot yes dist gaussian

    timestep 0.016
    fix 0 all metatomic flashmd-16fs.pt types 13
    fix 1 all langevin 700.0 700.0 0.1 12345

    thermo 10
    thermo_style custom step temp pe ke etotal

    run 100

Here is the same input file, using the KOKKOS version of the ``fix``. You can
save this file to ``input-kokkos.in``, and run it with ``lmp -in input-kokkos.in
-suffix kk -k on g 1``. See the `lammps-kokkos`_ documentation for more
information about kokkos options.

.. code-block:: LAMMPS

  package kokkos newton on neigh half

  units metal
  atom_style atomic/kk
  boundary p p p

  lattice fcc 4.05
  region box block 0 3 0 3 0 3
  create_box 1 box
  create_atoms 1 box

  mass 1 26.9815386

  velocity all create 800.0 12345 mom yes rot yes dist gaussian

  timestep 0.016

  fix 0 all metatomic/kk flashmd-16fs.pt types 13
  # fix 1 all langevin 700.0 700.0 0.1 12345

  thermo 10
  thermo_style custom step temp pe ke etotal

  run_style verlet/kk
  run 100

``compute metatomic``
---------------------

``compute metatomic`` is a custom LAMMPS compute that allows using a metatomic
model to compute properties of the system and output them to the LAMMPS log
file.

.. code-block:: LAMMPS

    compute ID group-ID metatomic model_path output_name ... keyword values ...

* ``model_path`` = path to the file containing the exported metatomic model
* ``output_name`` = name of the output to compute, e.g., energy, forces, stress,
  energy/pbe0. Must be a valid output name of the model, which can be accessed through
  :py:attr:`ModelCapabilities.outputs <metatomic.torch.ModelCapabilities.outputs>`.
* ``keyword`` = **device** or **extensions_directory** or **check_consistency** or
  **unit** or **shape**

  .. parsed-literal::

      **device** values = device_name
        device_name = name of the Torch device to use for the calculations
      **extensions_directory** values = directory
        directory = path to a directory containing TorchScript extensions as
        shared libraries. If the model uses extensions, we will try to load them
        from this directory first
      **types** values = list of types
        list of types = list of atom types that will be passed to the model,
        mapping the LAMMPS atom types to the model atom types. Each type is
        separated by a space. The first type in the list corresponds to the
        first LAMMPS atom type in the system, the second type in the list
        corresponds to the second LAMMPS atom type, and so on.
      **unit** values = string or off
        string = unit of the output, e.g., eV, kcal/mol, kJ/mol, etc. Must be
        specified if the output is not the :ref:`standard quantity
        <standard-quantities>`. If not specified, the output will be converted
        according to the current unit style.
      **shape** values = scalar or vector or off
        string = shape of the output, etc. Must be specified if the output is
        not the :ref:`standard quantity <standard-quantities>` or is a
        *feature*. When a vector is specified, the number of components must be
        specified as well, e.g., ``shape vector 3``.

Examples
~~~~~~~~

.. code-block:: LAMMPS

    compute energy all metatomic model.pt energy types 28
    # use the `doubled` variant of the energy output
    compute energy_doubled all metatomic model.pt energy/doubled types 28

    compute heat_flux all metatomic model.pt heat_flux extensions_directory collected-heat-flux-extensions types 28

    compute forces all metatomic model.pt non_conservative_force types 28

    thermo 10
    thermo_style custom step temp pe etotal press vol c_heat_flux[1] c_heat_flux[2] c_heat_flux[3]

    dump 1 all custom 10 dump.compute_metatomic id type x y z c_energy c_energy_doubled c_forces[1] c_forces[2] c_forces[3]

Description
~~~~~~~~~~~

Compute ``metatomic`` provides access to models following :ref:`metatomic models
<model-interface>` interface; and enables using such models to calculate the
properties of the system within the capabilities of the model. The models can be
fully defined and trained by the user using Python code, or be existing
pre-trained models. The interface can be used with any type of machine learning
model, as long as the implementation of the model is compatible with
TorchScript.

The only required arguments for ``compute metatomic`` are the path to the model
file, which should be an exported metatomic model, and the name of the desired
output quantity, which must be a valid output name for the model.

.. note::

    The output name can be a standard quantity, as defined in the
    :ref:`standard-quantities` section, or a non-standard quantity. In the
    latter case, the user must specify the unit and shape of the output. The
    unit and shape of the output can be specified using the ``unit`` and
    ``shape`` keywords, respectively. The unit must be a valid unit for the
    output quantity, and the shape must be ``scalar`` or ``vector``. If the
    output is a vector, the number of components must also be specified.

Optionally, users can define which torch ``device`` (e.g. cpu, cuda, cuda:0,
*etc.*) should be used to run the model. If this is not given, the code will run
on the best available device. If the model uses custom TorchScript operators
defined in a TorchScript extension, the shared library defining these extensions
will be searched in the ``extensions_directory`` path, and loaded before trying
to load the model itself. Finally, ``check_consistency`` can be set to ``on`` or
``off`` to enable (or disable) additional internal consistency checks in the
data being passed from LAMMPS to the model and back.

Sample input file
~~~~~~~~~~~~~~~~~

Below is an example input file that creates an FCC crystal of Nickel, run a
short simulation, and output the atomic energy, heat flux, and forces through
`compute metatomic`. You can save this file to ``input.in`` and run the
simulation with ``lmp -in input.in``.

.. code-block:: LAMMPS

  units metal
  boundary p p p

  atom_style atomic
  lattice fcc 3.6
  region box block 0 2 0 2 0 2
  create_box 1 box
  create_atoms 1 box

  mass 1 58.693

  velocity all create 123 42

  pair_style metatomic energy-model.pt
  pair_coeff * * 11

  timestep 0.001
  fix 1 all npt temp 123 123 $(100 * dt) iso 0 0 $(1000 * dt) drag 1.0

  compute heat_flux all metatomic heat-flux-model.pt heat_flux types 11

  thermo 10
  thermo_style custom step temp pe etotal press vol c_heat_flux[1] c_heat_flux[2] c_heat_flux[3]

  run 30
