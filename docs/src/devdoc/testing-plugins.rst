.. _devdoc-testing-plugins:

Testing plugins
===============

Metatomic installations include a shifted Lennard-Jones plugin for testing
simulation-engine integrations (LAMMPS, i-PI, ASE, and similar). Use it from a
normal checkout or ``pip`` install.

The Python package exposes paths of installed test plugins via
:mod:`metatomic.testing` (resolved from the CMake install layout):

.. code-block:: bash

    python -c "import metatomic; print(metatomic.testing.plugin_path('lj-plugin'))"

Load this path with :c:func:`mta_load_plugin`, then load the model named
``lennard-jones`` (or ``lj``) from the ``lj-plugin`` plugin. Model options are a
JSON object with string keys and string values. The supported options and
defaults are:

================= ============
Option            Default
================= ============
``sigma``         ``"1.0"``
``epsilon``       ``"1.0"``
``cutoff``        ``"3.0"``
``atomic_type``   ``"1"``
``length_unit``   ``"Angstrom"``
``energy_unit``   ``"eV"``
================= ============

``atomic_type`` may be a comma-separated list (``"1,6,8"``) when the engine
needs more than one type.

The model supports ``selected_atoms`` with the usual domain-decomposition
convention: pair energies are split equally between the two endpoints, and only
the shares belonging to selected atoms are returned. See the model capabilities
for devices, dtype, outputs, and gradients.
