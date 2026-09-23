.. _devdoc-testing-plugins:

Testing plugins
===============

Metatomic installations include a shifted Lennard-Jones plugin for testing
simulation-engine integrations (LAMMPS, i-PI, ASE, and similar). Use it from a
normal checkout or ``pip`` install; do not copy the sources under
``examples/c/`` (those files are sphinx-gallery tutorials).

The Python package exposes the absolute path of the shared library:

.. code-block:: bash

    python -c "import metatomic; print(metatomic.utils.lj_plugin_path())"

Load this path with :c:func:`mta_load_plugin`, then load the model named
``lennard-jones`` (or ``lj``) from the ``lj-plugin`` plugin. Model options are a
JSON object. Numeric options may be JSON numbers or strings; ``atomic_type``
may be a JSON array of integers or a comma-separated string. The supported
options and defaults are:

================= ============
Option            Default
================= ============
``sigma``         ``1.0``
``epsilon``       ``1.0``
``cutoff``        ``3.0``
``atomic_type``   ``1``
``length_unit``   ``"Angstrom"``
``energy_unit``   ``"eV"``
================= ============

``atomic_type`` may be a JSON array (``[1, 6, 8]``) or a comma-separated list
(``"1,6,8"``) when the engine needs more than one type.

The model runs on the CPU with ``float64`` data. It supports:

- any number of systems in one ``mta_execute_model`` call
- the ``energy`` output as either system samples or per-atom samples
- an optional positions gradient on the system-level energy
- ``selected_atoms`` with the usual domain-decomposition convention: pair
  energies are split equally between the two endpoints, and only the shares
  belonging to selected atoms are returned
