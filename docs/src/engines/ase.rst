.. _engine-ase:

ASE
===

.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatomic supported?
   * - https://ase-lib.org/
     - Via the ``metatomic-ase`` package

Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

.. py:currentmodule:: metatomic_ase

- the :ref:`energy <energy-quantity>`, non-conservative :ref:`forces
  <non-conservative-force-quantity>`, and :ref:`stress
  <non-conservative-stress-quantity>`, including their :ref:`variants
  <quantity-variants>`, are integrated with the ASE calculator interface;
- arbitrary outputs can be computed for any :py:class:`ase.Atoms` with
  :py:meth:`MetatomicCalculator.run_model`.

Installation
^^^^^^^^^^^^

Install the calculator with ``pip install metatomic-ase``. Rotational averaging
with :py:class:`SymmetrizedCalculator` requires SciPy 1.15 or newer when
``l_max > 0``. Space-group projection requires ``spglib``. For example,

.. code-block:: bash

    pip install "scipy>=1.15" spglib

Calculators
^^^^^^^^^^^

:py:class:`MetatomicCalculator` is the standard ASE calculator.
:py:class:`SymmetrizedCalculator` wraps it and computes finite-quadrature
rotational averages for total/per-atom energy, forces, and global stress.
Per-atom stress is not supported.

.. code-block:: python

    from metatomic_ase import MetatomicCalculator, SymmetrizedCalculator

    base = MetatomicCalculator("model.pt")
    atoms.calc = SymmetrizedCalculator(
        base,
        l_max=3,
        batch_size=16,
        include_inversion=True,
    )
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

Increase ``l_max`` and compare the requested results before using the finite
average in production.

Quadrature and batching
-----------------------

``l_max`` is the assumed angular bandwidth :math:`L`. It must cover both the
model response and target representation (at least 1 for forces and 2 for
stress), and can not be inferred from output rank alone. For componentwise
rotational standard deviations it must cover the complete back-rotated response,
whose bandwidth can be the model-response bandwidth plus the target rank.

The calculator uses a Lebedev rule of degree at least :math:`2L` and
:math:`2L+1` in-plane rotations, as required by the product-quadrature theorem.
The largest supported ``l_max`` is 65. Increase it until every requested mean
and standard deviation has converged. With ``include_inversion=True``, every
proper grid rotation is paired with its inverted partner, adding the full
improper coset. With ``l_max=0`` there is no proper rotational average; only
identity and inversion are averaged in this case.

ASE stores Cartesian vectors and cell vectors by row. For an active operation
:math:`R`, the calculator constructs each orbit member as

.. math::

    r_R = r R^\mathrm{T}, \qquad
    H_R = H R^\mathrm{T}, \qquad
    v_R = v R^\mathrm{T},

where :math:`v` is any requested polar-vector input. Predictions are returned
to the input frame before averaging: forces contribute :math:`F_R R`, stress
contributes :math:`R^\mathrm{T} S_R R`, and scalar energies are unchanged. The
rotation does not wrap positions into the cell, so the identity operation
preserves the supplied Cartesian coordinates and lattice images exactly.

``batch_size`` limits the rotated :py:class:`ase.Atoms` objects and predictions
in one model call, but does not change the grid or total work. Each batch is
reduced immediately using reference-centered moments, so the live prediction
payload scales with the batch instead of the full orbit. Quadrature rotations
and weights are precomputed for the full grid and are not bounded by
``batch_size``; at ``l_max=65`` they occupy about 58 MiB without inversion and
about 116 MiB with inversion. Setting ``batch_size=None`` evaluates the full
grid in one batch.

With ``store_rotational_std=True``, the results include componentwise
finite-grid diagnostics such as ``energy_rot_std``, ``energies_rot_std``,
``forces_rot_std``, and ``stress_rot_std`` in the reference frame. Except for a
single scalar component, these are not the sample- and component-aggregated
equivariance RMSE returned by
:py:class:`~metatomic.torch.symmetrized_model.SymmetrizedModel`. Signed Lebedev
weights can make an under-resolved finite-grid variance negative. The calculator
clamps only a summation-scale round-off allowance and otherwise asks for a
larger ``l_max``.

Requested ASE inputs
--------------------

Polar vector inputs requested by the model, such as momenta and velocities, are
rotated with the geometry. Every requested real numerical or Boolean
``(n_atoms, 3)`` ``ase::arrays::*`` input is interpreted as a polar vector. All
other requested per-atom arrays are treated as scalars. The reserved
``ase::arrays::positions`` alias is rejected: models must use
``System.positions`` so position and strain derivatives remain connected.
Unrequested arrays do not affect the averaging orbit.

Requested arrays/info and standard dependencies, including the masses used for
velocities, participate in ASE cache invalidation. Every requested value must be
real and finite, and custom ``ase::info::*`` inputs must be scalar. In-place
changes to requested mutable scalar values also invalidate the cache.

Vector ``initial_magmoms``, including the custom-array alias, are accepted only
for proper-rotation averaging (``include_inversion=False``). Their physical
axial parity is not represented by the ``xyz`` TensorMap produced by the ASE
converter, so including inversion would make the ASE and TensorMap actions
inconsistent.

During space-group filtering, scalar per-atom inputs must be preserved exactly
by the induced site permutation. Vector inputs admit only matrix-arithmetic
round-off, rather than a physical tolerance. ``initial_magmoms`` is the only
requested ASE input recognized as axial.

Space-group projection
----------------------

With ``apply_space_group_symmetry=True``, rotations, translations, and atom
permutations from ``spglib`` are applied after the O(3) average. Operations that
do not preserve the requested per-atom inputs are discarded. When this filtering
is active, the retained actions are checked for group closure so the average
remains an idempotent projector.

For Cartesian row vectors, a fractional rotation :math:`R_f` is converted with
:math:`Q=A R_f A^{-1}`, where :math:`A` is the transposed ASE cell. If
:math:`p(i)` is the original site matched by the image of site :math:`i` under
:math:`(R_f,t_f)`, one projector term uses ``values[p]`` for per-atom scalars,
``forces[p] @ Q`` for forces, and ``Q.T @ stress @ Q`` for stress. Translations
therefore enter per-atom outputs through :math:`p`. Operations with the same
rotation but different translations are retained as distinct actions whenever
they induce different permutations. Polar and axial inputs are filtered with
their corresponding parity.

The current discovery tolerances are ``symprec=1e-6`` and spglib's default
``angle_tolerance=-1``.

Sites are matched by species using minimum-image Cartesian distances. An
optional periodic index accelerates candidate discovery, while a memory-bounded
pairwise fallback keeps all supported cells correct; the worst case remains
quadratic in the same-species population. Every retained mapping must be within
the effective ``symprec`` threshold and form a bijection. The retained
permutations require
:math:`O(GN)` storage for :math:`G` actions and :math:`N` atoms, potentially
:math:`O(N^2)` in highly symmetric supercells. ``batch_size`` does not bound
this storage, so use a primitive cell or disable space-group projection if it
becomes the memory bottleneck.

The space-group projector acts on per-atom energies, forces, and global stress;
total energy is already scalar. The optional ``*_rot_std`` values describe the
preceding O(3) orbit and are not space-group projected.

Space-group projection and three-dimensional stress are defined only for
systems periodic in all three directions. Stress additionally requires a finite,
non-singular cell. Force evaluation also computes stress for fully periodic
systems, so this cell check applies to force-only calls there. For systems that
are not fully periodic, space-group projection is skipped and requesting stress
raises :py:class:`ase.calculators.calculator.PropertyNotImplementedError`.

Both calculators implement their corresponding ASE properties and can be used
in workflows that request them. See also the
:ref:`atomistic tutorial <atomistic-tutorial-md>`.

.. _ase-integration-api:

API documentation
-----------------

.. _calculator: https://ase-lib.org/ase/calculators/calculators.html

.. autoclass:: metatomic_ase.MetatomicCalculator
    :show-inheritance:
    :members:

.. autoclass:: metatomic_ase.SymmetrizedCalculator
    :show-inheritance:
    :members:
