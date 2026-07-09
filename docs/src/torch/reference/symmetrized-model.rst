Symmetrized models
==================

The :py:mod:`metatomic.torch.symmetrized_model` module provides
:py:class:`SymmetrizedModel`, a wrapper that averages the outputs of any
atomistic model over the full orthogonal group :math:`O(3)`, and computes
equivariance metrics quantifying how far the model is from being exactly
equivariant.

The wrapped model can be either a module following the
:py:class:`metatomic.torch.ModelInterface` call convention or an exported
:py:class:`metatomic.torch.AtomisticModel`, for example one loaded with
:py:func:`metatomic.torch.load_atomistic_model`.

The wrapped model is evaluated on copies of each system rotated over a
quadrature grid on :math:`O(3)` (a Lebedev grid supplemented by in-plane
rotations, covering both proper and improper rotations). Each output is then
"back-rotated" using the :math:`O(3)` action appropriate for its tensorial
type, following the conventions described in :ref:`o3-conventions`, and
averaged over the grid with the quadrature weights.

Constructing a :py:class:`SymmetrizedModel` requires `scipy`_ (version 1.15 or
later, for the Lebedev quadrature rule); scipy is only imported when the class
is instantiated.

.. _scipy: https://scipy.org/

Outputs
-------

Each requested output is first decomposed into irreducible representations of
:math:`O(3)`: energies contribute an ``_l0`` (scalar) tensor, forces an ``_l1``
(vector) tensor, and stresses both ``_l0`` (trace) and ``_l2`` (symmetric
traceless) tensors. For every resulting tensor ``<name>``, the returned
dictionary contains three entries:

- ``<name>_mean``: the :math:`O(3)`-averaged, symmetrized output;
- ``<name>_var``: the variance of the back-rotated output over the group,
  which vanishes for a perfectly equivariant model;
- ``<name>_norm_squared``: the average squared norm of the output over the
  group.

For example, requesting ``energy`` with ``compute_gradients=True`` on a
periodic system returns ``energy_l0_mean``, ``energy_l0_var``,
``energy_l0_norm_squared``, the corresponding ``forces_l1_*`` entries, and the
``stress_l0_*`` and ``stress_l2_*`` entries; forces and stress are computed
from the energy via autograd, through the rotations.

Equivariance error
------------------

For evaluation pipelines that only need the equivariance error (e.g. computing
it alongside accuracy metrics during model evaluation), the
:py:meth:`SymmetrizedModel.equivariance_error` method reduces everything to
per-system RMSE values. The reduction preserves the block and property
structure of each output (e.g. the ``o3_lambda``/``o3_sigma`` blocks and
radial channels of a spherical target), so values can be aggregated later in
whichever way the analysis needs.

The normalization matches an element-wise accuracy RMSE over the same target
(per component, averaged over each system's samples), making the two numbers
directly comparable. In fact, for an exactly equivariant reference,

.. math::

    \left\langle \mathrm{RMSE}_\mathrm{accuracy}^2 \right\rangle_{O(3)}
    =
    \mathrm{RMSE}_\mathrm{accuracy}^2\big[\text{symmetrized model}\big]
    +
    \mathrm{RMSE}_\mathrm{equivariance}^2,

so the equivariance error is the contribution of symmetry breaking to the
(orientation-averaged) accuracy error.

To pool per-system values into a dataset-level number, combine the squares
weighted by the number of samples :math:`N_A` of each system,
:math:`\sqrt{\sum_A N_A\, \mathrm{rmse}_A^2 / \sum_A N_A}`; a plain mean of
per-system RMSEs is not the pooled RMSE when system sizes differ.

All statistics are accumulated in double precision, independently of the model
dtype (the variance is a difference of second moments and would cancel
catastrophically in single precision for outputs of large magnitude, such as
total energies). The measurable error is still limited by the precision of the
model outputs themselves: for a float32 model, deviations below roughly
:math:`10^{-7}` times the magnitude of the output are dominated by round-off,
and errors reported at that scale do not indicate symmetry breaking.

The :py:func:`per_system_equivariance_rmse` helper applies the same reduction
to an existing output dictionary.

Character projections
---------------------

When ``compute_character_projections=True``, the output additionally contains
``<name>_character_projection``. These decompose the output into the
isotypical components of :math:`O(3)`, labeled by ``chi_lambda`` and
``chi_sigma`` in the keys, and describe how the output is distributed across
the irreducible sectors of the group; see the class documentation below for
the definitions.

Each stored value is the squared norm :math:`\| P_{\lambda,\sigma}\, x \|^{2}`
of the projection of the output onto the :math:`(\lambda,\sigma)` isotypical
subspace, resolved per sample (and per component and property of the output).
Dividing by ``<name>_norm_squared`` gives the fraction of the output's squared
norm in that sector, as done by :py:func:`per_system_character_fractions`.

Computing the projections requires ``max_o3_lambda_character`` to be set, on a
quadrature grid able to integrate them exactly (``max_o3_lambda_grid`` at
least ``2 * max_o3_lambda_character``, satisfied by the default); a
:py:exc:`ValueError` is raised otherwise.

Reference
---------

.. autoclass:: metatomic.torch.symmetrized_model.SymmetrizedModel
    :members:

.. autofunction:: metatomic.torch.symmetrized_model.per_system_equivariance_rmse

.. autofunction:: metatomic.torch.symmetrized_model.per_system_character_fractions

.. autofunction:: metatomic.torch.symmetrized_model.get_euler_angles_quadrature

.. autofunction:: metatomic.torch.symmetrized_model.get_rotation_quadrature
