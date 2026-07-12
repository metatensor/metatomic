.. _symmetrized-model:

Symmetrized models
==================

The :py:mod:`metatomic.torch.symmetrized_model` module provides
:py:class:`SymmetrizedModel`, an evaluation wrapper that approximates the
:math:`O(3)` Haar projection of an atomistic model with a deterministic finite
quadrature. It also reports orientation-dependent diagnostics.

The wrapped model can follow the :py:class:`metatomic.torch.ModelInterface`
convention or be an exported :py:class:`metatomic.torch.AtomisticModel`,
including one loaded with :py:func:`metatomic.torch.load_atomistic_model`.

Before constructing the wrapper, install the optional quadrature dependency:
``python -m pip install "scipy>=1.15"``.

A minimal evaluation looks like this (``systems`` is a list of
:py:class:`~metatomic.torch.System` objects in the model's length unit):

.. code-block:: python

    from metatomic.torch import ModelOutput, load_atomistic_model
    from metatomic.torch.symmetrized_model import SymmetrizedModel

    base = load_atomistic_model("model.pt").eval()
    model = SymmetrizedModel(
        base,
        max_o3_lambda_target=2,
        max_o3_lambda_grid=8,
        batch_size=16,
    )
    result = model(systems, {"energy": ModelOutput(sample_kind="system")})
    energy_mean = result["energy_l0_mean"]  # a TensorMap
    energy_values = energy_mean.block().values

Repeat with a larger ``max_o3_lambda_grid`` and compare the returned mean and
diagnostics before relying on a finite-grid result.

The projector, character decomposition, O(3) convention, and product-grid
theorem follow Domina *et al.*, `How unconstrained machine-learning models
learn physical symmetries <https://arxiv.org/abs/2603.24638v1>`_, especially
main-text Sec. II and SI Secs. IV–V.

Projector and finite quadrature
-------------------------------

For an input :math:`x`, group element :math:`g`, and orthogonal target
representation :math:`\rho`, define the back-rotated response

.. math::

    z(g) = \rho(g^{-1}) f(gx).

The symmetrized output is the Haar average of :math:`z`. The finite grid
combines a Lebedev rule on :math:`S^2`, equispaced in-plane rotations, and the
proper and improper cosets of :math:`O(3)` to approximate

.. math::

    \overline f(x) = \int_{O(3)} z(g)\,\mathrm{d}\mu(g).

The finite sum equals this integral when the relevant orbit response and its
products are band-limited within the grid. A general neural-network response
need not satisfy such a bound, so increase ``max_o3_lambda_grid`` until the
mean, variance, and any character fractions converge.

The implementation accumulates moments around a fixed grid response for
numerical stability. For normalized finite-grid weights :math:`q_i`, choose
:math:`r=z(g_0)` and set :math:`y_i=z(g_i)-r`. It computes

.. math::

    \mu = r + \sum_i q_i y_i,
    \qquad
    V = \sum_i q_i \lVert y_i\rVert^2
        - \left\lVert\sum_i q_i y_i\right\rVert^2.

This is algebraically the usual mean and component-summed variance. It applies
to tensorial outputs because every :math:`z(g_i)` is already in the same
back-rotated representation frame. The reference is only a numerical device:
it is not a shift of the physical input or output and does not alter the
projector. Character projections instead use the direct response and are not
centered this way.

Because Lebedev rules can contain signed weights, an under-resolved finite sum
can make the nominal variance or squared norm negative even though the exact
Haar quantities are non-negative. The wrapper also accumulates
:math:`\sum_i |q_i|\lVert y_i\rVert^2` as a round-off scale. A negative residual
within a bound proportional to the number of grid points and float64 epsilon is
set to zero; a materially negative or non-finite diagnostic raises
:py:class:`ValueError` and asks for a larger grid. This prevents severe aliasing
from being reported as zero equivariance error.

Angular parameters
~~~~~~~~~~~~~~~~~~

The three angular parameters are deliberately independent:

- ``max_o3_lambda_target`` validates spherical components returned directly by
  the base model and provides the target-only default-grid heuristic. Work is
  sized from the ranks actually returned; this parameter is not a response
  bandwidth.
- ``max_o3_lambda_character`` selects the character sectors to compute. It does
  not assert that the response has no content above the cutoff.
- ``max_o3_lambda_grid`` is the declared integration degree :math:`D`. The
  wrapper selects a Lebedev rule of at least that degree and uses :math:`D+1`
  in-plane angles.

The product-quadrature theorem requires Lebedev degree :math:`2L` and at least
:math:`2L+1` in-plane points for products of Wigner-D functions through
bandwidth :math:`L`. Character projections through :math:`L` therefore require
``max_o3_lambda_grid >= 2 * max_o3_lambda_character``. Exactness still requires
the response itself to be covered; otherwise check convergence.

Constructor defaults, limits, and validation are documented in the API
reference below. Quadrature construction requires `SciPy`_ 1.15 or newer.

.. _SciPy: https://scipy.org/

Evaluation, devices, and units
------------------------------

``SymmetrizedModel`` is intended for evaluation and diagnostics, not training
or model export. Calls without ``compute_gradients=True`` run under
``torch.no_grad()`` and return detached results. Put the base model in evaluation
mode so stochastic layers are not mistaken for symmetry breaking.

Input Systems and the base model must use compatible devices. Do not downcast
the wrapper: its float64 buffers carry the quadrature accuracy. This wrapper
does not support MPS; that restriction does not apply to general metatomic O(3)
or metatrain use.

For an exported :py:class:`~metatomic.torch.AtomisticModel`, the wrapper passes
an empty input ``length_unit``. Positions, cells, and neighbor-list vectors must
therefore already use the model length unit. Requested custom System data still
uses the normal AtomisticModel conversion when both the TensorMap and model
input request specify units. Raw modules receive the transformed Systems
directly and define their own unit semantics.

Required neighbor lists and custom data must already be attached. Cartesian and
spherical custom data are transformed according to :ref:`o3-conventions`;
spherical input rank is inferred from its ``o3_mu*`` axes independently of the
output ceiling.

For every base-model TensorBlock, each rotated copy must return the same
non-``system`` sample labels in the same order, with identical keys, components,
and properties. The ``system`` column itself is located by name and need not be
the first sample column. Changing the sample layout between rotated copies is
ambiguous and raises an error.

Every requested output must be returned by the base model. Requests are
forwarded unchanged, including requested output units: exported models perform
their normal output conversion, while raw modules define their own behavior.
New TensorMaps created by the wrapper do not carry machine-readable unit
metadata. Numerically, means use the base output unit; variances, squared norms,
and character projections use its square. Autograd-derived forces use the
returned energy unit divided by the input-position unit, and stress uses the
energy unit divided by the input-length unit cubed. For exported models, that
input length unit is the model length unit described above.

Explicit gradients attached to TensorMap blocks are not supported and are
rejected before reduction. ``compute_gradients=True`` is a separate autograd
path that derives ordinary force and stress outputs from an energy; it does not
preserve attached TensorMap gradients.

Output schema and bases
-----------------------

Standard quantities, including variants after ``/``, are decomposed as follows.
The complete requested name is preserved before the ``_l<rank>`` suffix.

.. list-table::
   :header-rows: 1

   * - Requested base quantity
     - Generated name(s)
     - Irrep key(s)
   * - ``energy``, ``energy_ensemble``, ``energy_uncertainty``
     - ``<name>_l0``
     - ``o3_lambda=0``, ``o3_sigma=1``
   * - ``forces``, ``non_conservative_force``, ``non_conservative_forces``
     - ``<name>_l1``
     - ``o3_lambda=1``, ``o3_sigma=1``
   * - ``stress``, ``non_conservative_stress``
     - ``<name>_l0`` and ``<name>_l2``
     - ``(0, 1)`` and ``(2, 1)``

For example, ``energy/pbe`` produces ``energy/pbe_l0``. Generated spherical
blocks carry an ``o3_mu`` component. When the source has only metatensor's
dummy ``_`` key, their key names are exactly ``o3_lambda`` and ``o3_sigma``;
semantic source keys are preserved before these two irrep keys. Custom
quantities retain their original layout and are back-rotated from their
Cartesian or spherical metadata. Requested names must remain distinct after
these suffixes are generated; ambiguous collisions are rejected before model
evaluation.

The force conversion maps Cartesian ``(x, y, z)`` to
``o3_mu=(-1, 0, 1) = (y, z, x)``. For a stress :math:`S`, let
:math:`S^{\mathrm{sym}}=(S+S^T)/2`. The :math:`l=0` value is the raw trace
:math:`t=\mathrm{tr}(S)`, and the :math:`l=2` values in
``o3_mu=(-2, -1, 0, 1, 2)`` order are

.. math::

    c = \left(
        S^{\mathrm{sym}}_{xy},
        S^{\mathrm{sym}}_{yz},
        \frac{2S_{zz}-S_{xx}-S_{yy}}{2\sqrt{3}},
        S^{\mathrm{sym}}_{xz},
        \frac{S_{xx}-S_{yy}}{2}
    \right),

with

.. math::

    \lVert S^{\mathrm{sym}}\rVert_F^2 = \frac{t^2}{3} + 2\lVert c\rVert^2.

Only the trace and symmetric-traceless parts used for physical stress are
returned. A skew input component is omitted; callers needing torque or
antisymmetric diagnostics must analyze it separately.

For every decomposed name ``<name>``, the result contains:

- ``<name>_mean``: the finite-grid symmetrized mean;
- ``<name>_var``: the component-summed orientation variance;
- ``<name>_norm_squared``: the average squared component norm.

Back-rotation and accumulation use float64 even when the base model runs in
float32. This limits cancellation but can not recover information already
rounded in the model output.

Autograd-derived forces and stress
-----------------------------------

With ``compute_gradients=True``, ``energy_name`` must identify a returned,
single-block energy-like TensorMap. Its values are summed before autograd, and
the wrapper derives

.. math::

    F = -\frac{\partial E}{\partial r}, \qquad
    S = \frac{1}{V}\frac{\partial E}{\partial\epsilon}.

The names ``forces`` and ``stress`` are reserved for these derived outputs in
this mode and can not also be requested from the base model. Forces are returned
for every System, subject to ``selected_atoms``. Three-dimensional stress is
returned only for Systems whose three PBC flags are all true. A mixed batch
therefore contains stress samples only for fully periodic Systems. Their cell
must have finite, nonzero volume; non-periodic and partially periodic Systems
have no 3D stress entry.

Following the metatrain composition-model convention, a detached energy emits a
:py:class:`RuntimeWarning`, is treated as constant, and produces exact zero
derivatives. This is correct for genuinely coordinate-independent outputs such
as type-only composition energies; autograd can not distinguish them from a
position-dependent energy that was detached accidentally. A graph-connected
energy that does not use a particular position or strain target also produces
zero for that target. Callers must therefore keep the graph connected for every
genuinely coordinate-dependent energy.

Equivariance error
------------------

:py:meth:`SymmetrizedModel.equivariance_error` reduces each variance to a
per-system value with :py:func:`per_system_equivariance_rmse`. It preserves keys
and properties, divides by the total component multiplicity, averages over the
system's samples, and then takes the square root. The result is comparable to an
element-wise accuracy RMSE in the same returned basis.

For the raw mean-square and variance sums on the same normalized finite grid, an
exactly equivariant reference satisfies

.. math::

    \left\langle \mathrm{RMSE}_{\mathrm{accuracy}}^2 \right\rangle
    =
    \mathrm{RMSE}_{\mathrm{accuracy}}^2[\text{mean}]
    +
    \mathrm{RMSE}_{\mathrm{equivariance}}^2.

This algebraic identity does not prove that the grid equals the Haar integral.
Some valid Lebedev rules contain signed weights, so an under-resolved finite-grid
variance can be materially negative. The wrapper rejects this state using the
scale-aware check described above; the reduction helper also rejects negative or
non-finite manually supplied variances. Increase ``max_o3_lambda_grid`` until
the diagnostic converges.

To pool systems of different sizes, combine squared per-system values with their
sample counts and then take the square root. A plain mean is not the pooled RMSE.
Systems without samples in a block receive zero by convention; this is not a
measured zero error.

Character projections
---------------------

Characters analyze the direct-frame response

.. math::

    u(g) = f(gx),

not the back-rotated :math:`z(g)`. An exactly equivariant polar vector can
therefore have zero equivariance error while carrying direct-frame content in
the :math:`(1,+1)` sector. The projector onto sector
:math:`(\ell,\sigma)` is

.. math::

    (P_{\ell,\sigma}u)(g)
    = (2\ell+1)
      \int_{O(3)} \chi_{\ell,\sigma}(h^{-1}g)u(h)\,\mathrm{d}\mu(h).

For a proper rotation :math:`R` and inversion :math:`i`, the character convention
is

.. math::

    \chi_{\ell,\sigma}(R) = \operatorname{tr}D^\ell(R),
    \qquad
    \chi_{\ell,\sigma}(iR)
    = \sigma(-1)^\ell\operatorname{tr}D^\ell(R).

With ``compute_character_projections=True``, every decomposed output also has
``<name>_character_projection``. Keys append ``chi_lambda`` and ``chi_sigma``;
values are squared Fourier-coefficient norms and are non-negative by
construction. :py:func:`per_system_character_fractions` aggregates and
normalizes them by ``<name>_norm_squared``.

Fractions sum to one only when the response is band-limited, the grid resolves
the required products, and the character cutoff contains the full response
bandwidth. Negative or non-finite norms and sector weights are rejected instead
of being repaired with an absolute value. A zero norm paired with all-zero
sector weights returns all-zero fractions; a zero norm with nonzero sector
weight is inconsistent and raises.

Performance and memory
----------------------

If the selected Lebedev rule has :math:`M` nodes and grid degree :math:`D`, each
original System creates :math:`2M(D+1)` rotated copies and makes

.. math::

    2\left\lceil\frac{M(D+1)}{\mathtt{batch\_size}}\right\rceil

base-model calls. ``batch_size`` changes peak batch memory and call size, not
the total evaluated copies. Original Systems are processed one at a time, and
geometry and neighbor vectors are transformed in batches where possible.

Mean, variance, norm, and the round-off scale are streamed around the fixed
reference, so persistent statistical state scales with output size rather than
orbit size, while the live model-prediction payload scales with ``batch_size``.
Setting ``storage_device`` can move back-rotation, accumulation, and results off
the model device; the base-model evaluation and quadrature grid remain there.

The grid itself grows with ``max_o3_lambda_grid`` and is not bounded by
``batch_size`` or ``storage_device``. Spherical outputs and character
projections additionally use a lazy Wigner-D cache. ``wigner_cache_max_bytes``
bounds retained Wigner tensors, not total forward memory; setting it to zero
uses a numerically equivalent batched fallback. Wigner-D construction is
CPU-backed and completed stacks are transferred in batches for CUDA evaluation.

For practical tuning, first choose the smallest grid that gives converged
results, then increase ``batch_size`` until throughput stops improving or device
memory becomes limiting. Reduce ``wigner_cache_max_bytes`` if persistent cache
memory is a concern. Character projections grow rapidly with the requested
rank and are best run on representative data. ``compute_gradients=True`` adds a
backward pass for every rotated batch and should be enabled only when force or
stress diagnostics are required.

Reference
---------

.. autoclass:: metatomic.torch.symmetrized_model.SymmetrizedModel
    :members:

.. autofunction:: metatomic.torch.symmetrized_model.per_system_equivariance_rmse

.. autofunction:: metatomic.torch.symmetrized_model.per_system_character_fractions

.. autofunction:: metatomic.torch.symmetrized_model.get_rotation_quadrature
