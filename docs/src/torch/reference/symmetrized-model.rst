.. _symmetrized-model:

O(3)-symmetrized models
=======================

The :py:mod:`metatomic.torch.symmetrized_model` module wraps an exported
:py:class:`~metatomic.torch.AtomisticModel` with finite-quadrature O(3)
averaging and equivariance diagnostics. Ordinary outputs are averaged over
rotated and inverted copies of each input. Additional output names request an
equivariance variance or squared character-projection contributions of the
model response.

Output requests
---------------

The requested output name selects both the source output and the calculation:

.. list-table::
   :header-rows: 1

   * - Requested and returned name
     - Result
   * - ``<name>``
     - O(3) average of the underlying ``<name>`` output
   * - ``o3::variance::<name>``
     - component-averaged equivariance variance of ``<name>``
   * - ``o3::character_projection::<name>``
     - unnormalized squared character-projection contributions of ``<name>``

``<name>`` is preserved verbatim. It can therefore be a standard quantity, a
variant such as ``energy/pbe``, or a custom name such as
``mtt::feature::node``. For example,
``o3::variance::energy/pbe`` evaluates the underlying ``energy/pbe`` output.

Average and variance
--------------------

For an input :math:`x`, an O(3) operation :math:`g`, and the target
representation :math:`\rho_\alpha`, define the response transformed back to the
input frame as

.. math::

    z_\alpha(g;x) = \rho_\alpha(g^{-1}) f(gx).

The ordinary result is the normalized Haar average

.. math::

    \Pi_\alpha(f,x)
    = \int_{\mathrm{O}(3)} z_\alpha(g;x)\,\mathrm{d}\mu(g).

For a TensorMap block with component multiplicity :math:`d`, the corresponding
variance output contains

.. math::

    v_\alpha(f,x)
    = \frac{1}{d}\left[
      \int_{\mathrm{O}(3)} \lVert z_\alpha(g;x) \rVert_2^2\,
      \mathrm{d}\mu(g)
      - \lVert \Pi_\alpha(f,x) \rVert_2^2
      \right].

This value is returned separately for every sample and property. It has no
component axes, and it is not reduced across samples or square-rooted. A
weighted mean of these values over a group of samples, followed by a square
root, gives a block-wise equivariance RMSE.

TensorMap representation
------------------------

An averaged output retains the physical schema declared by the source model.
For diagnostics, the standard quantities are represented as follows:

.. list-table::
   :header-rows: 1

   * - Source quantity
     - Diagnostic target keys
   * - ``energy``, ``energy_ensemble``, ``energy_uncertainty``
     - ``o3_lambda=0``, ``o3_sigma=1``
   * - ``non_conservative_force``
     - ``o3_lambda=1``, ``o3_sigma=1``
   * - ``non_conservative_stress``
     - ``(o3_lambda, o3_sigma)=(0,1)`` and ``(2,1)``

Variants after ``/`` use the same representation as their base quantity.

Energy-like scalars acquire an ``o3_mu`` component of size one for diagnostics.
Cartesian force components are reordered into the real spherical
:math:`\ell=1` basis described in :ref:`o3-conventions`. Models should provide
symmetric ``non_conservative_stress`` tensors. Stress diagnostics retain only
the scalar trace and symmetric-traceless sectors, silently discarding any
antisymmetric part.

Already-spherical outputs retain their ``o3_lambda`` and ``o3_sigma`` keys and
``o3_mu`` components, and other semantic source keys are preserved. The wrapper
does not infer the physical meaning of a custom output from its shape; in
particular, a custom Cartesian :math:`3\times3` output is not treated as a
symmetric stress.

Character projections
---------------------

Character projections analyze the direct response :math:`u(g;x)=f(gx)`, rather
than the back-transformed response used for averaging. For the character sector
:math:`\beta=(\lambda,\sigma)` with :math:`d_\beta=2\lambda+1`, the squared
projection norm is

.. math::

    B_\beta(u,x)
    = d_\beta \iint_{\mathrm{O}(3)}
      u(g_1;x)^\dagger
      \chi_\beta(g_1g_2^{-1})u(g_2;x)\,
      \mathrm{d}\mu(g_1)\,\mathrm{d}\mu(g_2).

Character results append ``chi_lambda`` and ``chi_sigma`` to the TensorMap
keys. These labels describe the O(3) dependence of the response over the
rotation orbit. They are distinct from ``o3_lambda`` and ``o3_sigma``, which
describe the target representation of the output itself. Target component axes
are retained; summing over them gives the complete component norm in the
equation above.

Quadrature
----------

The deterministic grid combines a Lebedev rule on the sphere, uniformly spaced
in-plane rotations, and both O(3) cosets. Its weights are normalized to sum to
one. A general machine-learning model need not be band-limited, so a finite
grid is not automatically exact. ``max_o3_lambda_grid`` controls the quadrature
resolution, not the representation: increase it until the averages, variances,
and character projections of interest converge.

Reference
---------

.. py:currentmodule:: metatomic.torch.symmetrized_model

.. autoclass:: SymmetrizedModel
    :members:

.. autofunction:: get_rotation_quadrature
