.. _symmetrized-model:

O(3)-symmetrized models
=======================

The :py:class:`metatomic.torch.o3.SymmetrizedModel` class wraps an existing
:py:class:`metatomic.torch.AtomisticModel` with finite-quadrature O(3) averaging
and equivariance diagnostics. Pre-existing outputs of the model are averaged
over rotated and inverted copies of each input.

Models outputs
--------------

:py:class:`metatomic.torch.o3.SymmetrizedModel` adds extra outputs to the model,
computing the equivariance variance or squared character-projection
contributions of the model response.

.. list-table::
   :header-rows: 1

   * - output name
     - Result
   * - ``<name>``
     - O(3) average of the wrapped model's ``<name>`` output
   * - ``o3::variance::<name>``
     - component-averaged equivariance variance of ``<name>``
   * - ``o3::character_projection::<name>``
     - unnormalized squared character-projection contributions of ``<name>``

The above outputs are added for every output of the wrapped model, including
variants (such as ``energy/pbe``) and custom outputs (such as
``custom::feature::node``). For example, ``o3::variance::energy/pbe`` would
compute the equivariance variance of the ``energy/pbe`` output.

Quadrature
----------

The deterministic grid combines a Lebedev rule on the sphere, uniformly spaced
in-plane rotations, and both parities: O(3) splits into two cosets of SO(3),
the proper rotations, and the improper ones (a rotation composed with
inversion). Its weights are normalized to sum to one. A general machine-learning
model need not be band-limited, so a finite grid is not automatically exact.
``max_angular_momentum_grid`` controls the quadrature resolution, not the
representation: increase it until the averages, variances, and character
projections of interest converge.

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

The meaning of the variance depends on the structure of the wrapped model's
output. When its blocks carry recognized component labels (``o3_mu``-style
spherical or ``xyz``-style Cartesian axes, see the :ref:`o3-conventions`
documentation), each output is rotated back to the input frame first, and the
variance measures the breaking of *equivariance*. Outputs without such
components cannot be rotated back: their responses are compared as-is across the
quadrature, so their variance measures the deviation from *invariance* only. An
equivariant but unlabelled output --- for example an internal equivariant
feature vector --- can thus report a large variance even when it transforms
correctly.

Variance metadata
~~~~~~~~~~~~~~~~~

The ``o3::variance::<name>`` outputs produced by
:py:class:`metatomic.torch.o3.SymmetrizedModel` have the following metadata
structure:

.. list-table:: Metadata for ``"o3::variance::<name>"``
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``[<keys...>, "o3_lambda", "o3_sigma"]``
    - The keys are the same as the original ``<name>`` output, with
      ``"o3_lambda"`` and ``"o3_sigma"`` dimensions added if they are not
      already present, and the ``_ = 0`` dummy key removed if present. The
      ``"o3_lambda"`` dimension contains the angular momentum of each block, and
      the ``"o3_sigma"`` dimension contains its parity under inversion.

  * - samples
    - ``[<samples...>]``
    - the samples are the same as the original ``<name>`` output.

  * - components
    -
    - Since the variance is computed over the ``o3_mu`` components of each
      block, the resulting TensorMap does not have any component axes.

  * - properties
    - ``[<properties...>]``
    - the properties are the same as the original ``<name>`` output.

When computing the variance of :ref:`standard quantities <standard-quantities>`,
the data is first converted to spherical representation as follow:

- Scalar quantities (such as :ref:`energy <energy-quantity>`, :ref:`charge
  <charge-quantity>`, *etc.*) gain a ``o3_lambda=0, o3_sigma=1`` key, as well as
  an ``o3_mu`` component of size one.
- Cartesian vector quantities (such as :ref:`non-conservative force
  <non-conservative-force-quantity>`) gain a ``o3_lambda=1, o3_sigma=1`` key,
  and their ``xyz`` components are replaced by an ``o3_mu`` component of size
  three.
- Cartesian rank-2 tensor quantities (such as :ref:`non-conservative stress
  <non-conservative-stress-quantity>`) gain ``o3_lambda=0, o3_sigma=1``,
  ``o3_lambda=1, o3_sigma=-1``, and ``o3_lambda=2, o3_sigma=1`` keys, and their
  ``xyz_1`` and ``xyz_2`` components are replaced by an ``o3_mu`` component of
  size one, three, and five, respectively.
- Already-spherical outputs retain their ``o3_lambda`` and ``o3_sigma`` keys and
  ``o3_mu`` components, and other keys are preserved.
- Custom outputs with components that do not match the :ref:`convention <o3-conventions>` are
  not supported, and will raise an error when the variance is requested.

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

Character results append ``chi_lambda`` and ``chi_sigma`` to the TensorMap keys.
These labels describe the O(3) dependence of the response over the rotation
orbit. They are distinct from ``o3_lambda`` and ``o3_sigma``, which describe the
target representation of the output itself. Any other pre-existing component
axes are retained; summing over them gives the complete component norm in the
equation above.

Character projections metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``o3::character_projection::<name>`` outputs produced by
:py:class:`metatomic.torch.o3.SymmetrizedModel` have the following metadata
structure:

.. list-table:: Metadata for ``"o3::variance::<name>"``
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``[<keys...>, "o3_lambda", "o3_sigma", "chi_lambda", "chi_sigma"]``
    - The keys are the same as the original ``<name>`` output, with
      ``"o3_lambda"`` and ``"o3_sigma"`` dimensions added if they are not
      already present, the ``_ = 0`` dummy key removed if present, and
      ``"chi_lambda"`` and ``"chi_sigma"`` keys added.

  * - samples
    - ``[<samples...>]``
    - the samples are the same as the original ``<name>`` output.

  * - components
    - ``[<components...>, "o3_mu"]``
    - The components are the same as the original ``<name>`` output, with an
      additional ``"o3_mu"`` component of size :math:`2\lambda+1` added for each
      block. This ``"o3_mu"`` component replaces existing ``"xyz"`` or
      ``"o3_mu"`` components, if present.

  * - properties
    - ``[<properties...>]``
    - the properties are the same as the original ``<name>`` output.

When computing character projections, the data is first converted to spherical
representation like for the variance, with the exception that arbitrary
pre-existing component axes are allowed and will not raise an exception. These
are retained as-is in the output.

API reference
-------------

.. autoclass:: metatomic.torch.o3.SymmetrizedModel
    :members:
