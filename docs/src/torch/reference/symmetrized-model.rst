.. _symmetrized-model:

O(3)-symmetrized models
=======================

The :py:mod:`metatomic.torch.symmetrized_model` module wraps an exported
:py:class:`~metatomic.torch.AtomisticModel` with finite-quadrature O(3)
averaging and equivariance diagnostics. Ordinary outputs are averaged over
rotated and inverted copies of each input. Additional output names request an
equivariance variance or squared character-projection contributions of the
model response.

Constructing a wrapper requires SciPy 1.15 or newer for its Lebedev quadrature.
SciPy is not required to evaluate a wrapper that has already been saved.

Wrapping and evaluating a model
-------------------------------

Use
:py:meth:`~metatomic.torch.symmetrized_model.SymmetrizedModel.wrap` to retain
the model metadata, capabilities, requested neighbor lists, and requested
custom inputs:

.. code-block:: python

    from metatomic.torch import (
        ModelEvaluationOptions,
        ModelOutput,
        load_atomistic_model,
    )
    from metatomic.torch.symmetrized_model import SymmetrizedModel

    base_model = load_atomistic_model("model.pt")
    model = SymmetrizedModel.wrap(
        base_model,
        max_o3_lambda_target=2,
        max_o3_lambda_character=3,
    )

    options = ModelEvaluationOptions(
        length_unit="angstrom",
        outputs={
            "energy": ModelOutput(unit="eV", sample_kind="system"),
            "o3::variance::energy": ModelOutput(
                unit="(eV)^2",
                sample_kind="system",
            ),
            "o3::character_projection::energy": ModelOutput(
                unit="(eV)^2",
                sample_kind="system",
            ),
        },
    )
    results = model(systems, options, check_consistency=True)
    model.save("symmetrized-model.pt")

The requested units must be compatible with the capabilities of the wrapped
model. The example assumes that its length and energy units are ``angstrom``
and ``eV``.

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

Several calculations for the same source output share the same model
predictions. Their :py:attr:`~metatomic.torch.ModelOutput.sample_kind` values
must agree. The returned dictionary contains exactly the requested names.
Character-projection requests are available only when
``max_o3_lambda_character`` was set during construction.

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
component axes, and it is not reduced across samples or square-rooted. A later
evaluation operation can obtain a block-wise RMSE for a group of samples
:math:`G` as

.. math::

    \operatorname{RMSE}_{G}
    = \sqrt{
      \frac{\sum_{s\in G} w_s v_\alpha(f,x_s)}
           {\sum_{s\in G} w_s}
      }.

Different TensorMap blocks, including different irreducible sectors, remain
separate by default. Combining blocks with different component multiplicities
requires weighting each block by that multiplicity.

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
antisymmetric part. A custom Cartesian :math:`3\times3` output is not assumed
to be a stress or to be symmetric.

Already-spherical outputs retain their ``o3_lambda`` and ``o3_sigma`` keys and
``o3_mu`` components. Other semantic source keys are preserved. The wrapper
does not infer the physical meaning of a custom output from its shape.

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

Quadrature and angular-momentum limits
--------------------------------------

The deterministic grid combines a Lebedev rule on the sphere, uniformly spaced
in-plane rotations, and both O(3) cosets. Its weights are normalized to sum to
one. A general machine-learning model need not be band-limited, so a finite grid
is not automatically exact. Increase ``max_o3_lambda_grid`` until the averages,
variances, and character projections of interest converge. A materially
negative or non-finite squared diagnostic is rejected instead of being reported
as a physical result.

The constructor keeps four angular-momentum limits separate:

- ``max_o3_lambda_input`` is the largest spherical rank accepted in custom
  :py:class:`~metatomic.torch.System` data. Its default of zero still permits
  Cartesian vectors and tensors; it restricts only already-spherical component
  axes.
- ``max_o3_lambda_target`` is the largest spherical rank accepted in outputs
  that must be transformed back to the input frame.
- ``max_o3_lambda_character`` is the largest character sector included in a
  character-projection result. ``None`` disables these outputs.
- ``max_o3_lambda_grid`` controls the quadrature resolution, not an input or
  output representation.

The :py:class:`~metatomic.torch.ModelOutput` declarations returned by a model's
``requested_inputs()`` do not specify the spherical ranks that may occur in the
corresponding TensorMaps. The input limit must therefore be supplied before
export so that all required Wigner-D matrices can be serialized. At runtime, an
already-spherical custom input or an output requiring back-rotation is rejected
when its rank exceeds the corresponding declared limit; the error identifies
the offending name and rank.

Execution, devices, and gradients
---------------------------------

The wrapper supports CPU and CUDA execution with float32 or float64 model
values. It stores quadrature and Wigner-D buffers in float64 and converts final
results back to the model dtype. Move a wrapped model between supported devices
without changing these buffer dtypes. MPS is not supported.

``batch_size`` controls how many transformed copies are passed to the source
model in one call. It does not change the quadrature or its result. The
statistical accumulators are streamed, but the rotation grid and packed
Wigner-D matrices are persistent buffers. Construction rejects packed
Wigner-D storage larger than ``max_wigner_storage_bytes``.

Explicit TensorBlock gradients are not supported in requests or source
outputs. Ordinary PyTorch autograd remains available through the returned
values. When an input requires gradients, differentiating an averaged result
retains the source-model activations from all quadrature batches;
``batch_size`` does not bound their total size. Use
:py:func:`torch.inference_mode` or :py:func:`torch.no_grad` when derivatives are
not required.

Reference
---------

.. py:currentmodule:: metatomic.torch.symmetrized_model

.. autoclass:: SymmetrizedModel
    :members:

.. autofunction:: get_rotation_quadrature
