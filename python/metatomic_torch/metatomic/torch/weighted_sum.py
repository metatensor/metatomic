"""
:py:class:`WeightedSum` wraps a model exposing several variants of the same physical
quantity (for example several energies computed with different functionals, such as
``"energy/pbe"``, ``"energy/r2scan"`` and ``"energy/lda"``) and adds a new output
computing a fixed linear combination of these variants.
"""

import warnings
from typing import Dict, List, Optional

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorMap

from . import (
    ModelCapabilities,
    ModelOutput,
    System,
)
from .model import (
    AtomisticModel,
    ModelInterface,
)


class WeightedSum(torch.nn.Module):
    """
    Wrap a model, adding a new variant computing a fixed weighted sum of several
    existing variants.

    The variants combined in the sum (for example ``"energy/pbe"``, ``"energy/r2scan"``
    and ``"energy/lda"``) are all computed by the wrapped model in a single call. This
    does not detach or otherwise interrupt the autograd graph, so the weighted sum
    output stays connected to the same computational graph as each individual variant.

    A single :py:class:`WeightedSum` can hold several weighted sums. Calling
    :py:meth:`wrap` several times on the same model adds the new sums to the existing
    :py:class:`WeightedSum` instead of nesting a new one around it, and the wrapped
    model is always called from a single place in :py:meth:`forward`. This keeps the
    TorchScript graph the same size regardless of the number of weighted sums.

    :param model: underlying :py:class:`ModelInterface`. The :py:meth:`wrap` method
        obtains this module from :py:attr:`AtomisticModel.module`.
    :param output_name: name of the new output computing the weighted sum, e.g.
        ``"energy/mix"``.
    :param weights: mapping from the name of a existing output variants of ``model``
        (e.g. ``"energy/pbe"``) to its fixed coefficient in the weighted sum.
    """

    # all the weighted sums computed by this module, {output_name: weights}. Calling
    # `wrap` on a model that is already a WeightedSum adds an entry here instead of
    # nesting a new WeightedSum around the old one.
    _sums: Dict[str, Dict[str, float]]

    def __init__(
        self,
        model: ModelInterface,
        output_name: str,
        weights: Dict[str, float],
    ):
        super().__init__()

        if len(weights) == 0:
            raise ValueError(
                "`weights` must contain at least one output name to combine"
            )

        self._model = model
        self._sums = {output_name: weights}

    @staticmethod
    def wrap(
        model: AtomisticModel,
        output_name: str,
        weights: Dict[str, float],
        normalize_coefficients: bool = False,
    ) -> AtomisticModel:
        """
        Wrap an existing model, adding a new ``output_name`` output computing the fixed
        weighted sum of the outputs named in ``weights``.

        The returned model retains every output declared by ``model`` under its original
        name, and adds the new weighted sum output. The original metadata, requested
        inputs, neighbor lists, and compatible capabilities are preserved.

        In particular, the individual variants entering the sum stay accessible, and a
        model declaring both a default quantity and variants of it (for example
        ``"energy"`` and ``"energy/pbe"``) keeps both. At the same time, both a quantity
        and variants of it (for example ``"energy"`` and ``"energy/pbe"``) can be used
        as inputs in the same weighted sum call. If ``output_name`` collides with an
        existing output of ``model``, a ``ValueError`` is raised.

        If ``model`` was itself created by :py:meth:`wrap`, the new weighted sum is
        added to the existing wrapper instead of nesting another one around it. A
        weighted sum can then also use previously defined weighted sums as inputs,
        which are expanded in terms of the outputs of the underlying model.

        :param model: the :py:class:`AtomisticModel` to wrap
        :param output_name: name of the new output to add, e.g. ``"energy/mix"``
        :param weights: mapping from the name of an existing output of ``model`` (e.g.
            ``"energy/pbe"``) to its fixed coefficient in the weighted sum
        :param normalize_coefficients: if ``True``, rescale ``weights`` so they sum to
            one.
        """
        if not isinstance(model, AtomisticModel):
            raise TypeError("model must be an AtomisticModel")

        if len(weights) == 0:
            raise ValueError(
                "`weights` must contain at least one output name to combine"
            )

        if normalize_coefficients:
            coefficients_sum = sum(weights.values())
            if abs(coefficients_sum) < 1e-6:
                raise ValueError(
                    "the sum of `weights` is too close to zero, they can not "
                    "be normalized"
                )
            if coefficients_sum < 0:
                warnings.warn(
                    "the sum of `weights` is negative; normalizing to sum to one "
                    "flips the sign of every coefficient",
                    stacklevel=2,
                )
            weights = {
                name: weight / coefficients_sum for name, weight in weights.items()
            }

        capabilities = model.capabilities()
        if output_name in model._model_capabilities_outputs_names:
            raise ValueError(
                f"this model already has an output named '{output_name}', which "
                "conflicts with the weighted sum output"
            )

        reference: Optional[ModelOutput] = None
        for variant_name in weights.keys():
            if variant_name not in capabilities.outputs:
                raise ValueError(
                    f"this model does not have a '{variant_name}' output, which is "
                    f"required to compute the '{output_name}' weighted sum"
                )

            variant_output = capabilities.outputs[variant_name]
            if reference is None:
                reference = variant_output
            else:
                if variant_output.sample_kind != reference.sample_kind:
                    raise ValueError(
                        "all variants combined in a weighted sum must share the same "
                        f"sample_kind; got '{reference.sample_kind}' and "
                        f"'{variant_output.sample_kind}'"
                    )
                if variant_output.unit != reference.unit:
                    raise ValueError(
                        "all variants combined in a weighted sum must share the same "
                        f"unit; got '{reference.unit}' and '{variant_output.unit}'"
                    )
        assert reference is not None

        # the wrapped module is a child module of `wrapper`, so the AtomisticModel
        # built below finds its requested neighbor lists and inputs on its own
        if isinstance(model.module, WeightedSum):
            # flatten: add the new sum to the existing WeightedSum instead of nesting.
            # Nested WeightedSum modules each call the wrapped model from two places in
            # `forward`, and the TorchScript optimizing executor inlines both, doubling
            # the graph at each nesting level.
            inner = model.module
            # sums referring to previous weighted sums are expanded in terms of the
            # variants of the underlying model
            flat_weights: Dict[str, float] = {}
            for variant_name, weight in weights.items():
                if variant_name in inner._sums:
                    for sub_name, sub_weight in inner._sums[variant_name].items():
                        flat_weights[sub_name] = (
                            flat_weights.get(sub_name, 0.0) + weight * sub_weight
                        )
                else:
                    flat_weights[variant_name] = (
                        flat_weights.get(variant_name, 0.0) + weight
                    )

            wrapper = WeightedSum(inner._model, output_name, flat_weights)
            sums = dict(inner._sums)
            sums[output_name] = flat_weights
            wrapper._sums = sums
        else:
            wrapper = WeightedSum(model.module, output_name, weights)

        outputs: Dict[str, ModelOutput] = {
            name: capabilities.outputs[name]
            for name in model._model_capabilities_outputs_names
        }
        outputs[output_name] = ModelOutput(
            unit=reference.unit,
            sample_kind=reference.sample_kind,
            explicit_gradients=[],
            description=(
                "Fixed weighted sum of the "
                + ", ".join(f"'{name}'" for name in weights.keys())
                + " outputs of this model."
            ),
        )

        new_capabilities = ModelCapabilities(
            outputs=outputs,
            atomic_types=capabilities.atomic_types,
            interaction_range=capabilities.interaction_range,
            length_unit=capabilities.length_unit,
            supported_devices=capabilities.supported_devices,
            dtype=capabilities.dtype,
        )
        return AtomisticModel(
            wrapper.eval(),
            model.metadata(),
            capabilities=new_capabilities,
        )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        """Evaluate the wrapped model and the requested weighted sums."""
        # everything the caller asked for, other than the weighted sums themselves
        model_outputs: Dict[str, ModelOutput] = {}
        for name, output in outputs.items():
            if name not in self._sums:
                model_outputs[name] = output

        # make sure all the variants entering the requested weighted sums are requested
        # from the wrapped model as well
        for sum_name, weights in self._sums.items():
            if sum_name in outputs:
                requested = outputs[sum_name]
                for variant_name in weights.keys():
                    if variant_name not in model_outputs:
                        model_outputs[variant_name] = ModelOutput(
                            unit=requested.unit,
                            sample_kind=requested.sample_kind,
                            explicit_gradients=[],
                        )

        # the wrapped model is called from a single place, so that the TorchScript
        # optimizing executor only inlines it once
        raw_outputs = self._model(systems, model_outputs, selected_atoms)

        results: Dict[str, TensorMap] = {}
        for name in outputs:
            if name in self._sums:
                weighted_sum: Optional[TensorMap] = None
                for variant_name, weight in self._sums[name].items():
                    contribution = mts.multiply(raw_outputs[variant_name], weight)
                    if weighted_sum is None:
                        weighted_sum = contribution
                    else:
                        weighted_sum = mts.add(weighted_sum, contribution)
                assert weighted_sum is not None
                results[name] = weighted_sum
            else:
                results[name] = raw_outputs[name]
        return results
