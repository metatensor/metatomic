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

    :param model: underlying :py:class:`ModelInterface`. The :py:meth:`wrap` method
        obtains this module from :py:attr:`AtomisticModel.module`.
    :param output_name: name of the new output computing the weighted sum, e.g.
        ``"energy/mix"``.
    :param weights: mapping from the name of a existing output variants of ``model``
        (e.g. ``"energy/pbe"``) to its fixed coefficient in the weighted sum.
    """

    _output_name: str
    _weights: Dict[str, float]

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
        self._output_name = output_name
        self._weights = weights

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
        """Evaluate the wrapped model and the requested weighted sum."""
        if self._output_name not in outputs:
            return self._model(systems, outputs, selected_atoms)

        requested = outputs[self._output_name]

        # everything the caller asked for, other than the weighted sum itself
        model_outputs: Dict[str, ModelOutput] = {}
        for name, output in outputs.items():
            if name != self._output_name:
                model_outputs[name] = output

        # make sure all the variants entering the weighted sum are requested from the
        # wrapped model as well, without evaluating it more than once
        for variant_name in self._weights.keys():
            if variant_name not in model_outputs:
                model_outputs[variant_name] = ModelOutput(
                    unit=requested.unit,
                    sample_kind=requested.sample_kind,
                    explicit_gradients=[],
                )

        raw_outputs = self._model(systems, model_outputs, selected_atoms)

        weighted_sum: Optional[TensorMap] = None
        for variant_name, weight in self._weights.items():
            contribution = mts.multiply(raw_outputs[variant_name], weight)
            if weighted_sum is None:
                weighted_sum = contribution
            else:
                weighted_sum = mts.add(weighted_sum, contribution)
        assert weighted_sum is not None

        results: Dict[str, TensorMap] = {}
        for name in outputs:
            if name == self._output_name:
                results[name] = weighted_sum
            else:
                results[name] = raw_outputs[name]
        return results
