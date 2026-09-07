"""
:py:class:`WeightedSum`, which wraps a model exposing several "proper" head outputs
for the same physical quantity (for example several energies computed with
different functionals, such as ``"energy/pbe"``, ``"energy/r2scan"`` and
``"energy/lda"``) and adds a new output computing a fixed linear combination of
these heads.
"""

import warnings
from typing import Dict, List, Optional

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorMap

from . import (
    ModelCapabilities,
    ModelOutput,
    NeighborListOptions,
    System,
)
from .model import (
    AtomisticModel,
    ModelInterface,
)


class WeightedSum(torch.nn.Module):
    """
    Wrap a model with a new output computing a fixed weighted sum of several of its
    existing outputs.

    The proper heads combined in the sum (for example ``"energy/pbe"``,
    ``"energy/r2scan"`` and ``"energy/lda"``) are all requested from the wrapped
    model in a single call, and the weighted-sum output is built by directly
    combining the resulting :py:class:`TensorMap` values with :py:func:`mts.add` and
    :py:func:`mts.multiply`. This does not evaluate the model more than once and does
    not detach or otherwise interrupt the autograd graph, so the weighted-sum output
    stays connected to the same computational graph as each individual head. If the
    input systems require gradients, a single ``backward()`` call through the
    weighted-sum output therefore produces forces and stresses that are exactly the
    weighted sum of the forces and stresses of the individual heads.

    :param model: underlying :py:class:`ModelInterface`. The :py:meth:`wrap` method
        obtains this module from :py:attr:`AtomisticModel.module`.
    :param output_name: name of the new output computing the weighted sum, e.g.
        ``"energy"``.
    :param weights: mapping from the name of an existing head output of ``model``
        (e.g. ``"energy/pbe"``) to its fixed coefficient in the weighted sum.
    """

    _output_name: str
    _weights: Dict[str, float]
    _requested_inputs: Dict[str, ModelOutput]
    _requested_neighbor_lists: List[NeighborListOptions]

    def __init__(
        self,
        model: ModelInterface,
        output_name: str,
        weights: Dict[str, float],
    ):
        super().__init__()

        if len(weights) == 0:
            raise ValueError(
                "`weights` must contain at least one head output name to combine"
            )

        self._model = model
        self._output_name = output_name
        self._weights = weights
        self._requested_inputs = {}
        self._requested_neighbor_lists = []

    @staticmethod
    def wrap(
        model: AtomisticModel,
        output_name: str,
        weights: Dict[str, float],
        normalize_coefficients: bool = False,
    ) -> AtomisticModel:
        """
        Wrap an exported model, adding a new ``output_name`` output computing the
        fixed weighted sum of the outputs named in ``weights``.

        The returned model retains every output declared by ``model`` under its
        original name, and adds the new weighted-sum output. The original metadata,
        requested inputs, neighbor lists, and compatible capabilities are preserved.

        :param model: the :py:class:`AtomisticModel` to wrap
        :param output_name: name of the new weighted-sum output to add, e.g.
            ``"energy"``
        :param weights: mapping from the name of an existing head output of
            ``model`` (e.g. ``"energy/pbe"``) to its fixed coefficient in the
            weighted sum
        :param normalize_coefficients: if ``True``, rescale ``weights`` so they sum
            to one, by dividing every coefficient by their sum. This works with
            negative coefficients as well, as long as the sum of all coefficients is
            not zero; a zero sum (e.g. a pure difference of two heads) cannot be
            normalized and raises a ``ValueError``. If the sum is negative, every
            coefficient's sign is flipped by the normalization (a ``UserWarning``
            is emitted in this case).
        """
        if not isinstance(model, AtomisticModel):
            raise TypeError("model must be an AtomisticModel")

        if len(weights) == 0:
            raise ValueError(
                "`weights` must contain at least one head output name to combine"
            )

        if normalize_coefficients:
            coefficients_sum = sum(weights.values())
            if coefficients_sum == 0:
                raise ValueError(
                    "the sum of `weights` is zero, they can not be normalized to "
                    "sum to one"
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
                "this model already has an output named '"
                + output_name
                + "', which conflicts with the weighted-sum output"
            )

        reference: Optional[ModelOutput] = None
        for head_name in weights.keys():
            if head_name not in capabilities.outputs:
                raise ValueError(
                    "this model does not have a '"
                    + head_name
                    + "' output, which is required to compute the '"
                    + output_name
                    + "' weighted sum"
                )

            head_output = capabilities.outputs[head_name]
            if reference is None:
                reference = head_output
            else:
                if head_output.sample_kind != reference.sample_kind:
                    raise ValueError(
                        "all heads combined in a weighted sum must share the same "
                        f"sample_kind; got '{reference.sample_kind}' and "
                        f"'{head_output.sample_kind}' for '{head_name}'"
                    )
                if head_output.unit != reference.unit:
                    raise ValueError(
                        "all heads combined in a weighted sum must share the same "
                        f"unit; got '{reference.unit}' and '{head_output.unit}' "
                        f"for '{head_name}'"
                    )
        assert reference is not None

        wrapper = WeightedSum(model.module, output_name, weights)
        # private field: the as-declared inputs, deliberately without deprecation
        # aliases
        wrapper._requested_inputs = {
            name: requested_input
            for name, requested_input in model._requested_inputs.items()
        }
        # copy the options: constructing the AtomisticModel below mutates them by
        # adding requestors and setting the length unit
        for options in model.requested_neighbor_lists():
            copied_options = NeighborListOptions(
                options.cutoff,
                options.full_list,
                options.strict,
            )
            for requestor in options.requestors():
                copied_options.add_requestor(requestor)
            wrapper._requested_neighbor_lists.append(copied_options)

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

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        """Return the neighbor lists requested by the wrapped model."""
        return self._requested_neighbor_lists

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        """Return the custom System data requested by the wrapped model."""
        return self._requested_inputs

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
        if len(requested.explicit_gradients) != 0:
            raise ValueError(
                "WeightedSum does not support explicit gradients for the "
                f"'{self._output_name}' output"
            )

        # everything the caller asked for, other than the weighted sum itself
        model_outputs: Dict[str, ModelOutput] = {}
        for name, output in outputs.items():
            if name != self._output_name:
                model_outputs[name] = output

        # make sure all the heads entering the weighted sum are requested from the
        # wrapped model as well, without evaluating it more than once
        for head_name in self._weights.keys():
            if head_name not in model_outputs:
                model_outputs[head_name] = ModelOutput(
                    unit=requested.unit,
                    sample_kind=requested.sample_kind,
                    explicit_gradients=[],
                )

        raw_outputs = self._model(systems, model_outputs, selected_atoms)

        weighted_sum: Optional[TensorMap] = None
        for head_name, weight in self._weights.items():
            if head_name not in raw_outputs:
                raise ValueError(
                    "underlying model did not return the requested head "
                    f"'{head_name}' needed to compute the '{self._output_name}' "
                    "weighted sum"
                )
            contribution = mts.multiply(raw_outputs[head_name], weight)
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
