"""
Framework-agnostic part of the models computing gradients with automatic
differentiation (:py:class:`TorchModel`, :py:class:`JaxModel`): splitting the requested
gradients between the model and autograd, adding the autograd gradients to the
capabilities, and collecting the pair lists and inputs requested by sub-modules.

This is the same logic as in ``_torch.py``, with the iteration over sub-modules left to
the subclasses (``_named_submodules``). ``TorchModel`` can move on top of this class by
implementing ``_named_submodules`` with :py:meth:`torch.nn.Module.named_modules`.
"""

import functools

from .._capabilities import ModelCapabilities
from .._quantity import Quantity
from .._system import PairListOptions
from ._base import BaseModel


# Standard quantities for which gradients can be computed with autograd. These are the
# only standard quantities that can contain gradients.
AUTOGRAD_STANDARD_QUANTITIES = ("energy", "energy_ensemble", "energy_uncertainty")
AUTOGRAD_GRADIENTS = ("positions", "strain")


def supports_autograd(quantity: Quantity) -> bool:
    """Wether a given quantity can have gradients computed with autograd."""
    if quantity.sample_kind != "system":
        return False

    if "::" in quantity.name:
        # custom quantities can have any gradients
        return True

    return quantity.name.split("/")[0] in AUTOGRAD_STANDARD_QUANTITIES


def _wrap_method(implementation, name: str):
    """
    Wrap the ``implementation`` of method ``name`` to add the behavior defined in the
    corresponding ``_<name>_wrapper`` function of the class.
    """

    @functools.wraps(implementation)
    def wrapper(self, *args):
        if getattr(type(self), name) is not wrapper:
            # this wrapper was called through `super().<name>()` in a subclass, only
            # run the implementation and let the subclass wrapper add the behavior
            return implementation(self, *args)

        return getattr(AutogradModel, f"_{name}_wrapper")(self, implementation, *args)

    return wrapper


class AutogradModel(BaseModel):
    """
    Base class for models computing the gradients of per-system outputs with respect
    to ``"positions"`` and ``"strain"`` with automatic differentiation.

    Subclasses must implement ``_named_submodules()``, and ``execute_inner()`` using
    ``_split_gradients()``. Their own subclasses (the user models) implement
    ``capabilities()``, ``requested_pair_lists()`` and ``requested_inputs()`` as usual,
    and these methods are wrapped to add the behavior of this class.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for name in ("capabilities", "requested_pair_lists", "requested_inputs"):
            implementation = cls.__dict__.get(name)
            if implementation is None:
                continue

            setattr(cls, f"_raw_{name}", implementation)
            setattr(cls, name, _wrap_method(implementation, name))

    def _named_submodules(self):
        """
        Iterate over ``(name, module)`` for this model (with an empty name) and all
        its sub-modules, depth-first.
        """
        raise NotImplementedError

    def _raw_requested_pair_lists(
        self, requested_outputs: list[Quantity]
    ) -> list[PairListOptions]:
        """
        List the pair lists (neighbor lists) this model needs as input when computing
        the requested outputs. By default, no pair lists are needed.

        The pair lists requested by any sub-module of this model (any object with a
        ``requested_pair_lists()`` method) are automatically added to the ones returned
        by the implementation in the subclass.

        :param requested_outputs: outputs the model will compute. The gradients computed
            with autograd are removed from these requests.
        """
        return []

    requested_pair_lists = _wrap_method(
        _raw_requested_pair_lists, "requested_pair_lists"
    )

    def _raw_requested_inputs(
        self, requested_outputs: list[Quantity]
    ) -> list[Quantity]:
        """
        List the additional per-system inputs this model needs when computing the
        requested outputs. By default, no additional inputs are needed.

        The inputs requested by any sub-module of this model (any object with a
        ``requested_inputs()`` method) are automatically added to the ones returned by
        the implementation in the subclass.

        :param requested_outputs: outputs the model will compute. The gradients computed
            with autograd are removed from these requests.
        """
        return []

    requested_inputs = _wrap_method(_raw_requested_inputs, "requested_inputs")

    def _split_gradients(
        self, requested_outputs: list[Quantity]
    ) -> tuple[list[Quantity], list[list[str]]]:
        """
        Split the gradients in ``requested_outputs`` between the ones computed
        explicitly by the model and the ones computed with autograd.

        :return: the outputs to request from the model, containing only the explicit
            gradients; and for each output the list of gradients to compute with
            autograd.
        """
        if all(len(request.gradients) == 0 for request in requested_outputs):
            return list(requested_outputs), [[] for _ in requested_outputs]

        capabilities = self._raw_capabilities()

        inner_outputs = []
        autograd_gradients = []
        for request in requested_outputs:
            declared = capabilities.find_output(request)
            explicit = [] if declared is None else declared.gradients

            gradients = [g for g in request.gradients if g not in explicit]
            if len(gradients) != 0 and not supports_autograd(request):
                raise ValueError(
                    f"gradients of '{request.name}' with sample kind "
                    f"'{request.sample_kind}' can not be computed with autograd"
                )
            autograd_gradients.append(gradients)

            inner_outputs.append(
                Quantity(
                    name=request.name,
                    unit=request.unit,
                    sample_kind=request.sample_kind,
                    description=request.description,
                    gradients=[g for g in request.gradients if g in explicit],
                )
            )

        return inner_outputs, autograd_gradients

    def _capabilities_wrapper(self, implementation) -> ModelCapabilities:
        capabilities = implementation(self)
        if not isinstance(capabilities, ModelCapabilities):
            # let the caller report the error
            return capabilities

        # work on a copy, the implementation might return a cached object
        capabilities = ModelCapabilities.from_dict(capabilities.to_dict())

        outputs = []
        for output in capabilities.outputs:
            if supports_autograd(output):
                gradients = output.gradients
                gradients += [g for g in AUTOGRAD_GRADIENTS if g not in gradients]
                output = Quantity(
                    name=output.name,
                    unit=output.unit,
                    sample_kind=output.sample_kind,
                    description=output.description,
                    gradients=gradients,
                )
            outputs.append(output)

        capabilities.outputs = outputs
        return capabilities

    def _requested_pair_lists_wrapper(
        self, implementation, requested_outputs: list[Quantity]
    ) -> list[PairListOptions]:
        inner_outputs, _ = self._split_gradients(requested_outputs)

        requested: list[PairListOptions] = []
        for name, module in self._modules_with("requested_pair_lists"):
            if module is self:
                new_requests = implementation(self, inner_outputs)
            elif isinstance(module, AutogradModel):
                # only this module's requests, its sub-modules are visited separately
                new_requests = module._raw_requested_pair_lists(inner_outputs)
            else:
                new_requests = module.requested_pair_lists(inner_outputs)

            for new_options in new_requests:
                if not isinstance(new_options, PairListOptions):
                    raise TypeError(
                        f"requested_pair_lists() of {name} must return a list of "
                        f"PairListOptions, got an element of type {type(new_options)}"
                    )

                new_options = PairListOptions(
                    cutoff=new_options.cutoff,
                    full_list=new_options.full_list,
                    strict=new_options.strict,
                    requestors=new_options.requestors,
                )
                new_options.add_requestor(name)

                for existing in requested:
                    if existing == new_options:
                        for requestor in new_options.requestors:
                            existing.add_requestor(requestor)
                        break
                else:
                    requested.append(new_options)

        return requested

    def _requested_inputs_wrapper(
        self, implementation, requested_outputs: list[Quantity]
    ) -> list[Quantity]:
        inner_outputs, _ = self._split_gradients(requested_outputs)

        requested: dict[str, Quantity] = {}
        requestors: dict[str, str] = {}
        for name, module in self._modules_with("requested_inputs"):
            if module is self:
                new_inputs = implementation(self, inner_outputs)
            elif isinstance(module, AutogradModel):
                new_inputs = module._raw_requested_inputs(inner_outputs)
            else:
                new_inputs = module.requested_inputs(inner_outputs)

            for new_input in new_inputs:
                if not isinstance(new_input, Quantity):
                    raise TypeError(
                        f"requested_inputs() of {name} must return a list of "
                        f"Quantity, got an element of type {type(new_input)}"
                    )

                if new_input.name not in requested:
                    requested[new_input.name] = new_input
                    requestors[new_input.name] = name
                    continue

                existing = requested[new_input.name]
                for attribute in ["unit", "sample_kind"]:
                    new_value = getattr(new_input, attribute)
                    existing_value = getattr(existing, attribute)
                    if new_value != existing_value:
                        raise ValueError(
                            f"the input '{new_input.name}' is requested with "
                            f"different {attribute} by '{name}' "
                            f"({attribute}='{new_value}') and "
                            f"'{requestors[new_input.name]}' "
                            f"({attribute}='{existing_value}')"
                        )

        return list(requested.values())

    def _modules_with(self, method: str):
        root_name = self.__class__.__name__
        for name, module in self._named_submodules():
            if hasattr(module, method):
                yield (root_name if name == "" else f"{root_name}.{name}", module)
