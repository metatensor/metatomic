from typing import Dict, List, Literal, Optional

from metatensor_torch import Labels

from .. import ModelMetadata


THIS_CODE_SHOULD_NOT_RUN = RuntimeError(
    "This is the documentation-only version of this class, it should not be used at "
    "runtime. If you see this error while running examples, check above for the actual "
    "error message."
)


class ModelOutput:
    """Information about one of the quantity a model can compute."""

    def __init__(
        self,
        quantity: str = "",
        unit: str = "",
        sample_kind: Literal["system", "atom", "atom_pair"] = "system",
        explicit_gradients: List[str] = [],  # noqa B006
        description: str = "",
    ):
        pass

    @property
    def quantity(self) -> str:
        """
        Quantity of the output (e.g. energy, dipole, …).  If this is an empty string, no
        unit conversion will be performed.

        .. deprecated:: 0.1.12

            The ``quantity`` field is deprecated and will be removed.
            Unit conversion determines dimensions from the unit expression.
            Set ``quantity`` to an empty string to suppress deprecation warnings.

        The list of possible quantities is available :ref:`here
        <known-base-units>`.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def unit(self) -> str:
        """
        Unit of the output. If this is an empty string, no unit conversion will be
        performed.

        The list of possible units is available :ref:`here <known-base-units>`.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def per_atom(self) -> bool:
        """Whether this output is a per-atom quantity or a global quantity.

        This is deprecated, and only exists for backward compatibility. If
        ``sample_kind`` is not one of ``"system"`` or ``"atom"``, trying to get this
        property will raise an error.
        """

    @property
    def sample_kind(self) -> Literal["system", "atom", "atom_pair"]:
        """Kind of sample for this output, e.g.:

        - "system" for global properties of the system (e.g. energy, dipole, ...)
        - "atom" for per-atom properties (e.g. atomic energy, atomic charge, ...)
        - "atom_pair" for properties of pairs of atoms (e.g. bond order, ...)
        """

    explicit_gradients: List[str]
    """
    Which gradients should be computed eagerly and stored inside the output
    :py:class:`TensorMap`.
    """

    @property
    def description(self) -> str:
        """
        A description of this output. Especially recommended for non-standard outputs
        and variants of the one unit.
        """
        raise THIS_CODE_SHOULD_NOT_RUN


class LegacyModelCapabilities:
    """Description of a model capabilities, i.e. everything a model can do."""

    def __init__(
        self,
        outputs: Dict[str, ModelOutput] = {},  # noqa B006
        atomic_types: List[int] = [],  # noqa B006
        interaction_range: float = -1,
        length_unit: str = "",
        supported_devices: List[str] = [],  # noqa B006
        dtype: str = "",
    ):
        pass

    @property
    def outputs(self) -> Dict[str, ModelOutput]:
        """
        All possible outputs from this model and corresponding settings.

        During a specific run, a model might be asked to only compute a subset of these
        outputs. Some outputs are standardized, and have additional constrains on how
        the associated metadata should look like, documented in the
        :ref:`standard-quantities` section.

        If you want to define a new output for your own usage, its name should look like
        ``"<domain>::<output>"``, where ``<domain>`` indicates who defines this new
        output and ``<output>`` describes the output itself. For example,
        ``"my-package::foobar"`` for a ``foobar`` output defined in ``my-package``.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    atomic_types: List[int]
    """which atomic types the model can handle"""

    interaction_range: float
    """
    How far a given atom needs to know about other atoms, in the length unit of the
    model.

    For a short range model, this is the same as the largest neighbors list cutoff. For
    a message passing model, this is the cutoff of one environment times the number of
    message passing steps. For an explicit long range model, this should be set to
    infinity (``float("inf")``/``math.inf``/``torch.inf`` in Python).
    """

    @property
    def length_unit() -> str:
        """
        Unit used by the model for its inputs.

        This applies to the ``interaction_range``, any cutoff in neighbors lists, the
        atoms positions and the system cell.

        The list of possible units is available :ref:`here <known-base-units>`.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def dtype() -> str:
        """
        The dtype of this model

        This can be ``"float32"`` or ``"float64"``, and must be used by the engine as
        the dtype of all inputs and outputs for this model.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def engine_interaction_range(self, engine_length_unit: str) -> float:
        """
        Same as :py:attr:`interaction_range`, but in the unit of length used by the
        engine.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    supported_devices: List[str]
    """
    What devices can this model run on? This should only contain the ``device_type``
    part of the device, and not the device number (i.e. this should be ``"cuda"``, not
    ``"cuda:0"``).

    Devices should be ordered in order of preference: the first entry in this list
    should be the best device for this model, and so on.
    """


class ModelEvaluationOptions:
    """
    Options requested by the simulation engine/evaluation code when doing a single model
    evaluation.
    """

    def __init__(
        self,
        length_unit: str = "",
        outputs: Dict[str, ModelOutput] = {},  # noqa B006
        selected_atoms: Optional[Labels] = None,
    ):
        pass

    @property
    def length_unit(self) -> str:
        """
        Unit of lengths the engine uses for the model input.

        The list of possible units is available :ref:`here <known-base-units>`.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    outputs: Dict[str, ModelOutput]
    """requested outputs for this run and corresponding settings"""

    @property
    def selected_atoms() -> Optional[Labels]:
        """
        Only run the calculation for a selected subset of atoms.

        If this is set to ``None``, run the calculation on all atoms. If this is a set
        of :py:class:`metatensor_torch.Labels`, it will have two dimensions named
        ``"system"`` and ``"atom"``, containing the 0-based indices of all the atoms in
        the selected subset.
        """
        raise THIS_CODE_SHOULD_NOT_RUN


def read_model_metadata(path: str) -> ModelMetadata:
    """
    Read metadata of a saved atomistic model.

    This function allows to access the metadata of a model without loading it
    in advance.

    :param path: path to the exported model file
    """
    raise THIS_CODE_SHOULD_NOT_RUN


def check_atomistic_model(path: str):
    """
    Check that the file at ``path`` contains an exported atomistic model, and that this
    model can be loaded in the current process.

    This function should be called before :py:func:`torch.jit.load()` when loading an
    existing model.

    :param path: path to the exported model file
    """
    raise THIS_CODE_SHOULD_NOT_RUN


def load_model_extensions(path: str, extensions_directory: Optional[str] = None):
    """
    Load the TorchScript extensions (and their dependencies) that the model at ``path``
    uses.

    If ``extensions_directory`` is provided, we look for the extensions and their
    dependencies in there first. If this function fails to load some library, it will
    produce a warning using Torch's warnings infrastructure. Users can set the
    ``METATENSOR_DEBUG_EXTENSIONS_LOADING`` environment variable to get more
    informations about a failure in the standard error output.

    :param path: path to the exported model file
    :param extensions_directory: path to a directory containing the extensions. This
        directory will typically be created by calling
        :py:meth:`AtomisticModel.export` with
        ``collect_extensions=extensions_directory``.
    """
    raise THIS_CODE_SHOULD_NOT_RUN


def unit_dimension_for_quantity(name: str) -> str:
    """
    Get the physical dimension of the standard quantity (input or outptu) with the given
    ``name``.

    This function will return one of the following strings:

    - an empty string for non-standard outputs
    - "none" for outputs that should be dimensionless (features, …).
    - "length" for length-like quantities (positions, …);
    - "momentum" for momentum-like quantities (momenta, …);
    - "velocity" for velocity-like quantities (velocities, …);
    - "mass" for mass-like quantities (masses, …);
    - "energy" for energy-like quantities (energy, energy_ensemble, …);
    - "force" for force-like quantities (non_conservative_forces, …);
    - "pressure" for pressure-like quantities (non_conservative_stress, …);
    - "charge" for charge-like quantities (charges, …);
    - "heat_flux" for heat flux-like quantities (heat_flux, …);

    :param name: name of the output/input
    :return: physical dimension of the output
    """
    raise THIS_CODE_SHOULD_NOT_RUN


def pick_device(model_devices: List[str], desired_device: Optional[str]) -> str:
    """
    Select the best device according to the list of ``model_devices`` from a model, the
    user-provided ``desired_device`` and what's available on the current machine.

    If ``desired_device`` is provided, it is checked against the ``model_devices`` and
    the machine availability. If it contains a device index (e.g. ``"cuda:1"``), the
    base device type (``"cuda"``) is used for these checks, and the full string is
    returned if successful.

    If ``desired_device`` is ``None`` or an empty string, the first available device
    from ``model_devices`` will be picked and returned.

    :param model_devices: list of devices supported by a model in order of preference
    :param desired_device: user-provided desired device.
    """
    raise THIS_CODE_SHOULD_NOT_RUN


def pick_output(
    requested_output: str,
    outputs: Dict[str, ModelOutput],
    desired_variant: Optional[str] = None,
) -> str:
    """
    Pick the output for the given ``requested_output`` from the availabilities of the
    model's ``outputs``, according to the optional ``desired_variant``.

    :param requested_output: name of the output to pick a variant for
    :param outputs: all available outputs from the model
    :param desired_variant: if provided, try to pick this specific variant
    """
    raise THIS_CODE_SHOULD_NOT_RUN
