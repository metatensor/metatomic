from typing import Dict, List, Literal, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


THIS_CODE_SHOULD_NOT_RUN = RuntimeError(
    "This is the documentation-only version of this class, it should not be used at "
    "runtime. If you see this error while running examples, check above for the actual "
    "error message."
)


class System:
    """
    A System contains all the information about an atomistic system; and should be used
    as the input of atomistic models.
    """

    def __init__(
        self,
        types: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
    ):
        """
        You can create a :py:class:`System` with ``types``, ``positions`` and ``cell``
        tensors, or convert data from other libraries.

        .. admonition:: Converting data to metatensor `System`

            We provide a way to convert :py:class:`ase.Atoms` instances to
            :py:class:`System` using the :py:func:`systems_to_torch()` function.

            In addition, some external packages provide ways to create
            :py:class:`System` using data from other libraries:

            - `featomic`_ has the :py:func:`featomic.torch.systems_to_torch()`
              function that can convert from ASE, chemfiles and PySCF.

            .. _featomic: https://metatensor.github.io/featomic/latest/index.html

        :param types: 1D tensor of integer representing the particles identity. For
            atoms, this is typically their atomic numbers.

        :param positions: 2D tensor of shape (len(types), 3) containing the Cartesian
            positions of all particles in the system.

        :param cell: 2D tensor of shape (3, 3), describing the bounding box/unit cell of
            the system. Each row should be one of the bounding box vector; and columns
            should contain the x, y, and z components of these vectors (i.e. the cell
            should be given in row-major order). Systems that are not periodic along
            one or more directions should set the corresponding cell vectors to 0.

        :param pbc: tensor containing 3 boolean values, indicating which dimensions are
            periodic along each axis, in the same order as the cell vectors.
        """

        raise THIS_CODE_SHOULD_NOT_RUN

    def __len__(self) -> int:
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def types(self) -> torch.Tensor:
        """Tensor of 32-bit integers representing the particles identity"""
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def positions(self) -> torch.Tensor:
        """
        Tensor of floating point values containing the particles cartesian coordinates
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def cell(self) -> torch.Tensor:
        """Tensor of floating point values containing bounding box/cell of the system"""
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def pbc(self) -> torch.Tensor:
        """Tensor of boolean values indicating which dimensions are periodic"""
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def device(self) -> torch.device:
        """get the device of all the arrays stored inside this :py:class:`System`"""
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def dtype(self) -> torch.dtype:
        """
        get the dtype of all the arrays stored inside this :py:class:`System`

        .. warning::

            Due to limitations in TorchScript C++ extensions, the dtype is returned as
            an integer, which can not be compared with :py:class:`torch.dtype`
            instances. See :py:attr:`TensorBlock.dtype
            <metatensor.torch.TensorBlock.dtype>` for more information.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def to(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        non_blocking: bool = False,
    ) -> "System":
        """
        Move all the arrays in this system to the given ``dtype`` and ``device``.

        :param dtype: new dtype to use for all arrays. The dtype stays the same if this
            is set to ``None``.
        :param device: new device to use for all arrays. The device stays the same if
            this is set to ``None``.
        :param bool non_blocking: If this is ``True``, the function tries to move the
            data asynchronously. See :py:meth:`torch.Tensor.to` for more information.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def add_neighbor_list(
        self,
        options: "NeighborListOptions",
        neighbors: TensorBlock,
    ):
        """
        Add a new neighbors list in this system corresponding to the given ``options``.

        The neighbors list should have the following samples: ``"first_atom"``,
        ``"second_atom"``, ``"cell_shift_a"``, ``"cell_shift_b"``, ``"cell_shift_c"``,
        containing the index of the first and second atoms (matching the "atom" sample
        in the positions); and the number of cell vector a/b/c to add to the positions
        difference to get the pair vector.

        The neighbors should also have a single component ``"xyz"`` with values ``[0, 1,
        2]``; and a single property ``"distance"`` with value 0.

        The neighbors values must contain the distance vector from the first to the
        second atom, i.e. ``positions[second_atom] - positions[first_atom] +
        cell_shift_a * cell_a + cell_shift_b * cell_b + cell_shift_c * cell_c``.

        :param options: options of the neighbors list
        :param neighbors: list of neighbors stored in a :py:class:`TensorBlock`
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def get_neighbor_list(
        self,
        options: "NeighborListOptions",
    ) -> TensorBlock:
        """
        Retrieve a previously stored neighbors list with the given ``options``, or throw
        an error if no such neighbors list exists.

        :param options: options of the neighbors list to retrieve
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def known_neighbor_lists(self) -> List["NeighborListOptions"]:
        """
        Get all the neighbors lists options registered with this :py:class:`System`
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def add_data(self, name: str, tensor: TensorMap, override: bool = False):
        """
        Add custom data to this system, stored as :py:class:`TensorBlock`.

        This is intended for experimentation with models that need more data as input,
        and moved into a field of ``System`` later.

        :param name: name of the custom data
        :param tensor: the data to store
        :param override: if ``True``, allow this function to override existing data with
            the same name
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def get_data(self, name: str) -> TensorMap:
        """
        Retrieve custom data stored in this System with the given ``name``, or throw
        an error if no data can be found.

        :param name: name of the custom data to retrieve
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def known_data(self) -> List[str]:
        """
        Get the name of all the custom data registered with this :py:class:`System`
        """
        raise THIS_CODE_SHOULD_NOT_RUN


class NeighborListOptions:
    """Options for the calculation of a neighbors list"""

    def __init__(
        self, cutoff: float, full_list: bool, strict: bool, requestor: str = ""
    ):
        """
        :param cutoff: spherical cutoff radius for the neighbors list, in the
            model units
        :param full_list: should the list be a full or half neighbors list
        :param strict: whether the list guarantee to have no pairs farther than cutoff
        :param requestor: who requested this neighbors list, you can add additional
            requestors later using :py:meth:`add_requestor`
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def cutoff(self) -> float:
        """Spherical cutoff radius for this neighbors list in model units"""
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def length_unit(self) -> str:
        """
        The unit of length used for the cutoff.

        This is typically set by :py:class:`AtomisticModel` when collecting
        all neighbors list requests.

        The list of possible units is available :ref:`here <known-base-units>`.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def engine_cutoff(self, engine_length_unit: str) -> float:
        """
        Spherical cutoff radius for this neighbors list in engine units.

        The engine must provide the unit it uses for lengths, and the cutoff will
        automatically be converted.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def full_list(self) -> bool:
        """
        Should the list be a full neighbors list (contains both the pair ``i->j`` and
        ``j->i``) or a half neighbors list (contains only the pair ``i->j``)
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def strict(self) -> bool:
        """
        Does the list guarantee to have no pairs beyond the cutoff (strict) or
        can it also have pairs that are farther apart (non strict)
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def requestors(self) -> List[str]:
        """Get the list of modules requesting this neighbors list"""
        raise THIS_CODE_SHOULD_NOT_RUN

    def add_requestor(self, requestor: str):
        """
        Add another ``requestor`` to the list of modules requesting this neighbors list
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def __repr__(self) -> str:
        raise THIS_CODE_SHOULD_NOT_RUN

    def __str__(self) -> str:
        raise THIS_CODE_SHOULD_NOT_RUN

    def __eq__(self, other: "NeighborListOptions") -> bool:
        raise THIS_CODE_SHOULD_NOT_RUN

    def __ne__(self, other: "NeighborListOptions") -> bool:
        raise THIS_CODE_SHOULD_NOT_RUN


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
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def quantity(self) -> str:
        """
        Quantity of the output (e.g. energy, dipole, …).  If this is an empty string, no
        unit conversion will be performed.

        .. deprecated::
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


class ModelCapabilities:
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
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def outputs(self) -> Dict[str, ModelOutput]:
        """
        All possible outputs from this model and corresponding settings.

        During a specific run, a model might be asked to only compute a subset of these
        outputs. Some outputs are standardized, and have additional constrains on how
        the associated metadata should look like, documented in the
        :ref:`atomistic-models-outputs` section.

        If you want to define a new output for your own usage, it name should looks like
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
        raise THIS_CODE_SHOULD_NOT_RUN

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
        of :py:class:`metatensor.torch.Labels`, it will have two dimensions named
        ``"system"`` and ``"atom"``, containing the 0-based indices of all the atoms in
        the selected subset.
        """
        raise THIS_CODE_SHOULD_NOT_RUN


class ModelMetadata:
    """
    Metadata about a specific exported model

    This class implements the ``__str__`` and ``__repr__`` methods, so its
    representation can be easily printed, logged, inserted into other strings, etc.
    """

    def __init__(
        self,
        name: str = "",
        description: str = "",
        authors: List[str] = [],  # noqa: B006
        references: Dict[str, List[str]] = {},  # noqa: B006
        extra: Dict[str, str] = {},  # noqa: B006
    ):
        raise THIS_CODE_SHOULD_NOT_RUN

    name: str
    """Name of this model"""

    description: str
    """Description of this model"""

    authors: List[str]
    """List of authors for this model"""

    references: Dict[str, List[str]]
    """
    Academic references for this model. The top level dict can have three keys:

    - "implementation": for reference to software used in the implementation
      of the model
    - "architecture": for reference that introduced the general architecture
      used by this model
    - "model": for reference specific to this exact model
    """

    extra: Dict[str, str]
    """
    Any additional metadata that is not contained in the other fields. There are
    no constraints on the keys or values of this dictionary. The extra metadata
    is intended to be used by models to store data they need.
    """


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


def register_autograd_neighbors(
    system: System, neighbors: TensorBlock, check_consistency: bool
):
    """
    Register a new torch autograd node going from (``system.positions``,
    ``system.cell``) to the ``neighbors`` distance vectors.

    This does not recompute the distance vectors, but work as-if all the data in
    ``neighbors.values`` was computed directly from ``system.positions`` and
    ``system.cell``, allowing downstream models to use it directly with full autograd
    integration.

    :param system: system containing the positions and cell used to compute the
        neighbors list
    :param neighbors: neighbors list, following the same format as
        :py:meth:`System.add_neighbor_list`
    :param check_consistency: can be set to ``True`` to run additional checks in case
        the data in neighbors does not follow what's expected.
    """
    raise THIS_CODE_SHOULD_NOT_RUN


def unit_conversion_factor(from_unit: str, to_unit: str) -> float:
    """
    Get the multiplicative conversion factor from ``from_unit`` to
    ``to_unit``.

    Both ``from_unit`` and ``to_unit`` are parsed as unit expressions
    supporting compound forms like ``"kJ/mol/A^2"`` or
    ``"(eV*u)^(1/2)"``. The parser validates that both expressions have
    matching physical dimensions.

    This function is TorchScript-compatible. The deprecated 3-argument form
    ``unit_conversion_factor(quantity, from_unit, to_unit)`` is available
    via ``torch.ops.metatomic.unit_conversion_factor``.

    The set of recognized base units is available :ref:`here
    <known-base-units>`.

    .. rubric:: Migration from 3-argument form

    The 3-argument form ``unit_conversion_factor(quantity, from_unit, to_unit)``
    is deprecated. The ``quantity`` parameter is no longer needed because
    dimensional compatibility is checked automatically by the parser.

    **Before (deprecated):**

    .. code-block:: python

        factor = unit_conversion_factor("energy", "eV", "meV")
        factor = unit_conversion_factor("force", "eV/A", "Hartree/Bohr")

    **After (recommended):**

    .. code-block:: python

        factor = unit_conversion_factor("eV", "meV")
        factor = unit_conversion_factor("eV/A", "Hartree/Bohr")

    The new 2-argument form also supports compound expressions that were not
    possible with the old API:

    .. code-block:: python

        # Momentum conversion (fractional powers)
        factor = unit_conversion_factor("(eV*u)^(1/2)", "u*A/fs")
        # Complex compound expression
        factor = unit_conversion_factor("kJ/mol/A^2", "Hartree/Bohr^3")

    :param from_unit: current unit of the data (expression string)
    :param to_unit: target unit of the data (expression string)
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


def pick_device(
    model_devices: List[str], desired_device: Optional[str]
) -> torch.device:
    """
    Select the best device according to the list of ``model_devices`` from a model, the
    user-provided ``desired_device`` and what's available on the current machine.

    If ``desired_device`` is provided, it is checked against the ``model_devices`` and
    the machine availability. If it contains a device index (e.g. ``"cuda:1"``), the
    base device type (``"cuda"``) is used for these checks, and the full
    :py:class:`torch.device` (including the provided index) is returned if successful.

    If ``desired_device`` is ``None`` or an empty string, the first available device
    from ``model_devices`` will be picked and returned without a specific index.

    :param model_devices: list of devices supported by a model in order of preference
    :param desired_device: user-provided desired device string (e.g. ``"cuda"``,
        ``"cuda:1"``, ``"cpu"``), or ``None`` to auto-select.
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
