from typing import Dict, List, Optional

import torch
from metatensor_torch import TensorBlock, TensorMap


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

        .. seealso::

            The :py:func:`metatomic_torch.systems_to_torch` function gives a way to
            convert from ASE, chemfiles and PySCF data to ``System``.

        :param types: 1D tensor of integer representing the particles identity. For
            atoms, this is typically their atomic numbers.

        :param positions: 2D tensor of shape (len(types), 3) containing the Cartesian
            positions of all particles in the system.

        :param cell: 2D tensor of shape (3, 3), describing the bounding box/unit cell of
            the system. Each row should be one of the bounding box vector; and columns
            should contain the x, y, and z components of these vectors (i.e. the cell
            should be given in row-major order). Systems that are not periodic along one
            or more directions should set the corresponding cell vectors to 0.

        :param pbc: tensor containing 3 boolean values, indicating which dimensions are
            periodic along each axis, in the same order as the cell vectors.
        """

        raise THIS_CODE_SHOULD_NOT_RUN

    def __len__(self) -> int:
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def types(self) -> torch.Tensor:
        """
        Tensor of 32-bit integers representing the particles identity

        .. deprecated:: 0.2.0

            Setting this property is deprecated and will be removed in a future
            version. Create a new :py:class:`System` instead.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def positions(self) -> torch.Tensor:
        """
        Tensor of floating point values containing the particles cartesian coordinates

        .. deprecated:: 0.2.0

            Setting this property is deprecated and will be removed in a future
            version. Create a new :py:class:`System` instead.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def cell(self) -> torch.Tensor:
        """
        Tensor of floating point values containing bounding box/cell of the system

        .. deprecated:: 0.2.0

            Setting this property is deprecated and will be removed in a future
            version. Create a new :py:class:`System` instead.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def pbc(self) -> torch.Tensor:
        """
        Tensor of boolean values indicating which dimensions are periodic

        .. deprecated:: 0.2.0

            Setting this property is deprecated and will be removed in a future
            version. Create a new :py:class:`System` instead.
        """
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
            <metatensor_torch.TensorBlock.dtype>` for more information.
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

    def add_pairs(
        self,
        options: "PairListOptions",
        pairs: TensorBlock,
    ):
        """
        Add a new pair list in this system corresponding to the given ``options``.

        The pair list should have the following samples: ``"first_atom"``,
        ``"second_atom"``, ``"cell_shift_a"``, ``"cell_shift_b"``, ``"cell_shift_c"``,
        containing the index of the first and second atoms (matching the "atom" sample
        in the positions); and the number of cell vector a/b/c to add to the positions
        difference to get the pair vector.

        The pairs should also have a single component ``"xyz"`` with values ``[0, 1,
        2]``; and a single property ``"distance"`` with value 0.

        The pairs values must contain the distance vector from the first to the
        second atom, i.e. ``positions[second_atom] - positions[first_atom] +
        cell_shift_a * cell_a + cell_shift_b * cell_b + cell_shift_c * cell_c``.

        :param options: options of the pair list
        :param pairs: list of pairs stored in a :py:class:`TensorBlock`
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def pairs(
        self,
        options: "PairListOptions",
    ) -> TensorBlock:
        """
        Retrieve a previously stored pair list with the given ``options``, or throw
        an error if no such pair list exists.

        :param options: options of the pair list to retrieve
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def known_pairs(self) -> List["PairListOptions"]:
        """
        Get the options of all the pair lists registered with this :py:class:`System`
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def add_custom_data(self, name: str, data: TensorMap):
        """
        Add custom data to this system, stored as :py:class:`TensorMap`.

        This is intended for experimentation with models that need more data as input,
        and moved into a field of ``System`` later.

        :param name: name of the custom data
        :param data: the data to store
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def custom_data(self, name: str) -> TensorMap:
        """
        Retrieve custom data stored in this System with the given ``name``, or throw
        an error if no data can be found.

        :param name: name of the custom data to retrieve
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def known_custom_data(self) -> List[str]:
        """
        Get the name of all the custom data registered with this :py:class:`System`
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def add_neighbor_list(
        self,
        options: "PairListOptions",
        neighbors: TensorBlock,
    ):
        """
        .. deprecated:: 0.2.0

            Use :py:meth:`System.add_pairs` instead.

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
        options: "PairListOptions",
    ) -> TensorBlock:
        """
        .. deprecated:: 0.2.0

            Use :py:meth:`System.pairs` instead.

        Retrieve a previously stored neighbors list with the given ``options``, or throw
        an error if no such neighbors list exists.

        :param options: options of the neighbors list to retrieve
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def known_neighbor_lists(self) -> List["PairListOptions"]:
        """
        .. deprecated:: 0.2.0

            Use :py:meth:`System.known_pairs` instead.

        Get all the neighbors lists options registered with this :py:class:`System`
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def add_data(self, name: str, tensor: TensorMap):
        """
        .. deprecated:: 0.2.0

            Use :py:meth:`System.add_custom_data` instead.

        Add custom data to this system, stored as :py:class:`TensorBlock`.

        This is intended for experimentation with models that need more data as input,
        and moved into a field of ``System`` later.

        :param name: name of the custom data
        :param tensor: the data to store
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def get_data(self, name: str) -> TensorMap:
        """
        .. deprecated:: 0.2.0

            Use :py:meth:`System.custom_data` instead.

        Retrieve custom data stored in this System with the given ``name``, or throw
        an error if no data can be found.

        :param name: name of the custom data to retrieve
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def known_data(self) -> List[str]:
        """
        .. deprecated:: 0.2.0

            Use :py:meth:`System.known_custom_data` instead.

        Get the name of all the custom data registered with this :py:class:`System`
        """
        raise THIS_CODE_SHOULD_NOT_RUN


class PairListOptions:
    """Options for the calculation of a pair list"""

    def __init__(
        self, cutoff: float, full_list: bool, strict: bool, requestor: str = ""
    ):
        """
        :param cutoff: spherical cutoff radius for the pair list, in the
            model units
        :param full_list: should the list be a full or half pair list
        :param strict: whether the list guarantee to have no pairs farther than cutoff
        :param requestor: who requested this pair list, you can add additional
            requestors later using :py:meth:`add_requestor`
        """
        pass

    @property
    def cutoff(self) -> float:
        """Spherical cutoff radius for this pair list in model units"""
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

    def __eq__(self, other: "PairListOptions") -> bool:
        raise THIS_CODE_SHOULD_NOT_RUN

    def __ne__(self, other: "PairListOptions") -> bool:
        raise THIS_CODE_SHOULD_NOT_RUN


class Quantity:
    """
    A physical quantity that a model can take as input or produce as output.

    This class implements the ``__str__`` and ``__repr__`` methods, so its
    representation can be easily printed, logged, inserted into other strings, etc.
    """

    def __init__(
        self,
        name: str,
        unit: str,
        sample_kind: str,
        gradients: Optional[List[str]] = None,
        description: Optional[str] = None,
    ):
        """
        :param name: name of the quantity, either one of the :ref:`standard names
            <standard-quantities>` or a custom name of the form
            ``<namespace>::<name>[/<variant>]``.
        :param unit: unit of the quantity
        :param sample_kind: kind of samples this quantity is associated with
        :param gradients: list of gradients stored explicitly in the
            :py:class:`TensorMap <metatensor.torch.TensorMap>` for this quantity
        :param description: optional description of this quantity, especially useful
            when a model defines multiple variants of the same quantity
        """
        pass

    @property
    def name(self) -> str:
        """
        Name of this quantity, this can be one of the :ref:`standard names
        <standard-quantities>` or a custom name of the form
        ``<namespace>::<name>[/<variant>]``.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def unit(self) -> str:
        """Unit of this quantity"""
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def sample_kind(self) -> str:
        """
        Kind of samples this quantity is associated with. This is can be one of the
        following:

        - ``system`` for per-system/global quantity
        - ``atom`` for per-atom quantity
        - ``atom_pair`` for quantities defined over a pair of atoms
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def gradients(self) -> List[str]:
        """
        List of gradients stored explicitly in the :py:class:`TensorMap
        <metatensor.torch.TensorMap>` for this quantity.

        Gradients can be one of the following:

        - ``positions`` for gradients with respect to positions, e.g. the forces
        - ``strain`` for gradients with respect to the strain, e.g. the stress tensor

        The returned list is a copy, assign to this property to change the gradients.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def description(self) -> Optional[str]:
        """
        Description of this quantity, used to provide more details about it, especially
        when a model defines multiple variants of the same quantity. Setting this to an
        empty string is the same as setting it to ``None``.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def __repr__(self) -> str:
        raise THIS_CODE_SHOULD_NOT_RUN

    def __str__(self) -> str:
        raise THIS_CODE_SHOULD_NOT_RUN

    def __eq__(self, other: "Quantity") -> bool:
        raise THIS_CODE_SHOULD_NOT_RUN

    def __ne__(self, other: "Quantity") -> bool:
        raise THIS_CODE_SHOULD_NOT_RUN


class ModelCapabilities:
    """
    Capabilities of a model: which outputs it can provide, which atomic types it
    supports, and other constraints on how it should be run.

    This class implements the ``__str__`` and ``__repr__`` methods, so its
    representation can be easily printed, logged, inserted into other strings, etc.
    """

    def __init__(
        self,
        atomic_types: List[int],
        interaction_range: float,
        length_unit: str,
        supported_devices: List[str],
        dtype: str,
        outputs: Optional[List[Quantity]] = None,
    ):
        """
        :param atomic_types: atomic types this model supports. The meaning of these
            integers is up to the model, and is not required to be the atomic numbers.
        :param interaction_range: interaction range of the model, in the length unit of
            the model
        :param length_unit: length unit of the model, e.g. ``"angstrom"``. All systems
            will be given to the model in this unit.
        :param supported_devices: devices on which the model can run, in order of
            preference. See :py:attr:`supported_devices` for the allowed values.
        :param dtype: data type of the model, used for all its inputs and outputs
        :param outputs: outputs this model can provide
        """
        pass

    @property
    def outputs(self) -> List[Quantity]:
        """
        Outputs this model can provide.

        During a specific run, a model might be asked to only compute a subset of these
        outputs.

        The returned list and the :py:class:`Quantity` it contains are copies, assign to
        this property to change the outputs.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def atomic_types(self) -> List[int]:
        """
        Atomic types this model supports.

        The meaning of the integers in this list is up to the model, and is not required
        to be the atomic numbers.

        The returned list is a copy, assign to this property to change the atomic types.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def interaction_range(self) -> float:
        """
        Interaction range of the model, in the length unit of the model.

        This is the maximum distance between two atoms for which the model's output can
        depend on their relative position. For a short range model, this is the same as
        the largest pair list cutoff; for a message passing model, this is the cutoff of
        one environment times the number of message passing steps; and for an explicit
        long range model, this should be set to infinity (``torch.inf``).
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def length_unit(self) -> str:
        """
        Length unit used by the model for its inputs, e.g. ``"angstrom"`` or
        ``"nanometer"``.

        This applies to the :py:attr:`interaction_range`, any cutoff in pair lists, the
        system positions, cell and pair lists given to the model.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def supported_devices(self) -> List[str]:
        """
        Devices on which this model can run, each entry is one of ``"cpu"``,
        ``"cuda"``, ``"rocm"`` or ``"metal"``.

        When setting this property, the torch names ``"hip"`` and ``"mps"`` are also
        accepted, and converted to ``"rocm"`` and ``"metal"`` respectively.

        The devices should be ordered by preference: the first entry in this list should
        be the best device for this model, and so on.

        The returned list is a copy, assign to this property to change the supported
        devices.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def dtype(self) -> str:
        """
        Data type of this model, used for all its inputs and outputs. This is either
        ``"float32"`` or ``"float64"``.

        The model is free to use a different data type for its internal computations.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    def __repr__(self) -> str:
        raise THIS_CODE_SHOULD_NOT_RUN

    def __str__(self) -> str:
        raise THIS_CODE_SHOULD_NOT_RUN

    def __eq__(self, other: "ModelCapabilities") -> bool:
        raise THIS_CODE_SHOULD_NOT_RUN

    def __ne__(self, other: "ModelCapabilities") -> bool:
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
        pass

    @property
    def name(self) -> str:
        """Name of this model"""
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def description(self) -> str:
        """Description of this model"""
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def authors(self) -> List[str]:
        """
        List of authors for this model.

        The returned list is a copy, assign to this property to change the authors.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def references(self) -> Dict[str, List[str]]:
        """
        Academic references for this model. The top level dict contains three keys:

        - "implementation": for reference to software used in the implementation
          of the model
        - "architecture": for reference that introduced the general architecture
          used by this model
        - "model": for reference specific to this exact model

        When setting this property, missing keys are treated as an empty list of
        references. The returned dictionary is a copy, assign to this property to
        change the references.
        """
        raise THIS_CODE_SHOULD_NOT_RUN

    @property
    def extra(self) -> Dict[str, str]:
        """
        Any additional metadata that is not contained in the other fields. There are
        no constraints on the keys or values of this dictionary. The extra metadata
        is intended to be used by models to store data they need.

        The returned dictionary is a copy, assign to this property to change the
        extra metadata.
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
        :py:meth:`System.add_pairs`
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

    The set of recognized base units is available :ref:`here
    <known-base-units>`.

    :param from_unit: current unit of the data (expression string)
    :param to_unit: target unit of the data (expression string)
    """
    raise THIS_CODE_SHOULD_NOT_RUN
