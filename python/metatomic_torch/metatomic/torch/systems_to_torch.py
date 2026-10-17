import warnings
from typing import List, Optional, Union

import numpy as np
import torch

from . import System


try:
    import ase

    HAS_ASE = True
except ImportError:
    HAS_ASE = False


try:
    import chemfiles

    HAS_CHEMFILES = True
except ImportError:
    HAS_CHEMFILES = False


try:
    import pyscf
    import pyscf.data.elements
    import pyscf.data.nist
    import pyscf.gto.mole

    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False

if HAS_PYSCF:
    _PYSCF_SYSTEM_TYPES = (pyscf.gto.mole.Mole,)
    _PYSCF_CELL_TYPE = None
    try:
        # `pyscf.pbc` is a separate subpackage, not loaded by `import pyscf`,
        # and `Cell` is not a subclass of `Mole`
        import pyscf.pbc.gto.cell

        _PYSCF_CELL_TYPE = pyscf.pbc.gto.cell.Cell
        _PYSCF_SYSTEM_TYPES = _PYSCF_SYSTEM_TYPES + (_PYSCF_CELL_TYPE,)
    except ImportError:
        pass


class IntoSystem:
    """
    A type that can be converted into a :py:class:`metatomic.torch.System`.

    This is an abstract class that is used to indicate a class whose objects can be
    converted into a :py:class:`System`. The supported types are
    :py:class:`ase.Atoms`, :py:class:`chemfiles.Frame`,
    :py:class:`pyscf.gto.mole.Mole` and :py:class:`pyscf.pbc.gto.cell.Cell`.
    """


def systems_to_torch(
    systems: Union[IntoSystem, List[IntoSystem]],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    positions_requires_grad: bool = False,
    cell_requires_grad: bool = False,
) -> Union[System, List[System]]:
    """Converts a system or a list of systems into a
    ``metatomic.torch.System`` or a list of such objects.

    :param: systems: The system or list of systems to convert.
    :param: dtype: The dtype of the output tensors. If ``None``, the default
        dtype is used.
    :param: device: The device of the output tensors. If ``None``, the default
        device is used.
    :param: positions_requires_grad: Whether the positions tensors of
        the outputs should require gradients.
    :param: cell_requires_grad: Whether the cell tensors of the outputs
        should require gradients.

    :return: The converted system or list of systems.
    """

    if isinstance(systems, list):
        return [
            _system_to_torch(
                system, dtype, device, positions_requires_grad, cell_requires_grad
            )
            for system in systems
        ]
    else:
        return _system_to_torch(
            systems, dtype, device, positions_requires_grad, cell_requires_grad
        )


def _system_to_torch(
    system: IntoSystem,
    dtype: Optional[torch.dtype],
    device: Optional[torch.device],
    positions_requires_grad: bool,
    cell_requires_grad: bool,
) -> System:
    """Converts a system into a ``metatomic.torch.System``.

    :param: system: The system to convert.
    :param: dtype: The dtype of the output tensors. If ``None``, the default
        dtype is used.
    :param: device: The device of the output tensors. If ``None``, the default
        device is used.
    :param: positions_requires_grad: Whether the positions tensors of
        the outputs should require gradients.
    :param: cell_requires_grad: Whether the cell tensors of the outputs
        should require gradients.

    :return: The converted system.
    """
    if HAS_ASE and isinstance(system, ase.Atoms):
        return _ase_to_torch(
            system, dtype, device, positions_requires_grad, cell_requires_grad
        )
    elif HAS_CHEMFILES and isinstance(system, chemfiles.Frame):
        return _chemfiles_to_torch(
            system, dtype, device, positions_requires_grad, cell_requires_grad
        )
    elif HAS_PYSCF and isinstance(system, _PYSCF_SYSTEM_TYPES):
        return _pyscf_to_torch(
            system, dtype, device, positions_requires_grad, cell_requires_grad
        )
    else:
        module = type(system).__module__
        if module.startswith("chemfiles") and not HAS_CHEMFILES:
            raise RuntimeError(
                "The `chemfiles` package is required to convert "
                "`chemfiles.Frame` objects to `System`s."
            )
        elif module.startswith("pyscf") and not HAS_PYSCF:
            raise RuntimeError(
                "The `pyscf` package is required to convert pyscf "
                "`Mole`/`Cell` objects to `System`s."
            )
        elif not HAS_ASE:
            raise RuntimeError(
                "The `ase` package is required to convert systems to torch."
            )
        else:
            raise ValueError(
                "Only `ase.Atoms`, `chemfiles.Frame`, `pyscf.gto.mole.Mole` and "
                f"`pyscf.pbc.gto.cell.Cell` objects can be converted to `System`s "
                f"for now; got {type(system)}."
            )


def _ase_to_torch(
    system: "ase.Atoms",
    dtype: Optional[torch.dtype],
    device: Optional[torch.device],
    positions_requires_grad: bool,
    cell_requires_grad: bool,
) -> System:
    if dtype is None:
        # this is necessary because creating torch tensors from numpy arrays
        # takes the dtype from the numpy array, which is not always the default
        # dtype
        dtype = torch.get_default_dtype()

    positions = torch.tensor(
        system.positions,
        requires_grad=positions_requires_grad,
        dtype=dtype,
        device=device,
    )

    cell_vectors_are_not_zero = np.any(system.cell != 0, axis=1)
    if not np.all(cell_vectors_are_not_zero == system.pbc):
        warnings.warn(
            "A conversion to `System` was requested for an `ase.Atoms` object "
            "with one or more non-zero cell vectors but where the corresponding "
            "boundary conditions are set to `False`. "
            "The corresponding cell vectors will be set to zero.",
            stacklevel=3,
        )

    cell = torch.zeros((3, 3), dtype=dtype, device=device)

    pbc = torch.tensor(system.pbc, dtype=torch.bool, device=device)

    cell[pbc] = torch.tensor(system.cell[system.pbc], dtype=dtype, device=device)

    if cell_requires_grad:
        cell.requires_grad_(True)

    types = torch.tensor(system.numbers, device=device, dtype=torch.int32)

    return System(positions=positions, cell=cell, types=types, pbc=pbc)


# global cache of type numbers for chemfiles atom types that are not elements of
# the periodic table. Ids start at 120, above all known atomic numbers, and are
# assigned in order of first encounter, matching the behavior of the equivalent
# converter in featomic.
_CHEMFILES_NON_ELEMENT_TYPES_CACHE = {}


def _chemfiles_type_for_non_element(name: str) -> int:
    if name not in _CHEMFILES_NON_ELEMENT_TYPES_CACHE:
        _CHEMFILES_NON_ELEMENT_TYPES_CACHE[name] = 120 + len(
            _CHEMFILES_NON_ELEMENT_TYPES_CACHE
        )
    return _CHEMFILES_NON_ELEMENT_TYPES_CACHE[name]


def _chemfiles_to_torch(
    system: "chemfiles.Frame",
    dtype: Optional[torch.dtype],
    device: Optional[torch.device],
    positions_requires_grad: bool,
    cell_requires_grad: bool,
) -> System:
    if dtype is None:
        dtype = torch.get_default_dtype()

    types = np.zeros(len(system.atoms), dtype=np.int32)
    for i, atom in enumerate(system.atoms):
        if atom.atomic_number != 0:
            types[i] = atom.atomic_number
        else:
            # atom types outside of the periodic table (e.g. "CH4" or "OW" in
            # coarse-grained force fields) get a synthetic id above 120
            types[i] = _chemfiles_type_for_non_element(atom.type)

    # chemfiles stores the cell in column-major format, so we transpose it to
    # get the cell vectors as rows
    cell = np.array(system.cell.matrix.T, dtype=np.float64)

    # chemfiles has no concept of boundary conditions: treat a frame with a
    # zero cell as non-periodic, and any axis with a non-zero cell vector as
    # periodic
    pbc = np.any(cell != 0, axis=1)

    return _make_system(
        types=types,
        positions=np.array(system.positions, dtype=np.float64),
        cell=cell,
        pbc=pbc,
        dtype=dtype,
        device=device,
        positions_requires_grad=positions_requires_grad,
        cell_requires_grad=cell_requires_grad,
    )


def _pyscf_charge_without_ghost(symbol: str) -> int:
    """Nuclear charge of the element underlying a pyscf atom symbol, stripping
    any ghost marker ("ghost-" or "X-" prefixes)."""
    bare = "".join(c for c in symbol.upper() if c.isalpha())
    if bare[:5] == "GHOST":
        bare = bare[5:]
    elif bare[0] == "X" and bare[:2] != "XE":
        bare = bare[1:]
    return int(pyscf.data.elements.charge(bare))


def _pyscf_to_torch(
    system: "pyscf.gto.mole.Mole",
    dtype: Optional[torch.dtype],
    device: Optional[torch.device],
    positions_requires_grad: bool,
    cell_requires_grad: bool,
) -> System:
    if dtype is None:
        dtype = torch.get_default_dtype()

    # ghost atoms (basis-set centers without a nucleus, used for counterpoise
    # corrections) have a nuclear charge of 0 in pyscf. They are given the
    # negative of the underlying element's atomic number, so they remain
    # distinguishable from real atoms while keeping the element identity.
    # This matches the documented behavior of the equivalent converter in
    # featomic.
    types = np.array(system.atom_charges(), dtype=np.int32)
    for i, atomic_type in enumerate(types):
        if atomic_type == 0:
            types[i] = -_pyscf_charge_without_ghost(system.atom_symbol(i))

    # atom_coords() is always in Bohr
    positions = np.array(system.atom_coords(), dtype=np.float64) * pyscf.data.nist.BOHR

    is_periodic = _PYSCF_CELL_TYPE is not None and isinstance(system, _PYSCF_CELL_TYPE)
    if is_periodic:
        cell = np.array(system.a, dtype=np.float64)
        if system.unit[0].lower() != "a":
            # the cell is stored in the unit given to pyscf
            cell = cell * pyscf.data.nist.BOHR
        # any axis with a non-zero cell vector is treated as periodic; this
        # also handles lower-dimensional cells, where the unused vectors are 0
        pbc = np.any(cell != 0, axis=1)
    else:
        cell = np.zeros((3, 3), dtype=np.float64)
        pbc = np.array([False, False, False])

    return _make_system(
        types=types,
        positions=positions,
        cell=cell,
        pbc=pbc,
        dtype=dtype,
        device=device,
        positions_requires_grad=positions_requires_grad,
        cell_requires_grad=cell_requires_grad,
    )


def _make_system(
    types: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    dtype: torch.dtype,
    device: Optional[torch.device],
    positions_requires_grad: bool,
    cell_requires_grad: bool,
) -> System:
    positions_t = torch.tensor(
        positions,
        requires_grad=positions_requires_grad,
        dtype=dtype,
        device=device,
    )

    cell_t = torch.tensor(
        cell,
        requires_grad=cell_requires_grad,
        dtype=dtype,
        device=device,
    )

    pbc_t = torch.tensor(pbc, dtype=torch.bool, device=device)

    types_t = torch.tensor(types, dtype=torch.int32, device=device)

    return System(positions=positions_t, cell=cell_t, types=types_t, pbc=pbc_t)
