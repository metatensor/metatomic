try:
    import ase

    HAS_ASE = True
except ImportError:
    HAS_ASE = False

import numpy as np
import pytest
import torch

from metatomic.torch import systems_to_torch


@pytest.mark.skipif(not HAS_ASE, reason="requires ASE")
def test_ase_to_torch_periodic():
    atoms = ase.Atoms(
        "CO",
        positions=[(0, 0, 0), (0, 0, 2)],
        cell=4 * np.eye(3),
        pbc=[True, True, True],
    )

    system = systems_to_torch(atoms)

    assert isinstance(system, torch.ScriptObject)

    assert isinstance(system.types, torch.Tensor)
    assert torch.all(system.types == torch.tensor([6, 8]))
    assert system.types.dtype == torch.int32
    assert not system.types.requires_grad

    assert isinstance(system.positions, torch.Tensor)
    assert torch.all(system.positions == torch.tensor([(0, 0, 0), (0, 0, 2)]))
    assert system.positions.dtype == torch.get_default_dtype()
    assert not system.positions.requires_grad

    assert isinstance(system.cell, torch.Tensor)
    assert torch.all(system.cell == 4 * torch.eye(3))
    assert system.cell.dtype == torch.get_default_dtype()
    assert not system.cell.requires_grad

    system = systems_to_torch(atoms, positions_requires_grad=True)

    assert system.positions.requires_grad
    assert not system.cell.requires_grad

    # test a list of ase.Atoms
    systems = systems_to_torch([atoms, atoms])
    assert isinstance(systems[0], torch.ScriptObject)


@pytest.mark.skipif(not HAS_ASE, reason="requires ASE")
def test_ase_to_torch_non_periodic():
    atoms = ase.Atoms(
        "CO",
        positions=[(0, 0, 0), (0, 0, 2)],
        cell=None,
        pbc=[False, False, False],
    )

    system = systems_to_torch(atoms)

    assert isinstance(system, torch.ScriptObject)

    assert isinstance(system.types, torch.Tensor)
    assert torch.all(system.types == torch.tensor([6, 8]))
    assert system.types.dtype == torch.int32
    assert not system.types.requires_grad

    assert isinstance(system.positions, torch.Tensor)
    assert torch.all(system.positions == torch.tensor([(0, 0, 0), (0, 0, 2)]))
    assert system.positions.dtype == torch.get_default_dtype()
    assert not system.positions.requires_grad

    assert torch.all(system.cell == 0)

    # test a list of ase.Atoms
    systems = systems_to_torch([atoms, atoms])
    assert isinstance(systems[0], torch.ScriptObject)

    # test the warning when converting a non-periodic
    # system with non-zero cell vectors
    atoms = ase.Atoms(
        "CO",
        positions=[(0, 0, 0), (0, 0, 2)],
        cell=4 * np.eye(3),
        pbc=[False, False, False],
    )

    message = (
        "A conversion to `System` was requested for an `ase.Atoms` object with one "
        "or more non-zero cell vectors but where the corresponding boundary conditions "
        "are set to `False`. The corresponding cell vectors will be set to zero."
    )
    with pytest.warns(UserWarning, match=message):
        system = systems_to_torch(atoms)

    assert torch.all(system.cell == 0)

    atoms = ase.Atoms(
        "CO",
        positions=[(0, 0, 0), (0, 0, 2)],
        cell=4 * np.eye(3),
        pbc=[False, False, True],
    )
    with pytest.warns(UserWarning, match=message):
        system = systems_to_torch(atoms)

    assert torch.all(system.cell[0] == 0)
    assert torch.all(system.cell[1] == 0)


try:
    import chemfiles

    HAS_CHEMFILES = True
except ImportError:
    HAS_CHEMFILES = False

try:
    import pyscf
    import pyscf.data.nist
    import pyscf.gto

    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False


@pytest.mark.skipif(not HAS_CHEMFILES, reason="requires chemfiles")
def test_chemfiles_to_torch_periodic():
    frame = chemfiles.Frame()
    frame.add_atom(chemfiles.Atom("O"), (0, 0, 0))
    frame.add_atom(chemfiles.Atom("H"), (0, 0, 1))
    frame.add_atom(chemfiles.Atom("H"), (0, 1, 0))
    frame.cell = chemfiles.UnitCell((10.0, 11.0, 12.0), (80.0, 85.0, 95.0))

    system = systems_to_torch(frame)

    assert isinstance(system, torch.ScriptObject)

    assert system.types.dtype == torch.int32
    assert torch.all(system.types == torch.tensor([8, 1, 1]))

    assert system.positions.dtype == torch.get_default_dtype()
    assert torch.all(
        system.positions == torch.tensor([(0, 0, 0), (0, 0, 1), (0, 1, 0)])
    )
    assert not system.positions.requires_grad

    # chemfiles stores the cell in column-major format, the transpose gives the
    # cell vectors as rows
    expected_cell = torch.tensor(np.array(frame.cell.matrix).T)
    assert torch.allclose(system.cell, expected_cell.to(torch.get_default_dtype()))
    # the row norms of the transposed cell are the cell lengths
    assert torch.allclose(
        torch.linalg.norm(system.cell, dim=1),
        torch.tensor([10.0, 11.0, 12.0], dtype=torch.get_default_dtype()),
    )
    assert torch.all(system.pbc == torch.tensor([True, True, True]))

    system = systems_to_torch(
        frame, positions_requires_grad=True, cell_requires_grad=True
    )
    assert system.positions.requires_grad
    assert system.cell.requires_grad

    # test a list of frames
    systems = systems_to_torch([frame, frame])
    assert isinstance(systems[0], torch.ScriptObject)


@pytest.mark.skipif(not HAS_CHEMFILES, reason="requires chemfiles")
def test_chemfiles_to_torch_non_periodic():
    frame = chemfiles.Frame()
    frame.add_atom(chemfiles.Atom("C"), (1, 2, 3))

    system = systems_to_torch(frame)

    assert torch.all(system.cell == 0)
    assert torch.all(system.pbc == torch.tensor([False, False, False]))
    assert torch.all(system.types == torch.tensor([6]))
    assert torch.all(system.positions == torch.tensor([(1, 2, 3)]))


@pytest.mark.skipif(not HAS_CHEMFILES, reason="requires chemfiles")
def test_chemfiles_to_torch_non_element_types():
    from metatomic.torch.systems_to_torch import _CHEMFILES_NON_ELEMENT_TYPES_CACHE

    _CHEMFILES_NON_ELEMENT_TYPES_CACHE.clear()

    frame = chemfiles.Frame()
    frame.add_atom(chemfiles.Atom("CH4"), (0, 0, 0))
    frame.add_atom(chemfiles.Atom("OW"), (1, 0, 0))
    frame.add_atom(chemfiles.Atom("C"), (2, 0, 0))
    frame.add_atom(chemfiles.Atom("CH4"), (3, 0, 0))

    system = systems_to_torch(frame)

    # non-element types get synthetic ids starting at 120, stable within a
    # process for the same type name
    assert torch.all(system.types == torch.tensor([120, 121, 6, 120]))
    assert system.types.dtype == torch.int32

    _CHEMFILES_NON_ELEMENT_TYPES_CACHE.clear()


@pytest.mark.skipif(not HAS_PYSCF, reason="requires pyscf")
def test_pyscf_molecule_to_torch():
    mol = pyscf.gto.M(
        atom="O 0 0 0; H 0 0 1; H 0 1 0",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )

    system = systems_to_torch(mol)

    assert isinstance(system, torch.ScriptObject)
    assert system.types.dtype == torch.int32
    assert torch.all(system.types == torch.tensor([8, 1, 1]))

    # pyscf reports positions in Bohr, the conversion gives Angstrom
    assert torch.allclose(
        system.positions,
        torch.tensor(
            [(0, 0, 0), (0, 0, 1), (0, 1, 0)], dtype=torch.get_default_dtype()
        ),
    )

    assert torch.all(system.cell == 0)
    assert torch.all(system.pbc == torch.tensor([False, False, False]))

    system = systems_to_torch(mol, dtype=torch.float64)
    assert system.positions.dtype == torch.float64

    system = systems_to_torch(mol, positions_requires_grad=True)
    assert system.positions.requires_grad


@pytest.mark.skipif(not HAS_PYSCF, reason="requires pyscf")
def test_pyscf_ghost_atoms_to_torch():
    mol = pyscf.gto.M(
        atom="O 0 0 0; ghost-H 0 0 1",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )

    system = systems_to_torch(mol)

    # ghost atoms get the negative of the underlying element's atomic number
    assert torch.all(system.types == torch.tensor([8, -1]))

    # the "X-" prefix form of ghost atoms is handled the same way
    mol = pyscf.gto.M(
        atom="He 0 0 0; X-He 0 0 1.5",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    system = systems_to_torch(mol)
    assert torch.all(system.types == torch.tensor([2, -2]))
    assert torch.allclose(
        system.positions,
        torch.tensor([(0, 0, 0), (0, 0, 1.5)], dtype=torch.get_default_dtype()),
    )


@pytest.mark.skipif(not HAS_PYSCF, reason="requires pyscf")
def test_pyscf_periodic_cell_to_torch():
    from pyscf.pbc import gto as pbc_gto

    cell = pbc_gto.Cell(verbose=0)
    cell.atom = "He 0 0 0; ghost-He 0 0 1.5"
    cell.a = np.eye(3) * 4.0
    cell.basis = "sto-3g"
    cell.unit = "Angstrom"
    cell.build()

    system = systems_to_torch(cell)

    assert torch.all(system.types == torch.tensor([2, -2]))
    assert torch.allclose(
        system.cell, 4.0 * torch.eye(3, dtype=torch.get_default_dtype())
    )
    assert torch.all(system.pbc == torch.tensor([True, True, True]))

    # a cell given in Bohr is converted to Angstrom
    cell_bohr = pbc_gto.Cell(verbose=0)
    cell_bohr.atom = "He 0 0 0"
    cell_bohr.a = np.eye(3) * 4.0
    cell_bohr.basis = "sto-3g"
    cell_bohr.unit = "Bohr"
    cell_bohr.build()

    system_bohr = systems_to_torch(cell_bohr)
    assert torch.allclose(
        system_bohr.cell,
        4.0 * pyscf.data.nist.BOHR * torch.eye(3, dtype=torch.get_default_dtype()),
    )


@pytest.mark.skipif(not HAS_PYSCF, reason="requires pyscf")
def test_pyscf_pseudopotential_cell_types():
    """Types for cells with pseudopotentials.

    Atoms described by a pseudopotential get their element's atomic number
    (14 for Si with gth-pade), not the *valence* charge returned by
    ``atom_charges()`` (4 in this case). This deliberately differs from
    featomic's ``PyscfSystem``, which uses ``atom_charges()``; see
    https://github.com/metatensor/metatomic/issues/51.
    """
    from pyscf.pbc import gto as pbc_gto

    cell = pbc_gto.Cell(verbose=0)
    cell.atom = "Si 0 0 0; Si 1.35775 1.35775 1.35775"
    cell.a = np.array([[0, 2.7155, 2.7155], [2.7155, 0, 2.7155], [2.7155, 2.7155, 0]])
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.unit = "Angstrom"
    cell.build()

    system = systems_to_torch(cell)

    # atom_charges() is 4 (valence) under gth-pade; types are the element Z
    assert list(cell.atom_charges()) == [4, 4]
    assert torch.all(system.types == torch.tensor([14, 14]))
    assert torch.all(system.pbc == torch.tensor([True, True, True]))
    assert torch.allclose(
        system.cell,
        torch.tensor(
            [[0, 2.7155, 2.7155], [2.7155, 0, 2.7155], [2.7155, 2.7155, 0]],
            dtype=torch.get_default_dtype(),
        ),
    )


@pytest.mark.skipif(not HAS_PYSCF, reason="requires pyscf")
def test_pyscf_ecp_molecule_types():
    """Types for molecules with effective core potentials.

    Same as for periodic cells: an ECP atom gets its element's atomic number,
    not the valence count that ``atom_charges()`` returns (pyscf's
    ``make_env`` subtracts the ECP electron count from ``CHARGE_OF`` for
    ``Mole`` objects as well).
    """
    mol = pyscf.gto.M(
        atom="Si 0 0 0; Si 0 0 2.35",
        basis="gth-szv",
        pseudo="gth-pade",
        unit="Angstrom",
        verbose=0,
    )

    system = systems_to_torch(mol)

    assert list(mol.atom_charges()) == [4, 4]
    assert torch.all(system.types == torch.tensor([14, 14]))


def test_unsupported_type_message():
    message = "can be converted to `System`s"
    with pytest.raises(ValueError, match=message):
        systems_to_torch("not a system")
