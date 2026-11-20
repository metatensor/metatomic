import math

import numpy as np
import pytest
import torch
import vesin.metatomic
from metatensor.torch import Labels

from metatomic.torch import (
    ModelEvaluationOptions,
    ModelOutput,
    System,
    load_atomistic_model,
)
from metatomic.torch.testing import (
    QTIP4PF,
    _ewald_n_max,
    _ewald_reciprocal,
    _inverse_cell,
    qtip4pf_model,
)


# Parameters from Table I of Habershon, Markland, Manolopoulos, J. Chem. Phys. 131,
# 024501 (2009). These are typed again here on purpose, to check the ones in the model
EPSILON = 0.1852  # kcal/mol
SIGMA = 3.1589  # A
Q_M = 1.1128  # e
GAMMA = 0.73612
D_R = 116.09  # kcal/mol
ALPHA_R = 2.287  # 1/A
R_EQ = 0.9419  # A
K_THETA = 87.85  # kcal/mol/rad^2
THETA_EQ = 107.4  # degrees

# e^2 / (4 pi eps_0) in kcal/mol A, from SI constants
K_E = (
    1.602176634e-19**2
    / (4 * math.pi * 8.8541878128e-12)
    * 6.02214076e23
    / 4184.0
    * 1e10
)
KCAL_MOL_TO_EV = 4184.0 / (6.02214076e23 * 1.602176634e-19)

NACL_MADELUNG = 1.747564594633182

# Energy (kcal/mol), forces (kcal/mol/A) and virial (-dE/dstrain, kcal/mol) for the
# structure defined in `water_box()`, with `ewald_accuracy=1e-10`. Calculated with
# https://github.com/ceriottm/qtip4pf-test, which was itself checked against the
# q-TIP4P/F implementation in the Fortran driver of i-PI.
_REFERENCE = {
    "energy": 42.22091597659619,
    "forces": np.array(
        [
            [-8.9591844486, 42.7937933218, 4.5532895562],
            [-20.8927310990, -23.4870462764, 22.2923282507],
            [-5.8561009445, -62.5603827814, -3.8935348920],
            [-30.0404560352, -7.4163438385, -37.3666114997],
            [15.0992750535, 1.6015624495, 29.0108882128],
            [10.1443782084, 1.9948968967, 6.4492678794],
            [29.6496829038, -38.9499517558, -0.2583891388],
            [3.7951839413, 61.0949164058, -9.3093303764],
            [-9.5563965167, 4.1879688066, 13.7642783040],
            [21.7287931195, 4.7998838536, 4.0528684090],
            [-1.6792048684, -7.1584672410, -21.9619625819],
            [-7.1905612794, 13.1648992031, 23.9666942045],
            [-4.9383859748, -31.6877197543, 10.1808453493],
            [27.7862088434, 30.3445922040, -22.5405297215],
            [6.4980762730, 4.6682006409, -5.3055316963],
            [4.1178782328, -45.4037430096, -19.8391493805],
            [13.3037462349, 11.1205068980, 2.8502642517],
            [-11.4655190114, 33.6909885304, 3.0133398373],
            [44.7428308157, 9.2867132789, 39.4831081340],
            [-25.7293276168, 7.2144151753, -40.8937601225],
            [-37.9662780262, -5.8414360511, -10.3055787858],
            [-31.2903005902, 28.3156694856, 8.9251962941],
            [-7.1435195769, -5.7412144171, 9.5255897542],
            [25.8419123619, -26.0327020247, -6.3935802417],
        ]
    ),
    "virial": np.array(
        [
            [88.333955738, -30.121265339, 32.770664951],
            [-30.121265339, 20.554746953, -25.727403962],
            [32.770664951, -25.727403962, 159.144593879],
        ]
    ),
}


def rotation(alpha, beta):
    """Rotation by ``alpha`` around z followed by a rotation by ``beta`` around x"""
    cos_a, sin_a = math.cos(alpha), math.sin(alpha)
    cos_b, sin_b = math.cos(beta), math.sin(beta)
    r_z = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cos_b, -sin_b], [0.0, sin_b, cos_b]])
    return r_x @ r_z


def water_molecule(r_1=R_EQ, r_2=R_EQ, theta=THETA_EQ):
    """O, H, H positions of a water molecule, with the oxygen at the origin"""
    theta = math.radians(theta)
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [r_1, 0.0, 0.0],
            [r_2 * math.cos(theta), r_2 * math.sin(theta), 0.0],
        ]
    )


def water_box():
    """
    8 distorted water molecules on a perturbed cubic lattice with a density close to
    1 g/cm^3. Returns the types, positions and cell.
    """
    spacing = 3.1
    positions = []
    for m in range(8):
        i, j, k = m // 4, (m // 2) % 2, m % 2
        center = spacing * np.array([i, j, k])
        center = center + 0.2 * np.sin([1.0 + m, 2.0 * m, 3.0 - m])

        molecule = water_molecule(
            r_1=R_EQ + 0.03 * math.sin(4.0 * m),
            r_2=R_EQ + 0.03 * math.cos(5.0 * m),
            theta=THETA_EQ + 4.0 * math.sin(6.0 + m),
        )
        positions.append(center + molecule @ rotation(0.7 * m, 1.3 * m + 0.4).T)

    types = np.array([8, 1, 1] * 8)
    return types, np.concatenate(positions), 2 * spacing * np.eye(3)


def triclinic(cell):
    """Another (sheared) representation of the same lattice"""
    a, b, c = cell
    return np.array([a, b + a, c - 2 * b + a])


def make_system(model, types, positions, cell, pbc=True):
    """Create a :py:class:`System` with the neighbor lists requested by ``model``"""
    system = System(
        types=torch.as_tensor(types, dtype=torch.int32),
        positions=torch.as_tensor(positions, dtype=torch.float64),
        cell=torch.as_tensor(cell, dtype=torch.float64),
        pbc=torch.tensor([pbc, pbc, pbc]),
    )

    calculators = vesin.metatomic.neighbor_lists_for_model(
        "angstrom", model, check_consistency=True
    )
    for calculator in calculators:
        calculator.add_neighbor_list([system])

    return system


def run_model(model, systems, unit="kcal/mol"):
    """Get the energy of each system as a 1D tensor"""
    options = ModelEvaluationOptions(
        length_unit="angstrom",
        outputs={"energy": ModelOutput(unit=unit, sample_kind="system")},
    )
    outputs = model(systems, options, check_consistency=True)

    block = outputs["energy"].block()
    assert block.samples.names == ["system"]
    assert block.properties.names == ["energy"]
    return block.values.reshape(-1)


def energy_only(model, types, positions, cell, pbc=True):
    system = make_system(model, types, positions, cell, pbc)
    with torch.no_grad():
        return run_model(model, [system]).item()


def evaluate(model, types, positions, cell, pbc=True):
    """
    Compute the energy, forces and virial (``-dE/dstrain``) with autograd, in the same
    way simulation engines do.
    """
    positions = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    cell = torch.tensor(cell, dtype=torch.float64)
    strain = torch.eye(3, dtype=torch.float64, requires_grad=True)

    system = make_system(model, types, positions @ strain, cell @ strain, pbc)
    energy = run_model(model, [system]).sum()
    positions_grad, strain_grad = torch.autograd.grad(energy, [positions, strain])
    return energy.item(), -positions_grad.numpy(), -strain_grad.numpy()


def reference_intramolecular(oxygen, hydrogen_1, hydrogen_2):
    def stretch(r):
        dr = r - R_EQ
        return D_R * (
            ALPHA_R**2 * dr**2 - ALPHA_R**3 * dr**3 + 7 / 12 * ALPHA_R**4 * dr**4
        )

    r_1 = np.linalg.norm(hydrogen_1 - oxygen)
    r_2 = np.linalg.norm(hydrogen_2 - oxygen)
    cos_theta = np.dot(hydrogen_1 - oxygen, hydrogen_2 - oxygen) / (r_1 * r_2)
    bend = 0.5 * K_THETA * (math.acos(cos_theta) - math.radians(THETA_EQ)) ** 2
    return stretch(r_1) + stretch(r_2) + bend


def reference_energy(positions, cutoff):
    """Energy of a non-periodic system, computed with explicit loops over pairs"""

    def charged_sites(oxygen, hydrogen_1, hydrogen_2):
        m_site = GAMMA * oxygen + (1 - GAMMA) * (hydrogen_1 + hydrogen_2) / 2
        return [(hydrogen_1, Q_M / 2), (hydrogen_2, Q_M / 2), (m_site, -Q_M)]

    molecules = positions.reshape(-1, 3, 3)
    energy = sum(reference_intramolecular(*molecule) for molecule in molecules)
    for i in range(len(molecules)):
        for j in range(i + 1, len(molecules)):
            r = np.linalg.norm(molecules[i][0] - molecules[j][0])
            if r < cutoff:
                energy += 4 * EPSILON * ((SIGMA / r) ** 12 - (SIGMA / r) ** 6)

            for r_i, q_i in charged_sites(*molecules[i]):
                for r_j, q_j in charged_sites(*molecules[j]):
                    energy += K_E * q_i * q_j / np.linalg.norm(r_i - r_j)

    return energy


def test_model():
    model = qtip4pf_model(cutoff=7.5)
    assert model.metadata().name == "Test q-TIP4P/F"

    capabilities = model.capabilities()
    assert capabilities.atomic_types == [1, 8]
    assert math.isinf(capabilities.interaction_range)
    assert capabilities.length_unit == "angstrom"
    assert capabilities.dtype == "float64"
    assert list(capabilities.outputs.keys()) == ["energy"]
    assert capabilities.outputs["energy"].unit == "kcal/mol"
    assert capabilities.outputs["energy"].sample_kind == "system"

    requested = model.requested_neighbor_lists()
    assert len(requested) == 1
    assert requested[0].cutoff == 7.5
    assert not requested[0].full_list


@pytest.mark.parametrize(
    "distortion",
    [(0.0, 0.0, 0.0), (0.08, -0.05, 0.0), (0.0, 0.0, -9.0), (-0.1, 0.12, 14.0)],
)
def test_single_molecule(distortion):
    delta_1, delta_2, delta_theta = distortion
    molecule = water_molecule(R_EQ + delta_1, R_EQ + delta_2, THETA_EQ + delta_theta)
    positions = molecule @ rotation(0.3, 1.9).T + np.array([1.0, -2.0, 0.5])

    energy = energy_only(
        qtip4pf_model(), [8, 1, 1], positions, np.zeros((3, 3)), pbc=False
    )
    expected = reference_intramolecular(*positions)
    assert energy == pytest.approx(expected, rel=1e-12, abs=1e-12)

    if distortion == (0.0, 0.0, 0.0):
        assert expected == pytest.approx(0.0, abs=1e-12)
    else:
        assert expected > 0.1


def test_cluster():
    types, positions, _ = water_box()

    # some of the O-O distances are outside of the Lennard-Jones cutoff
    cutoff = 5.0
    distances = np.linalg.norm(positions[3::3] - positions[0], axis=1)
    assert np.any(distances < cutoff)
    assert np.any(distances > cutoff)

    model = qtip4pf_model(cutoff=cutoff)
    energy = energy_only(model, types, positions, np.zeros((3, 3)), pbc=False)
    assert energy == pytest.approx(reference_energy(positions, cutoff), rel=1e-12)


@pytest.mark.parametrize("cell", ["conventional", "primitive"])
def test_ewald_madelung(cell):
    a = 1.0
    if cell == "conventional":
        cell = a * np.eye(3)
        sodium = a * np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
        positions = np.concatenate([sodium, sodium + a * np.array([0.5, 0, 0])])
        charges = [1.0] * 4 + [-1.0] * 4
    else:
        # triclinic cell
        cell = a / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        positions = np.array([[0, 0, 0], [a / 2, a / 2, a / 2]])
        charges = [1.0, -1.0]

    positions = torch.tensor(positions, dtype=torch.float64)
    charges = torch.tensor(charges, dtype=torch.float64)
    cell = torch.tensor(cell, dtype=torch.float64)

    # With a very large alpha, the real-space sum is negligible and only the
    # reciprocal and self terms contribute.
    alpha = 12.0
    k_cut = 2 * alpha * math.sqrt(-math.log(1e-16))

    inverse, volume = _inverse_cell(cell.unsqueeze(0))
    lengths = torch.linalg.vector_norm(cell, dim=1).tolist()
    reciprocal = _ewald_reciprocal(
        (positions @ inverse[0]).unsqueeze(0),
        charges,
        inverse,
        volume,
        _ewald_n_max(lengths, k_cut),
        alpha,
        k_cut,
        1.0,
    )
    energy = reciprocal[0] - alpha / math.sqrt(math.pi) * torch.sum(charges**2)

    # nearest neighbor distance is a/2
    expected = -len(charges) / 2 * NACL_MADELUNG / (a / 2)
    assert energy.item() == pytest.approx(expected, rel=1e-12)


def test_ewald_accuracy():
    # Changing ewald_accuracy changes the splitting parameter and the reciprocal-space
    # cutoff. The energy should not change, checking the consistency of the real-space,
    # reciprocal-space, self-interaction and intramolecular exclusion terms.
    types, positions, cell = water_box()

    energies = []
    for accuracy in [1e-10, 1e-12, 1e-14]:
        model = qtip4pf_model(ewald_accuracy=accuracy)
        energies.append(energy_only(model, types, positions, cell))

    assert energies[1] == pytest.approx(energies[0], rel=1e-9)
    assert energies[2] == pytest.approx(energies[0], rel=1e-9)

    # the default accuracy is enough for practical purposes
    energy = energy_only(qtip4pf_model(), types, positions, cell)
    assert energy == pytest.approx(energies[0], abs=1e-3)
    assert energy != energies[0]


def test_reference():
    types, positions, cell = water_box()
    model = qtip4pf_model(ewald_accuracy=1e-10)

    energy, forces, virial = evaluate(model, types, positions, cell)
    assert energy == pytest.approx(_REFERENCE["energy"], rel=1e-10)
    np.testing.assert_allclose(forces, _REFERENCE["forces"], atol=1e-8)
    np.testing.assert_allclose(virial, _REFERENCE["virial"], atol=1e-7)

    # forces sum to zero
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-9)

    # different parameters give different results
    model = qtip4pf_model(ewald_accuracy=1e-10, parameters={"sigma": 3.16})
    energy = energy_only(model, types, positions, cell)
    assert energy != pytest.approx(_REFERENCE["energy"], rel=1e-4)


@pytest.mark.parametrize("shape", ["cubic", "triclinic"])
def test_finite_differences(shape):
    types, positions, cell = water_box()
    if shape == "triclinic":
        cell = triclinic(cell)

    model = qtip4pf_model(ewald_accuracy=1e-10)
    _, forces, virial = evaluate(model, types, positions, cell)

    delta = 1e-5
    # one oxygen and two hydrogens from different molecules
    for atom in [3, 10, 20]:
        for xyz in range(3):
            displaced = positions.copy()
            displaced[atom, xyz] += delta
            energy_plus = energy_only(model, types, displaced, cell)
            displaced[atom, xyz] -= 2 * delta
            energy_minus = energy_only(model, types, displaced, cell)

            expected = -(energy_plus - energy_minus) / (2 * delta)
            assert forces[atom, xyz] == pytest.approx(expected, rel=1e-6, abs=1e-5)

    np.testing.assert_allclose(virial, virial.T, atol=1e-8)
    for xyz_1 in range(3):
        for xyz_2 in range(3):
            strain = np.eye(3)
            strain[xyz_1, xyz_2] += delta
            energy_plus = energy_only(model, types, positions @ strain, cell @ strain)
            strain[xyz_1, xyz_2] -= 2 * delta
            energy_minus = energy_only(model, types, positions @ strain, cell @ strain)

            expected = -(energy_plus - energy_minus) / (2 * delta)
            assert virial[xyz_1, xyz_2] == pytest.approx(expected, rel=1e-6, abs=1e-4)


def test_invariances():
    types, positions, cell = water_box()
    model = qtip4pf_model(ewald_accuracy=1e-10)
    energy, forces, virial = evaluate(model, types, positions, cell)

    # translation
    translated = positions + np.array([0.3, -1.7, 12.1])
    new_energy = energy_only(model, types, translated, cell)
    assert new_energy == pytest.approx(energy, rel=1e-12)

    # wrapping all atoms inside the cell breaks molecules across the boundary
    fractional = positions @ np.linalg.inv(cell)
    wrapped = (fractional - np.floor(fractional)) @ cell
    assert not np.allclose(wrapped, positions)

    new_energy, new_forces, new_virial = evaluate(model, types, wrapped, cell)
    assert new_energy == pytest.approx(energy, rel=1e-12)
    np.testing.assert_allclose(new_forces, forces, atol=1e-9)
    np.testing.assert_allclose(new_virial, virial, atol=1e-8)

    # rigid rotation of the whole system, including the cell
    matrix = rotation(0.4, 2.1)
    new_energy, new_forces, new_virial = evaluate(
        model, types, positions @ matrix.T, cell @ matrix.T
    )
    assert new_energy == pytest.approx(energy, rel=1e-10)
    np.testing.assert_allclose(new_forces, forces @ matrix.T, atol=1e-7)
    np.testing.assert_allclose(new_virial, matrix @ virial @ matrix.T, atol=1e-6)

    # different representation of the same lattice
    new_energy, new_forces, new_virial = evaluate(
        model, types, positions, triclinic(cell)
    )
    assert new_energy == pytest.approx(energy, rel=1e-10)
    np.testing.assert_allclose(new_forces, forces, atol=1e-7)
    np.testing.assert_allclose(new_virial, virial, atol=1e-6)

    # supercell
    super_cell = cell.copy()
    super_cell[0] *= 2
    new_energy, new_forces, new_virial = evaluate(
        model,
        np.concatenate([types, types]),
        np.concatenate([positions, positions + cell[0]]),
        super_cell,
    )
    assert new_energy == pytest.approx(2 * energy, rel=1e-10)
    np.testing.assert_allclose(new_forces[: len(types)], forces, atol=1e-7)
    np.testing.assert_allclose(new_virial, 2 * virial, atol=1e-6)


@pytest.mark.parametrize(
    "selection",
    [
        # periodic systems with the same size, evaluated together
        [0, 1, 2],
        # non-periodic systems with the same size, evaluated together
        [3, 3],
        # different sizes and periodicity, evaluated one by one
        [0, 3, 4, 1],
    ],
)
def test_multiple_systems(selection):
    types, positions, cell = water_box()
    structures = [
        (types, positions, cell, True),
        (types, positions + 0.05 * np.sin(np.arange(72.0)).reshape(24, 3), cell, True),
        # different cell shape and size, using a different reciprocal-space grid
        (types, 1.1 * positions, 1.1 * triclinic(cell), True),
        (types, positions, np.zeros((3, 3)), False),
        (types[:12], positions[:12], 1.5 * cell, True),
    ]

    model = qtip4pf_model(ewald_accuracy=1e-8)

    def energies_and_gradients(structures):
        systems = []
        inputs = []
        for types, positions, cell, pbc in structures:
            positions = torch.tensor(positions, requires_grad=True)
            cell = torch.tensor(cell, requires_grad=pbc)
            systems.append(make_system(model, types, positions, cell, pbc))
            inputs.append(positions)
            if pbc:
                inputs.append(cell)

        energies = run_model(model, systems)
        gradients = torch.autograd.grad(energies.sum(), inputs)
        return energies.detach(), gradients

    structures = [structures[i] for i in selection]
    energies, gradients = energies_and_gradients(structures)
    assert energies.shape == (len(selection),)

    expected_energies = []
    expected_gradients = []
    for structure in structures:
        energy, structure_gradients = energies_and_gradients([structure])
        expected_energies.append(energy)
        expected_gradients.extend(structure_gradients)

    # The results are not bitwise identical, since math functions on CPU can give
    # slightly different results between calls
    torch.testing.assert_close(
        energies, torch.cat(expected_energies), rtol=1e-10, atol=0.0
    )
    for gradient, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, expected, rtol=1e-8, atol=1e-8)


def test_export(tmp_path):
    model = qtip4pf_model()
    path = str(tmp_path / "qtip4pf.pt")
    model.save(path)
    loaded = load_atomistic_model(path)

    assert loaded.metadata().name == "Test q-TIP4P/F"
    assert math.isinf(loaded.capabilities().interaction_range)

    types, positions, cell = water_box()
    systems = [
        make_system(loaded, types, torch.tensor(positions, requires_grad=True), cell),
        make_system(loaded, types, positions, np.zeros((3, 3)), pbc=False),
    ]

    # the exported model gives the same results as the Python one, and the energy is
    # converted to the requested unit
    energies = run_model(loaded, systems, unit="eV")
    expected = run_model(model, systems, unit="kcal/mol") * KCAL_MOL_TO_EV
    torch.testing.assert_close(energies, expected, rtol=1e-10, atol=0.0)
    assert energies[0].item() != pytest.approx(energies[1].item(), rel=1e-3)

    # the exported model can compute forces with autograd
    (gradient,) = torch.autograd.grad(energies[0], [systems[0].positions])
    np.testing.assert_allclose(
        -gradient.numpy(), _REFERENCE["forces"] * KCAL_MOL_TO_EV, atol=1e-5
    )


def test_selected_atoms():
    types, positions, cell = water_box()
    model = QTIP4PF()
    system = make_system(qtip4pf_model(), types, positions, cell)
    outputs = {"energy": ModelOutput(unit="kcal/mol", sample_kind="system")}

    # all atoms can be selected explicitly
    selected = Labels(
        ["system", "atom"],
        torch.tensor([[0, atom] for atom in range(len(types))], dtype=torch.int32),
    )
    energy = model([system], outputs, selected)["energy"].block().values
    expected = model([system], outputs, None)["energy"].block().values
    assert torch.all(energy == expected)

    message = "q-TIP4P/F can not compute the energy of a subset of atoms"
    with pytest.raises(ValueError, match=message):
        model([system], outputs, Labels(["system", "atom"], selected.values[:6]))

    # nothing is computed if the energy is not requested
    assert model([system], {}, None) == {}


def test_errors():
    message = "ewald_accuracy must be between 0 and 1, got 2.0"
    with pytest.raises(ValueError, match=message):
        QTIP4PF(ewald_accuracy=2.0)

    message = "cutoff must be larger than 4 A, got 3.5"
    with pytest.raises(ValueError, match=message):
        QTIP4PF(cutoff=3.5)

    message = "unknown q-TIP4P/F parameter 'q_H'"
    with pytest.raises(ValueError, match=message):
        QTIP4PF(parameters={"q_H": 0.5})

    types, positions, cell = water_box()
    model = qtip4pf_model()
    module = QTIP4PF()
    outputs = {"energy": ModelOutput(unit="kcal/mol", sample_kind="system")}

    message = "q-TIP4P/F expects atoms ordered as O, H, H, O, H, H, ..."
    system = make_system(model, np.roll(types, 1), positions, cell)
    with pytest.raises(ValueError, match=message):
        module([system], outputs, None)

    system = make_system(model, types[:-1], positions[:-1], cell)
    with pytest.raises(ValueError, match=message):
        module([system], outputs, None)

    message = "q-TIP4P/F only supports fully periodic or non-periodic systems"
    system = System(
        types=torch.tensor(types, dtype=torch.int32),
        positions=torch.tensor(positions),
        cell=torch.diag(torch.tensor([20.0, 20.0, 0.0], dtype=torch.float64)),
        pbc=torch.tensor([True, True, False]),
    )
    with pytest.raises(ValueError, match=message):
        module([system], outputs, None)

    message = "q-TIP4P/F only provides the total energy of each system"
    system = make_system(model, types, positions, cell)
    outputs = {"energy": ModelOutput(unit="kcal/mol", sample_kind="atom")}
    with pytest.raises(ValueError, match=message):
        module([system], outputs, None)
