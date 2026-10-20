"""
Split gradients with :py:class:`TorchModel`.

Gradients declared in ``capabilities()`` are computed by the model itself: they are
part of the requests given to ``forward()``, and the model adds them to its outputs.
All the other gradients of per-system outputs are computed by ``TorchModel`` with
autograd. Here, a Lennard-Jones model computes analytic forces (``positions``
gradients) and leaves the virial (``strain`` gradients) to autograd.

``non_conservative_force`` is a separate per-atom output, not a gradient, so it is never
part of this split.
"""

import numpy as np
import pytest
import torch
from metatensor import Labels, TensorBlock, TensorMap

from metatomic import (
    ModelCapabilities,
    ModelMetadata,
    PairListOptions,
    Quantity,
    System,
    TorchModel,
    execute_model,
)


CUTOFF = 4.5
SIGMA = 3.4
EPSILON = 0.01
PAIRS_OPTIONS = PairListOptions(cutoff=CUTOFF, full_list=False, strict=True)
# the ensemble members are the same potential, scaled by these factors
ENSEMBLE_SCALES = [0.5, 1.0, 2.0]


def lennard_jones(vectors):
    """Energy of each pair, and its derivative with respect to the pair vector"""
    r2 = (vectors**2).sum(dim=1, keepdim=True)
    s6 = (SIGMA**2 / r2) ** 3
    energy = 4 * EPSILON * (s6**2 - s6)
    gradient = 4 * EPSILON * (6 * s6 - 12 * s6**2) / r2 * vectors
    return energy.reshape(-1), gradient


class LennardJones(TorchModel):
    def __init__(self, explicit_forces):
        super().__init__()
        self.explicit_forces = explicit_forces
        # what forward() was asked to compute, as (name, gradients) pairs
        self.received = []

    def capabilities(self):
        explicit = ["positions"] if self.explicit_forces else []
        return ModelCapabilities(
            atomic_types=[18],
            interaction_range=CUTOFF,
            length_unit="angstrom",
            supported_devices=["cpu"],
            dtype="float64",
            outputs=[
                Quantity(
                    name="energy", unit="eV", sample_kind="system", gradients=explicit
                ),
                Quantity(
                    name="energy_ensemble",
                    unit="eV",
                    sample_kind="system",
                    gradients=explicit,
                ),
                Quantity(
                    name="non_conservative_force",
                    unit="eV/angstrom",
                    sample_kind="atom",
                ),
            ],
        )

    def metadata(self):
        return ModelMetadata(name="Lennard-Jones")

    def requested_pair_lists(self, requested_outputs):
        return [PAIRS_OPTIONS]

    def forward(self, systems, selected_atoms, outputs):
        assert selected_atoms is None
        self.received.extend((request.name, request.gradients) for request in outputs)

        energies, gradients = zip(
            *(per_atom_terms(system) for system in systems), strict=True
        )
        scales = torch.tensor(ENSEMBLE_SCALES, dtype=torch.float64)

        results = []
        for request in outputs:
            if request.name == "non_conservative_force":
                # A direct force head, independent of the energy-gradient split.
                results.append(non_conservative_force(systems, gradients))
                continue

            energy = torch.stack([e.sum() for e in energies]).reshape(-1, 1)
            if request.name == "energy":
                properties = Labels.range("energy", 1)
            else:
                assert request.name == "energy_ensemble"
                energy = energy * scales
                properties = Labels.range("energy", len(ENSEMBLE_SCALES))

            block = TensorBlock(
                values=energy,
                samples=Labels.range("system", len(systems)),
                components=[],
                properties=properties,
            )
            if "positions" in request.gradients:
                values = torch.cat(gradients).unsqueeze(-1)
                if request.name == "energy_ensemble":
                    values = values * scales
                block.add_gradient(
                    "positions",
                    TensorBlock(
                        values=values,
                        samples=positions_gradient_samples(systems),
                        components=[Labels.range("xyz", 3)],
                        properties=properties,
                    ),
                )

            results.append(TensorMap(Labels.range("_", 1), [block]))

        return results


def per_atom_terms(system):
    """Pair energies of ``system``, and the gradient of the energy w.r.t. positions"""
    pairs = system.pairs(PAIRS_OPTIONS)
    energy, pair_gradient = lennard_jones(pairs.values.reshape(-1, 3))

    def column(name):
        return torch.from_numpy(np.asarray(pairs.samples.column(name), dtype=np.int64))

    gradient = torch.zeros((len(system), 3), dtype=pair_gradient.dtype)
    gradient.index_add_(0, column("second_atom"), pair_gradient)
    gradient.index_add_(0, column("first_atom"), pair_gradient, alpha=-1)
    return energy, gradient


def positions_gradient_samples(systems):
    samples = [[s, s, a] for s, sys in enumerate(systems) for a in range(len(sys))]
    return Labels(["sample", "system", "atom"], np.array(samples, dtype=np.int32))


def non_conservative_force(systems, gradients):
    samples = [[s, a] for s, system in enumerate(systems) for a in range(len(system))]
    positions = torch.cat([system.positions for system in systems])
    # A rotational drive F = k(-y, x, 0) has nonzero curl, so the total force
    # cannot be obtained as the gradient of a scalar energy. k is in eV/angstrom².
    drive = 0.01 * torch.stack(
        [-positions[:, 1], positions[:, 0], torch.zeros_like(positions[:, 0])], dim=1
    )
    block = TensorBlock(
        values=(-torch.cat(gradients) + drive).unsqueeze(-1),
        samples=Labels(["system", "atom"], np.array(samples, dtype=np.int32)),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("non_conservative_force", 1),
    )
    return TensorMap(Labels.range("_", 1), [block])


def make_system(seed):
    """FCC argon with 4 atoms and rattled positions, with its half pair list"""
    rng = np.random.default_rng(seed)
    cell = 5.2 * np.eye(3)
    fractional = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    positions = fractional @ cell + rng.normal(scale=0.1, size=(4, 3))

    pairs, vectors = [], []
    shifts = np.array(np.meshgrid(*[range(-1, 2)] * 3)).reshape(3, -1).T
    for i in range(4):
        for j in range(i, 4):
            for shift in shifts:
                # half list: (i, i, -S) is the same pair as (i, i, S)
                if i == j and tuple(shift) <= (0, 0, 0):
                    continue
                vector = positions[j] - positions[i] + shift @ cell
                if np.linalg.norm(vector) < CUTOFF:
                    pairs.append([i, j, *shift])
                    vectors.append(vector)

    system = System(
        "angstrom",
        np.full(4, 18, dtype=np.int32),
        positions,
        cell,
        np.array([True, True, True]),
    )
    block = TensorBlock(
        values=np.array(vectors).reshape(-1, 3, 1),
        samples=Labels(
            ["first_atom", "second_atom"] + [f"cell_shift_{d}" for d in "abc"],
            np.array(pairs, dtype=np.int32),
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )
    system.add_pairs(PAIRS_OPTIONS, block)
    return system, np.array(vectors)


def expected_virial(vectors):
    """dE/dε = sum_p r_p ⊗ dE/dr_p, for all the pairs p in the list"""
    _, gradient = lennard_jones(torch.from_numpy(vectors))
    return np.einsum("pa,pb->ab", vectors, gradient.numpy())


ENERGY = Quantity(
    name="energy", unit="eV", sample_kind="system", gradients=["positions", "strain"]
)


def test_capabilities():
    # strain is always added by TorchModel, positions is either explicit or autograd
    for explicit_forces in [True, False]:
        model = LennardJones(explicit_forces)
        energy, ensemble, nc_force = model.capabilities().outputs
        assert sorted(energy.gradients) == ["positions", "strain"]
        assert sorted(ensemble.gradients) == ["positions", "strain"]
        assert nc_force.gradients == []


def test_split():
    model = LennardJones(explicit_forces=True)
    systems = [make_system(seed) for seed in range(2)]
    results = execute_model(
        model, [s for s, _ in systems], [ENERGY], check_consistency=True
    )

    # the model only had to compute the forces, autograd took care of the virial
    assert model.received == [("energy", ["positions"])]

    reference = execute_model(
        LennardJones(explicit_forces=False),
        [make_system(seed)[0] for seed in range(2)],
        [ENERGY],
        check_consistency=True,
    )

    block, reference = results[0].block(), reference[0].block()
    np.testing.assert_allclose(block.values, reference.values)
    for parameter in ["positions", "strain"]:
        gradient, expected = block.gradient(parameter), reference.gradient(parameter)
        assert gradient.samples == expected.samples
        np.testing.assert_allclose(
            gradient.values,
            expected.values,
            rtol=1e-10,
            atol=1e-12,
        )

    for i, (_, vectors) in enumerate(systems):
        np.testing.assert_allclose(
            block.gradient("strain").values[i, :, :, 0],
            expected_virial(vectors),
            rtol=1e-10,
            atol=1e-12,
        )


def test_only_explicit_or_only_autograd():
    system, vectors = make_system(seed=3)

    # only explicit gradients: autograd is not used at all
    model = LennardJones(explicit_forces=True)
    request = Quantity(
        name="energy", unit="eV", sample_kind="system", gradients=["positions"]
    )
    results = execute_model(model, [system], [request], check_consistency=True)
    assert model.received == [("energy", ["positions"])]
    assert results[0].block().gradients_list() == ["positions"]

    # only autograd gradients: the model does not compute any gradient
    request = Quantity(
        name="energy", unit="eV", sample_kind="system", gradients=["strain"]
    )
    model.received = []
    results = execute_model(model, [system], [request], check_consistency=True)
    assert model.received == [("energy", [])]
    np.testing.assert_allclose(
        results[0].block().gradient("strain").values[0, :, :, 0],
        expected_virial(vectors),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("explicit_forces", [False, True])
def test_energy_ensemble(explicit_forces):
    # autograd computes one gradient per ensemble member, which is what LLPR in
    # metatrain implements by hand with `explicit_gradients`
    ensemble = Quantity(
        name="energy_ensemble",
        unit="eV",
        sample_kind="system",
        gradients=["positions", "strain"],
    )
    system, _ = make_system(seed=4)

    model = LennardJones(explicit_forces=explicit_forces)
    results = execute_model(model, [system], [ENERGY, ensemble], check_consistency=True)
    requested = ["positions"] if explicit_forces else []
    assert model.received == [("energy", requested), ("energy_ensemble", requested)]

    reference = execute_model(
        LennardJones(explicit_forces=False),
        [make_system(seed=4)[0]],
        [ENERGY],
        check_consistency=True,
    )
    energy, ensemble = reference[0].block(), results[1].block()
    for parameter in ["positions", "strain"]:
        np.testing.assert_allclose(
            ensemble.gradient(parameter).values,
            np.asarray(energy.gradient(parameter).values) * ENSEMBLE_SCALES,
            rtol=1e-10,
            atol=1e-12,
        )


def test_non_conservative_force():
    system, _ = make_system(seed=5)
    nc_force = Quantity(
        name="non_conservative_force", unit="eV/angstrom", sample_kind="atom"
    )

    model = LennardJones(explicit_forces=False)
    energy, forces = execute_model(
        model, [system], [ENERGY, nc_force], check_consistency=True
    )

    # the non-conservative forces are passed through untouched, without gradients
    assert model.received == [("energy", []), ("non_conservative_force", [])]
    assert forces.block().gradients_list() == []
    positions = np.asarray(system.positions)
    drive = 0.01 * np.stack(
        [-positions[:, 1], positions[:, 0], np.zeros(len(system))], axis=1
    )
    conservative = -np.asarray(energy.block().gradient("positions").values)
    np.testing.assert_allclose(
        forces.block().values,
        conservative + drive[:, :, None],
        rtol=1e-10,
        atol=1e-12,
    )
    assert not np.allclose(forces.block().values, conservative)

    # they are not a gradient, so autograd can not compute their gradients
    nc_force = Quantity(
        name="non_conservative_force",
        unit="eV/angstrom",
        sample_kind="atom",
        gradients=["positions"],
    )
    message = (
        "gradients of 'non_conservative_force' with sample kind 'atom' can not be "
        "computed with autograd"
    )
    with pytest.raises(ValueError, match=message):
        execute_model(model, [system], [nc_force])
