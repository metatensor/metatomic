import os
from typing import Dict, List, Optional

import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
    load_atomistic_model,
)
from metatomic.torch.weighted_sum import WeightedSum


ATOMIC_NUMBER = 6


class MultiHeadEnergyModel(torch.nn.Module):
    """Toy model exposing three energy heads, each a different (nonlinear)
    function of the atomic positions so that per-head forces/stresses differ."""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        results = torch.jit.annotate(Dict[str, TensorMap], {})
        keys = Labels("_", torch.tensor([[0]], dtype=torch.int64))
        properties = Labels("energy", torch.tensor([[0]], dtype=torch.int64))

        for name in outputs:
            coeff = 0.0
            if name == "energy/pbe":
                coeff = 1.0
            elif name == "energy/r2scan":
                coeff = 2.7
            elif name == "energy/lda":
                coeff = 0.5
            else:
                continue

            values = torch.jit.annotate(List[torch.Tensor], [])
            samples = torch.jit.annotate(List[torch.Tensor], [])
            for i, system in enumerate(systems):
                n_atoms = system.positions.shape[0]
                atom_indices = torch.arange(n_atoms, dtype=torch.int64)
                if selected_atoms is not None:
                    selected_values = selected_atoms.values.to(torch.int64)
                    mask = selected_values[:, 0] == i
                    atom_indices = selected_values[mask, 1]

                positions = system.positions.index_select(0, atom_indices)
                values.append(coeff * (positions**2).sum(dim=1, keepdim=True))
                samples.append(
                    torch.cat(
                        [
                            torch.full(
                                (atom_indices.shape[0], 1), i, dtype=torch.int64
                            ),
                            atom_indices.reshape(-1, 1),
                        ],
                        dim=1,
                    )
                )

            block = TensorBlock(
                values=torch.cat(values, dim=0),
                samples=Labels(["system", "atom"], torch.cat(samples, dim=0)),
                components=torch.jit.annotate(List[Labels], []),
                properties=properties,
            )
            results[name] = TensorMap(keys, [block])

        return results


@pytest.fixture
def model():
    return AtomisticModel(
        MultiHeadEnergyModel().eval(),
        ModelMetadata(),
        ModelCapabilities(
            outputs={
                "energy/pbe": ModelOutput(
                    sample_kind="atom", unit="eV", description="PBE energy head"
                ),
                "energy/r2scan": ModelOutput(
                    sample_kind="atom", unit="eV", description="r2SCAN energy head"
                ),
                "energy/lda": ModelOutput(
                    sample_kind="atom", unit="eV", description="LDA energy head"
                ),
                "test::extra": ModelOutput(
                    sample_kind="atom", unit="", description="unrelated output"
                ),
            },
            atomic_types=[ATOMIC_NUMBER],
            interaction_range=0.0,
            length_unit="Angstrom",
            supported_devices=["cpu", "cuda"],
            dtype="float64",
        ),
    )


@pytest.fixture
def weights():
    return {"energy/pbe": 0.6, "energy/r2scan": 0.3, "energy/lda": 0.1}


def _system(with_strain=False):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.3, 0.2, -0.4], [-0.5, 1.1, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    cell = torch.eye(3, dtype=torch.float64) * 10.0

    strain = None
    if with_strain:
        strain = torch.eye(3, dtype=torch.float64, requires_grad=True)
        positions = positions @ strain
        positions.retain_grad()
        cell = cell @ strain

    system = System(
        types=torch.full((3,), ATOMIC_NUMBER, dtype=torch.int32),
        positions=positions,
        cell=cell,
        pbc=torch.tensor([True, True, True]),
    )
    return system, strain


def _eval(model, system, outputs, selected_atoms=None):
    options = ModelEvaluationOptions(
        length_unit="Angstrom", outputs=outputs, selected_atoms=selected_atoms
    )
    return model([system], options, check_consistency=True)


def test_weighted_sum_wrap_capabilities(model, weights):
    wrapped = WeightedSum.wrap(model, "energy", weights)
    capabilities = wrapped.capabilities()

    assert "energy" in capabilities.outputs
    new_output = capabilities.outputs["energy"]
    assert new_output.unit == "eV"
    assert new_output.sample_kind == "atom"
    for head in weights:
        assert head in new_output.description

    # the original outputs are all preserved
    for name in ["energy/pbe", "energy/r2scan", "energy/lda", "test::extra"]:
        assert name in capabilities.outputs


def test_weighted_sum_values_match_manual_combination(model, weights):
    wrapped = WeightedSum.wrap(model, "energy", weights)
    system, _ = _system()

    results = _eval(
        wrapped,
        system,
        {"energy": ModelOutput(unit="eV", sample_kind="atom")},
    )
    combined = results["energy"].block().values

    raw_outputs = {name: ModelOutput(unit="eV", sample_kind="atom") for name in weights}
    raw = _eval(model, system, raw_outputs)
    expected = sum(w * raw[name].block().values for name, w in weights.items())

    assert torch.allclose(combined, expected)


def test_weighted_sum_forces_and_stress_match_reference(model, weights):
    """A single backward pass through the weighted-sum output must give forces
    and stresses equal to the weighted sum of the forces/stresses of the
    individual heads."""
    wrapped = WeightedSum.wrap(model, "energy", weights)
    system, strain = _system(with_strain=True)

    results = _eval(
        wrapped, system, {"energy": ModelOutput(unit="eV", sample_kind="atom")}
    )
    total_energy = results["energy"].block().values.sum()
    total_energy.backward()
    combined_forces = -system.positions.grad.clone()
    combined_stress = strain.grad.clone()

    expected_forces = torch.zeros_like(combined_forces)
    expected_stress = torch.zeros_like(combined_stress)
    for name, weight in weights.items():
        head_system, head_strain = _system(with_strain=True)
        head_results = _eval(
            model, head_system, {name: ModelOutput(unit="eV", sample_kind="atom")}
        )
        head_energy = head_results[name].block().values.sum()
        head_energy.backward()
        expected_forces += weight * (-head_system.positions.grad)
        expected_stress += weight * head_strain.grad

    np.testing.assert_allclose(
        combined_forces.detach().numpy(), expected_forces.detach().numpy()
    )
    np.testing.assert_allclose(
        combined_stress.detach().numpy(), expected_stress.detach().numpy()
    )
    # sanity check: the heads are not all identical, so this is a real test of
    # the weighted combination and not a coincidence
    assert not torch.allclose(expected_forces, torch.zeros_like(expected_forces))


def test_weighted_sum_head_and_sum_requested_together(model, weights):
    wrapped = WeightedSum.wrap(model, "energy", weights)
    system, _ = _system()

    results = _eval(
        wrapped,
        system,
        {
            "energy": ModelOutput(unit="eV", sample_kind="atom"),
            "energy/pbe": ModelOutput(unit="eV", sample_kind="atom"),
        },
    )

    raw = _eval(model, system, {"energy/pbe": ModelOutput(sample_kind="atom")})
    assert torch.allclose(
        results["energy/pbe"].block().values, raw["energy/pbe"].block().values
    )

    expected_combined = sum(
        w
        * _eval(model, system, {name: ModelOutput(sample_kind="atom")})[name]
        .block()
        .values
        for name, w in weights.items()
    )
    assert torch.allclose(results["energy"].block().values, expected_combined)


def test_weighted_sum_passthrough_when_sum_not_requested(model, weights):
    """When the weighted-sum output itself is not requested, WeightedSum must
    be a pure passthrough to the wrapped model, without evaluating any extra
    head that is not asked for."""
    wrapped = WeightedSum.wrap(model, "energy", weights)
    system, _ = _system()

    results = _eval(
        wrapped, system, {"energy/pbe": ModelOutput(unit="eV", sample_kind="atom")}
    )
    raw = _eval(model, system, {"energy/pbe": ModelOutput(sample_kind="atom")})

    assert set(results.keys()) == {"energy/pbe"}
    assert torch.allclose(
        results["energy/pbe"].block().values, raw["energy/pbe"].block().values
    )


def test_weighted_sum_calls_underlying_model_once(model, weights):
    call_count = 0

    class CountingMultiHeadEnergyModel(MultiHeadEnergyModel):
        def forward(
            self,
            systems: List[System],
            outputs: Dict[str, ModelOutput],
            selected_atoms: Optional[Labels] = None,
        ) -> Dict[str, TensorMap]:
            nonlocal call_count
            call_count += 1
            return super().forward(systems, outputs, selected_atoms)

    counting_model = AtomisticModel(
        CountingMultiHeadEnergyModel().eval(),
        ModelMetadata(),
        model.capabilities(),
    )
    wrapped = WeightedSum.wrap(counting_model, "energy", weights)
    system, _ = _system()

    call_count = 0
    _eval(
        wrapped,
        system,
        {
            "energy": ModelOutput(unit="eV", sample_kind="atom"),
            "energy/pbe": ModelOutput(unit="eV", sample_kind="atom"),
        },
    )
    assert call_count == 1


def test_weighted_sum_selected_atoms(model, weights):
    wrapped = WeightedSum.wrap(model, "energy", weights)
    system, _ = _system()

    one_atom = Labels(["system", "atom"], torch.tensor([[0, 1]], dtype=torch.int64))
    results = _eval(
        wrapped,
        system,
        {"energy": ModelOutput(unit="eV", sample_kind="atom")},
        selected_atoms=one_atom,
    )
    block = results["energy"].block()
    assert torch.equal(block.samples.values, one_atom.values)

    full_results = _eval(
        wrapped, system, {"energy": ModelOutput(unit="eV", sample_kind="atom")}
    )
    full_block = full_results["energy"].block()
    row = (full_block.samples.values == one_atom.values).all(dim=1)
    assert torch.allclose(block.values, full_block.values[row])


def test_weighted_sum_rejects_missing_head(model):
    with pytest.raises(
        ValueError,
        match="this model does not have a 'energy/pw92' output",
    ):
        WeightedSum.wrap(model, "energy", {"energy/pw92": 1.0})


def test_weighted_sum_rejects_output_name_conflict(model, weights):
    with pytest.raises(
        ValueError,
        match="this model already has an output named 'test::extra'",
    ):
        WeightedSum.wrap(model, "test::extra", weights)


def test_weighted_sum_rejects_empty_weights(model):
    with pytest.raises(ValueError, match="must contain at least one head"):
        WeightedSum.wrap(model, "energy", {})


def test_weighted_sum_normalize_coefficients(model):
    raw_weights = {"energy/pbe": 2.0, "energy/r2scan": 1.0, "energy/lda": 1.0}
    wrapped = WeightedSum.wrap(
        model, "energy", raw_weights, normalize_coefficients=True
    )
    system, _ = _system()

    results = _eval(
        wrapped, system, {"energy": ModelOutput(unit="eV", sample_kind="atom")}
    )
    normalized = results["energy"].block().values

    total = sum(raw_weights.values())
    raw = {name: ModelOutput(unit="eV", sample_kind="atom") for name in raw_weights}
    raw_results = _eval(model, system, raw)
    expected = (
        sum(w * raw_results[name].block().values for name, w in raw_weights.items())
        / total
    )

    assert torch.allclose(normalized, expected)


def test_weighted_sum_normalize_coefficients_with_negative_weight(model):
    """Normalization works with a negative coefficient, as long as the sum of
    all coefficients is not zero."""
    raw_weights = {"energy/pbe": 3.0, "energy/lda": -1.0}
    wrapped = WeightedSum.wrap(
        model, "energy", raw_weights, normalize_coefficients=True
    )
    system, _ = _system()

    results = _eval(
        wrapped, system, {"energy": ModelOutput(unit="eV", sample_kind="atom")}
    )
    normalized = results["energy"].block().values

    total = sum(raw_weights.values())
    raw = {name: ModelOutput(unit="eV", sample_kind="atom") for name in raw_weights}
    raw_results = _eval(model, system, raw)
    expected = (
        sum(w * raw_results[name].block().values for name, w in raw_weights.items())
        / total
    )

    assert torch.allclose(normalized, expected)


def test_weighted_sum_rejects_zero_sum_normalization(model):
    with pytest.raises(
        ValueError, match="the sum of `weights` is zero, they can not be normalized"
    ):
        WeightedSum.wrap(
            model,
            "energy",
            {"energy/pbe": 1.0, "energy/lda": -1.0},
            normalize_coefficients=True,
        )


def test_weighted_sum_normalize_negative_sum_warns_and_flips_signs(model):
    """A negative sum can still be normalized (dividing by a negative number is
    well-defined), but it flips the sign of every coefficient, which is
    surprising enough to warrant a warning."""
    raw_weights = {"energy/pbe": -1.5, "energy/lda": -0.5}
    with pytest.warns(UserWarning, match="flips the sign of every coefficient"):
        wrapped = WeightedSum.wrap(
            model, "energy", raw_weights, normalize_coefficients=True
        )
    system, _ = _system()

    results = _eval(
        wrapped, system, {"energy": ModelOutput(unit="eV", sample_kind="atom")}
    )
    normalized = results["energy"].block().values

    total = sum(raw_weights.values())
    raw = {name: ModelOutput(unit="eV", sample_kind="atom") for name in raw_weights}
    raw_results = _eval(model, system, raw)
    expected = (
        sum(w * raw_results[name].block().values for name, w in raw_weights.items())
        / total
    )
    assert torch.allclose(normalized, expected)


def test_weighted_sum_rejects_mismatched_sample_kind():
    mismatched = AtomisticModel(
        MultiHeadEnergyModel().eval(),
        ModelMetadata(),
        ModelCapabilities(
            outputs={
                "energy/pbe": ModelOutput(
                    sample_kind="atom", unit="eV", description="PBE energy head"
                ),
                "energy/lda": ModelOutput(
                    sample_kind="system", unit="eV", description="LDA energy head"
                ),
            },
            atomic_types=[ATOMIC_NUMBER],
            interaction_range=0.0,
            length_unit="Angstrom",
            supported_devices=["cpu"],
            dtype="float64",
        ),
    )
    with pytest.raises(ValueError, match="must share the same sample_kind"):
        WeightedSum.wrap(mismatched, "energy", {"energy/pbe": 0.5, "energy/lda": 0.5})


def test_weighted_sum_rejects_mismatched_unit():
    mismatched = AtomisticModel(
        MultiHeadEnergyModel().eval(),
        ModelMetadata(),
        ModelCapabilities(
            outputs={
                "energy/pbe": ModelOutput(
                    sample_kind="atom", unit="eV", description="PBE energy head"
                ),
                "energy/lda": ModelOutput(
                    sample_kind="atom",
                    unit="kcal/mol",
                    description="LDA energy head",
                ),
            },
            atomic_types=[ATOMIC_NUMBER],
            interaction_range=0.0,
            length_unit="Angstrom",
            supported_devices=["cpu"],
            dtype="float64",
        ),
    )
    with pytest.raises(ValueError, match="must share the same unit"):
        WeightedSum.wrap(mismatched, "energy", {"energy/pbe": 0.5, "energy/lda": 0.5})


def test_weighted_sum_rejects_explicit_gradients(model, weights):
    """WeightedSum itself rejects explicit gradients on its output as a defense
    in depth, on top of the outer AtomisticModel check (which normally rejects
    these first, since the "energy" output declares no explicit_gradients)."""
    wrapped = WeightedSum.wrap(model, "energy", weights)
    system, _ = _system()
    with pytest.raises(
        ValueError,
        match="WeightedSum does not support explicit gradients",
    ):
        wrapped.module(
            [system],
            {
                "energy": ModelOutput(
                    unit="eV", sample_kind="atom", explicit_gradients=["positions"]
                )
            },
            None,
        )


def test_weighted_sum_save_and_reload(tmp_path, model, weights):
    wrapped = WeightedSum.wrap(model, "energy", weights)
    system, _ = _system()

    outputs = {"energy": ModelOutput(unit="eV", sample_kind="atom")}
    original = _eval(wrapped, system, outputs)["energy"].block().values

    path = os.path.join(tmp_path, "weighted-sum.pt")
    wrapped.save(path)
    reloaded = load_atomistic_model(path)

    system, _ = _system()
    roundtrip = _eval(reloaded, system, outputs)["energy"].block().values

    assert torch.allclose(original, roundtrip)


def test_weighted_sum_rejects_non_atomistic_model(weights):
    with pytest.raises(TypeError, match="model must be an AtomisticModel"):
        WeightedSum.wrap(MultiHeadEnergyModel().eval(), "energy", weights)


class RequestingMultiHeadEnergyModel(MultiHeadEnergyModel):
    """Same heads as MultiHeadEnergyModel, but also requests a neighbor list and
    a custom input, to check that WeightedSum.wrap preserves them."""

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [NeighborListOptions(cutoff=5.0, full_list=True, strict=True)]

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        return {"mass": ModelOutput(unit="u", sample_kind="atom")}


def test_weighted_sum_preserves_requested_neighbor_lists_and_inputs():
    requesting_model = AtomisticModel(
        RequestingMultiHeadEnergyModel().eval(),
        ModelMetadata(),
        ModelCapabilities(
            outputs={
                "energy/pbe": ModelOutput(
                    sample_kind="atom", unit="eV", description="PBE energy head"
                ),
                "energy/lda": ModelOutput(
                    sample_kind="atom", unit="eV", description="LDA energy head"
                ),
            },
            atomic_types=[ATOMIC_NUMBER],
            interaction_range=5.0,
            length_unit="Angstrom",
            supported_devices=["cpu"],
            dtype="float64",
        ),
    )

    wrapped = WeightedSum.wrap(
        requesting_model, "energy", {"energy/pbe": 0.5, "energy/lda": 0.5}
    )

    neighbor_lists = wrapped.requested_neighbor_lists()
    assert len(neighbor_lists) == 1
    assert neighbor_lists[0].cutoff == 5.0
    assert neighbor_lists[0].full_list is True
    assert neighbor_lists[0].strict is True

    inputs = wrapped.requested_inputs(use_new_names=True)
    assert set(inputs.keys()) == {"mass"}
    assert inputs["mass"].unit == "u"


class DropHeadModel(torch.nn.Module):
    """Declares two heads in its capabilities but only ever returns one of them,
    simulating a broken/incomplete underlying model implementation."""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        results = torch.jit.annotate(Dict[str, TensorMap], {})
        keys = Labels("_", torch.tensor([[0]], dtype=torch.int64))
        properties = Labels("energy", torch.tensor([[0]], dtype=torch.int64))
        for name in outputs:
            if name != "energy/pbe":
                continue
            n_atoms = systems[0].positions.shape[0]
            block = TensorBlock(
                values=torch.zeros((n_atoms, 1), dtype=torch.float64),
                samples=Labels(
                    ["system", "atom"],
                    torch.stack(
                        [
                            torch.zeros(n_atoms, dtype=torch.int64),
                            torch.arange(n_atoms, dtype=torch.int64),
                        ],
                        dim=1,
                    ),
                ),
                components=torch.jit.annotate(List[Labels], []),
                properties=properties,
            )
            results[name] = TensorMap(keys, [block])
        return results


def test_weighted_sum_rejects_underlying_model_missing_head():
    broken_model = AtomisticModel(
        DropHeadModel().eval(),
        ModelMetadata(),
        ModelCapabilities(
            outputs={
                "energy/pbe": ModelOutput(
                    sample_kind="atom", unit="eV", description="PBE energy head"
                ),
                "energy/lda": ModelOutput(
                    sample_kind="atom", unit="eV", description="LDA energy head"
                ),
            },
            atomic_types=[ATOMIC_NUMBER],
            interaction_range=0.0,
            length_unit="Angstrom",
            supported_devices=["cpu"],
            dtype="float64",
        ),
    )
    wrapped = WeightedSum.wrap(
        broken_model, "energy", {"energy/pbe": 0.5, "energy/lda": 0.5}
    )
    system, _ = _system()

    with pytest.raises(
        ValueError,
        match="underlying model did not return the requested head 'energy/lda'",
    ):
        _eval(wrapped, system, {"energy": ModelOutput(unit="eV", sample_kind="atom")})


class EnergyAndNonConservativeModel(torch.nn.Module):
    """Exposes conservative energy heads *and* independent non-conservative
    force/stress heads (i.e. not the gradient of the energy heads) for two
    variants, "pbe" and "lda", to check that WeightedSum works just as well
    for these direct, non-autograd-derived outputs."""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        results = torch.jit.annotate(Dict[str, TensorMap], {})
        system = systems[0]
        n_atoms = system.positions.shape[0]
        keys = Labels("_", torch.tensor([[0]], dtype=torch.int64))
        atom_samples = Labels(
            ["system", "atom"],
            torch.stack(
                [
                    torch.zeros(n_atoms, dtype=torch.int64),
                    torch.arange(n_atoms, dtype=torch.int64),
                ],
                dim=1,
            ),
        )
        system_samples = Labels("system", torch.tensor([[0]], dtype=torch.int64))

        for name in outputs:
            coeff = 0.0
            if name.endswith("/pbe"):
                coeff = 1.0
            elif name.endswith("/lda"):
                coeff = 0.5
            else:
                continue

            if name.startswith("energy/"):
                values = coeff * (system.positions**2).sum(dim=1, keepdim=True)
                block = TensorBlock(
                    values=values,
                    samples=atom_samples,
                    components=torch.jit.annotate(List[Labels], []),
                    properties=Labels("energy", torch.tensor([[0]])),
                )
                results[name] = TensorMap(keys, [block])

            elif name.startswith("non_conservative_force/"):
                # deliberately unrelated to -d(energy)/d(positions)
                values = coeff * torch.sin(system.positions) * 3.0
                block = TensorBlock(
                    values=values.unsqueeze(-1),
                    samples=atom_samples,
                    components=[Labels("xyz", torch.arange(3).reshape(-1, 1))],
                    properties=Labels("non_conservative_force", torch.tensor([[0]])),
                )
                results[name] = TensorMap(keys, [block])

            elif name.startswith("non_conservative_stress/"):
                values = coeff * torch.eye(3, dtype=system.positions.dtype) * 7.0
                block = TensorBlock(
                    values=values.reshape(1, 3, 3, 1),
                    samples=system_samples,
                    components=[
                        Labels("xyz_1", torch.arange(3).reshape(-1, 1)),
                        Labels("xyz_2", torch.arange(3).reshape(-1, 1)),
                    ],
                    properties=Labels("non_conservative_stress", torch.tensor([[0]])),
                )
                results[name] = TensorMap(keys, [block])

        return results


@pytest.fixture
def nc_model():
    return AtomisticModel(
        EnergyAndNonConservativeModel().eval(),
        ModelMetadata(),
        ModelCapabilities(
            outputs={
                "energy/pbe": ModelOutput(
                    sample_kind="atom", unit="eV", description="pbe energy"
                ),
                "energy/lda": ModelOutput(
                    sample_kind="atom", unit="eV", description="lda energy"
                ),
                "non_conservative_force/pbe": ModelOutput(
                    sample_kind="atom",
                    unit="eV/Angstrom",
                    description="pbe non-conservative force",
                ),
                "non_conservative_force/lda": ModelOutput(
                    sample_kind="atom",
                    unit="eV/Angstrom",
                    description="lda non-conservative force",
                ),
                "non_conservative_stress/pbe": ModelOutput(
                    sample_kind="system",
                    unit="eV/Angstrom^3",
                    description="pbe non-conservative stress",
                ),
                "non_conservative_stress/lda": ModelOutput(
                    sample_kind="system",
                    unit="eV/Angstrom^3",
                    description="lda non-conservative stress",
                ),
            },
            atomic_types=[ATOMIC_NUMBER],
            interaction_range=0.0,
            length_unit="Angstrom",
            supported_devices=["cpu"],
            dtype="float64",
        ),
    )


def test_weighted_sum_chained_energy_and_non_conservative_heads(nc_model):
    """WeightedSum does not care what an output means physically: it works the
    same way for non_conservative_force/non_conservative_stress heads (direct,
    non-autograd-derived model outputs) as for energy heads. Combining several
    output "kinds" with the same coefficients requires chaining one wrap() call
    per output name, each wrapping the AtomisticModel returned by the previous
    one."""
    variant_weights = {"pbe": 0.6, "lda": 0.4}

    wrapped = WeightedSum.wrap(
        nc_model,
        "energy",
        {f"energy/{v}": w for v, w in variant_weights.items()},
    )
    wrapped = WeightedSum.wrap(
        wrapped,
        "non_conservative_force",
        {f"non_conservative_force/{v}": w for v, w in variant_weights.items()},
    )
    wrapped = WeightedSum.wrap(
        wrapped,
        "non_conservative_stress",
        {f"non_conservative_stress/{v}": w for v, w in variant_weights.items()},
    )

    for name in ["energy", "non_conservative_force", "non_conservative_stress"]:
        assert name in wrapped.capabilities().outputs

    system, _ = _system()
    results = _eval(
        wrapped,
        system,
        {
            "energy": ModelOutput(unit="eV", sample_kind="atom"),
            "non_conservative_force": ModelOutput(
                unit="eV/Angstrom", sample_kind="atom"
            ),
            "non_conservative_stress": ModelOutput(
                unit="eV/Angstrom^3", sample_kind="system"
            ),
        },
    )

    raw = _eval(
        nc_model,
        system,
        {
            f"{quantity}/{variant}": ModelOutput(sample_kind=sample_kind)
            for quantity, sample_kind in [
                ("energy", "atom"),
                ("non_conservative_force", "atom"),
                ("non_conservative_stress", "system"),
            ]
            for variant in variant_weights
        },
    )

    for quantity in ["energy", "non_conservative_force", "non_conservative_stress"]:
        expected = sum(
            w * raw[f"{quantity}/{v}"].block().values
            for v, w in variant_weights.items()
        )
        assert torch.allclose(results[quantity].block().values, expected)

    # the chained (3-layer-deep) wrapper must still TorchScript-compile, exactly
    # as AtomisticModel.save() would do
    torch.jit.script(wrapped)
