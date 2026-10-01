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
    System,
    load_atomistic_model,
)
from metatomic.torch.weighted_sum import WeightedSum


class MultiHeadEnergyModel(torch.nn.Module):
    """Toy model exposing three energy variants, each a different (nonlinear)
    function of the atomic positions so that per-variant forces/stresses differ."""

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
            if name == "energy":
                coeff = 10.0
            elif name == "energy/pbe":
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


def model_output(unit: str, sample_kind: str) -> ModelOutput:
    return ModelOutput(
        unit=unit,
        sample_kind=sample_kind,
        explicit_gradients=[],
        description="not empty",
    )


def eval_options(outputs: Dict[str, ModelOutput]) -> ModelEvaluationOptions:
    return ModelEvaluationOptions(
        length_unit="Angstrom",
        outputs=outputs,
        selected_atoms=None,
    )


@pytest.fixture
def model():
    return AtomisticModel(
        MultiHeadEnergyModel().eval(),
        ModelMetadata(),
        ModelCapabilities(
            outputs={
                "energy": model_output(unit="eV", sample_kind="atom"),
                "energy/pbe": model_output(unit="eV", sample_kind="atom"),
                "energy/r2scan": model_output(unit="eV", sample_kind="atom"),
                "energy/lda": model_output(unit="eV", sample_kind="atom"),
                "test::extra": model_output(sample_kind="atom", unit=""),
            },
            atomic_types=[6],
            interaction_range=0.0,
            length_unit="Angstrom",
            supported_devices=["cpu", "cuda"],
            dtype="float64",
        ),
    )


def get_system(with_strain=False):
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
        types=torch.full((3,), 6, dtype=torch.int32),
        positions=positions,
        cell=cell,
        pbc=torch.tensor([True, True, True]),
    )
    return system, strain


def test_wrap_capabilities(model):
    weights = {"energy/pbe": 0.6, "energy/r2scan": 0.3, "energy/lda": 0.1}
    wrapped = WeightedSum.wrap(model, "energy/mix", weights)
    capabilities = wrapped.capabilities()

    assert "energy/mix" in capabilities.outputs
    new_output = capabilities.outputs["energy/mix"]
    assert new_output.unit == "eV"
    assert new_output.sample_kind == "atom"
    for variant in weights:
        assert variant in new_output.description

    # the original outputs are all preserved, including the main `energy` one
    for name in ["energy", "energy/pbe", "energy/r2scan", "energy/lda", "test::extra"]:
        assert name in capabilities.outputs


def test_values_match_manual_combination(model):
    weights = {"energy/pbe": 0.6, "energy/r2scan": 0.3, "energy/lda": 0.1}
    wrapped = WeightedSum.wrap(model, "energy/mix", weights)
    system, _ = get_system()

    options = eval_options({"energy/mix": ModelOutput(unit="eV", sample_kind="atom")})
    results = wrapped([system], options, check_consistency=True)
    combined = results["energy/mix"].block().values

    options = eval_options(
        {name: ModelOutput(unit="eV", sample_kind="atom") for name in weights}
    )
    raw = model([system], options, check_consistency=True)
    expected = sum(w * raw[name].block().values for name, w in weights.items())

    assert torch.allclose(combined, expected)


def test_forces_and_stress_match_reference(model):
    """A single backward pass through the weighted-sum output must give forces
    and stresses equal to the weighted sum of the forces/stresses of the
    individual variants."""
    weights = {"energy/pbe": 0.6, "energy/r2scan": 0.3, "energy/lda": 0.1}
    wrapped = WeightedSum.wrap(model, "energy/mix", weights)
    system, strain = get_system(with_strain=True)

    options = eval_options({"energy/mix": ModelOutput(unit="eV", sample_kind="atom")})
    results = wrapped([system], options, check_consistency=True)

    total_energy = results["energy/mix"].block().values.sum()
    total_energy.backward()
    combined_forces = -system.positions.grad.clone()
    combined_stress = strain.grad.clone()

    expected_forces = torch.zeros_like(combined_forces)
    expected_stress = torch.zeros_like(combined_stress)
    for name, weight in weights.items():
        system, strain = get_system(with_strain=True)

        options = eval_options({name: ModelOutput(unit="eV", sample_kind="atom")})
        results = model([system], options, check_consistency=True)

        energy = results[name].block().values.sum()
        energy.backward()
        expected_forces += weight * (-system.positions.grad)
        expected_stress += weight * strain.grad

    np.testing.assert_allclose(
        combined_forces.detach().numpy(), expected_forces.detach().numpy()
    )
    np.testing.assert_allclose(
        combined_stress.detach().numpy(), expected_stress.detach().numpy()
    )
    # sanity check: the variants are not all identical, so this is a real test of
    # the weighted combination and not a coincidence
    assert not torch.allclose(expected_forces, torch.zeros_like(expected_forces))


def test_errors(model):
    message = (
        "this model does not have a 'energy/pw92' output, "
        "which is required to compute the 'energy/mix' weighted sum"
    )
    with pytest.raises(ValueError, match=message):
        WeightedSum.wrap(model, "energy/mix", {"energy/pw92": 1.0})

    message = (
        "this model already has an output named 'test::extra', "
        "which conflicts with the weighted sum output"
    )
    with pytest.raises(ValueError, match=message):
        WeightedSum.wrap(model, "test::extra", {"energy/pbe": 1.0})

    message = (
        "this model already has an output named 'energy', "
        "which conflicts with the weighted sum output"
    )
    with pytest.raises(ValueError, match=message):
        WeightedSum.wrap(model, "energy", {"energy/pbe": 1.0})

    message = "`weights` must contain at least one output name to combine"
    with pytest.raises(ValueError, match=message):
        WeightedSum.wrap(model, "energy/mix", {})


def test_normalize_coefficients(model):
    def check_normalization(model, wrapped, weights):
        system, _ = get_system()

        options = eval_options(
            {"energy/mix": ModelOutput(unit="eV", sample_kind="atom")}
        )
        results = wrapped([system], options, check_consistency=True)
        normalized = results["energy/mix"].block().values

        total = sum(weights.values())

        options = eval_options(
            {name: ModelOutput(unit="eV", sample_kind="atom") for name in weights}
        )
        raw_results = model([system], options, check_consistency=True)
        expected = (
            sum(w * raw_results[name].block().values for name, w in weights.items())
            / total
        )

        assert torch.allclose(normalized, expected)

    weights = {"energy/pbe": 2.0, "energy/r2scan": 1.0, "energy/lda": 1.0}
    wrapped = WeightedSum.wrap(
        model, "energy/mix", weights, normalize_coefficients=True
    )

    check_normalization(model, wrapped, weights)

    weights = {"energy/pbe": 3.0, "energy/lda": -1.0}
    wrapped = WeightedSum.wrap(
        model, "energy/mix", weights, normalize_coefficients=True
    )
    check_normalization(model, wrapped, weights)

    message = "the sum of `weights` is too close to zero, they can not be normalized"
    with pytest.raises(ValueError, match=message):
        WeightedSum.wrap(
            model,
            "energy/mix",
            {"energy/pbe": 1.0, "energy/lda": -1.0},
            normalize_coefficients=True,
        )


def test_rejects_mismatched_sample_kind():
    mismatched = AtomisticModel(
        MultiHeadEnergyModel().eval(),
        ModelMetadata(),
        ModelCapabilities(
            outputs={
                "energy/pbe": model_output(sample_kind="atom", unit="eV"),
                "energy/lda": model_output(sample_kind="system", unit="eV"),
            },
            atomic_types=[6],
            interaction_range=0.0,
            length_unit="Angstrom",
            supported_devices=["cpu"],
            dtype="float64",
        ),
    )

    message = (
        "all variants combined in a weighted sum must share the same sample_kind; "
        "got 'atom' and 'system'"
    )
    with pytest.raises(ValueError, match=message):
        WeightedSum.wrap(mismatched, "energy", {"energy/pbe": 0.5, "energy/lda": 0.5})


def test_rejects_mismatched_unit():
    mismatched = AtomisticModel(
        MultiHeadEnergyModel().eval(),
        ModelMetadata(),
        ModelCapabilities(
            outputs={
                "energy/pbe": model_output(sample_kind="atom", unit="eV"),
                "energy/lda": model_output(sample_kind="atom", unit="kcal/mol"),
            },
            atomic_types=[6],
            interaction_range=0.0,
            length_unit="Angstrom",
            supported_devices=["cpu"],
            dtype="float64",
        ),
    )

    message = (
        "all variants combined in a weighted sum must share the same unit; "
        "got 'eV' and 'kcal/mol'"
    )
    with pytest.raises(ValueError, match=message):
        WeightedSum.wrap(mismatched, "energy", {"energy/pbe": 0.5, "energy/lda": 0.5})


def test_save_and_reload(tmp_path, model):
    weights = {"energy/pbe": 0.6, "energy/r2scan": 0.3, "energy/lda": 0.1}
    wrapped = WeightedSum.wrap(model, "energy/mix", weights)
    system, _ = get_system()

    options = eval_options({"energy/mix": ModelOutput(unit="eV", sample_kind="atom")})
    original = wrapped([system], options, check_consistency=True)
    original = original["energy/mix"].block().values

    path = os.path.join(tmp_path, "weighted-sum.pt")
    wrapped.save(path)
    reloaded = load_atomistic_model(path)

    system, _ = get_system()
    roundtrip = reloaded([system], options, check_consistency=True)
    roundtrip = roundtrip["energy/mix"].block().values

    assert torch.allclose(original, roundtrip)
