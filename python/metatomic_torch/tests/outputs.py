from typing import Dict, List, Optional

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
)


def test_sample_kind():
    """Checks that ``sample_kind`` and ``per_atom`` are always
    consistent with each other.

    It also checks some other expected behaviors of the ModelOutput class.
    """
    # Initialize model output with defaults
    output = ModelOutput()
    assert output.per_atom is False
    assert output.sample_kind == "system"

    # Set per_atom to True and check that
    # sample_kind is updated accordingly
    output.per_atom = True
    assert output.per_atom is True
    assert output.sample_kind == "atom"

    # Set sample_kind back to "system" and check that
    # per_atom is updated accordingly
    output.sample_kind = "system"
    assert output.per_atom is False
    assert output.sample_kind == "system"

    # Initialize model output with per_atom=True and check that
    # sample_kind is set to "atom"
    output = ModelOutput(per_atom=True)
    assert output.per_atom is True
    assert output.sample_kind == "atom"

    # Initialize model output with sample_kind="atom"
    # and check that per_atom is set to True
    output = ModelOutput(sample_kind="atom")
    assert output.per_atom is True
    assert output.sample_kind == "atom"

    # Check that trying to set both per_atom and sample_kind raises an error
    with pytest.raises(ValueError):
        ModelOutput(per_atom=True, sample_kind="system")

    # Check that setting sample_kind to an invalid value raises an error
    with pytest.raises(ValueError):
        ModelOutput(sample_kind="invalid_value")

    # Initialize model output with sample_kind="atom_pair"
    # and check that per_atom can not be retrieved
    output = ModelOutput(sample_kind="atom_pair")
    assert output.sample_kind == "atom_pair"
    with pytest.raises(ValueError):
        _ = output.per_atom


@pytest.fixture
def system():
    return System(
        types=torch.tensor([1, 2, 3]),
        positions=torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=torch.float64),
        cell=torch.zeros([3, 3], dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )


@pytest.fixture
def get_capabilities() -> callable:
    def _create_capabilities(output_name: str) -> ModelCapabilities:
        return ModelCapabilities(
            length_unit="angstrom",
            atomic_types=[1, 2, 3],
            interaction_range=4.3,
            outputs={output_name: ModelOutput(per_atom=False)},
            supported_devices=["cpu"],
            dtype="float64",
        )

    return _create_capabilities


class BaseAtomisticModel(torch.nn.Module):
    """Base class for atomistic models"""

    def __init__(self, output_name: str):
        super().__init__()
        self.output_name = output_name

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        assert self.output_name in outputs
        assert not outputs[self.output_name].per_atom
        assert selected_atoms is None

        block = TensorBlock(
            values=torch.tensor(
                [
                    (
                        [0.0]
                        if self.output_name == "energy_uncertainty"
                        else [0.0, 1.0, 2.0]
                    )
                ]
                * len(systems),
                dtype=torch.float64,
            ),
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels(
                "energy",
                torch.tensor(
                    (
                        [[0]]
                        if self.output_name == "energy_uncertainty"
                        else [[0], [1], [2]]
                    )
                ),
            ),
        )
        return {self.output_name: TensorMap(Labels("_", torch.tensor([[0]])), [block])}


class EnergyEnsembleModel(BaseAtomisticModel):
    """An atomistic model returning an energy ensemble"""

    def __init__(self):
        super().__init__("energy_ensemble")


class EnergyUncertaintyModel(BaseAtomisticModel):
    """An atomistic model returning an energy ensemble"""

    def __init__(self):
        super().__init__("energy_uncertainty")


class FeaturesModel(BaseAtomisticModel):
    """An atomistic model returning features"""

    def __init__(self):
        super().__init__("features")


class PositionsMomentaModel(torch.nn.Module):
    """A model predicting positions and momenta"""

    def __init__(self):
        super().__init__()
        self.output_names = ["positions", "momenta"]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        assert "positions" in outputs
        assert "momenta" in outputs
        assert outputs["positions"].per_atom
        assert outputs["momenta"].per_atom
        assert selected_atoms is None

        sample_values = torch.stack(
            [
                torch.concatenate(
                    [
                        torch.full(
                            (len(system),),
                            i_system,
                        )
                        for i_system, system in enumerate(systems)
                    ],
                ),
                torch.concatenate(
                    [
                        torch.arange(
                            len(system),
                        )
                        for system in systems
                    ],
                ),
            ],
            dim=1,
        )
        samples = Labels(
            names=["system", "atom"],
            values=sample_values,
        )

        blocks = []
        for output_name in self.output_names:
            block = TensorBlock(
                values=torch.tensor(
                    [[[0.0], [1.0], [2.0]]] * sum(len(system) for system in systems),
                    dtype=torch.float64,
                ),
                samples=samples,
                components=[Labels("xyz", torch.tensor([[0], [1], [2]]))],
                properties=Labels(
                    output_name,
                    torch.tensor([[0]]),
                ),
            )
            blocks.append(block)

        return {
            output_name: TensorMap(Labels("_", torch.tensor([[0]])), [block])
            for output_name, block in zip(self.output_names, blocks, strict=False)
        }


def test_energy_ensemble_model(system, get_capabilities):
    model = EnergyEnsembleModel()
    capabilities = get_capabilities("energy_ensemble")
    atomistic = AtomisticModel(model.eval(), ModelMetadata(), capabilities)

    options = ModelEvaluationOptions(
        outputs={"energy_ensemble": ModelOutput(per_atom=False)}
    )

    result = atomistic([system, system], options, check_consistency=True)
    assert "energy_ensemble" in result

    ensemble = result["energy_ensemble"]

    assert ensemble.keys == Labels("_", torch.tensor([[0]]))
    assert list(ensemble.block().values.shape) == [2, 3]
    assert ensemble.block().samples.names == ["system"]
    assert ensemble.block().properties.names == ["energy"]


def test_energy_uncertainty_model(system, get_capabilities):
    model = EnergyUncertaintyModel()
    capabilities = get_capabilities("energy_uncertainty")
    atomistic = AtomisticModel(model.eval(), ModelMetadata(), capabilities)

    options = ModelEvaluationOptions(
        outputs={"energy_uncertainty": ModelOutput(per_atom=False)}
    )

    result = atomistic([system, system], options, check_consistency=True)
    assert "energy_uncertainty" in result

    uncertainty = result["energy_uncertainty"]
    assert uncertainty.keys == Labels("_", torch.tensor([[0]]))
    assert list(uncertainty.block().values.shape) == [2, 1]
    assert uncertainty.block().samples.names == ["system"]
    assert uncertainty.block().properties.names == ["energy"]


def test_features_model(system, get_capabilities):
    model = FeaturesModel()
    capabilities = get_capabilities("features")
    atomistic = AtomisticModel(model.eval(), ModelMetadata(), capabilities)

    options = ModelEvaluationOptions(outputs={"features": ModelOutput(per_atom=False)})

    result = atomistic([system, system], options, check_consistency=True)
    assert "features" in result

    features = result["features"]
    assert features.keys == Labels("_", torch.tensor([[0]]))
    assert list(features.block().values.shape) == [2, 3]
    assert features.block().samples.names == ["system"]
    assert features.block().properties.names == ["energy"]
    assert features.block().components == []
    assert len(result["features"].blocks()) == 1


def test_positions_momenta_model(system):
    model = PositionsMomentaModel()
    outputs = {
        "positions": ModelOutput(per_atom=True),
        "momenta": ModelOutput(per_atom=True),
    }
    capabilities = ModelCapabilities(
        length_unit="angstrom",
        atomic_types=[1, 2, 3],
        interaction_range=4.3,
        outputs=outputs,
        supported_devices=["cpu"],
        dtype="float64",
    )
    atomistic = AtomisticModel(model.eval(), ModelMetadata(), capabilities)

    options = ModelEvaluationOptions(outputs=outputs)

    result = atomistic([system, system], options, check_consistency=True)
    assert "positions" in result
    assert "momenta" in result

    positions = result["positions"]
    assert positions.keys == Labels("_", torch.tensor([[0]]))
    assert list(positions.block().values.shape) == [6, 3, 1]
    assert positions.block().samples.names == ["system", "atom"]
    assert positions.block().properties.names == ["positions"]
    assert positions.block().components == [
        Labels("xyz", torch.tensor([[0], [1], [2]]))
    ]
    assert len(result["positions"].blocks()) == 1

    momenta = result["momenta"]
    assert momenta.keys == Labels("_", torch.tensor([[0]]))
    assert list(momenta.block().values.shape) == [6, 3, 1]
    assert momenta.block().samples.names == ["system", "atom"]
    assert momenta.block().properties.names == ["momenta"]
    assert momenta.block().components == [Labels("xyz", torch.tensor([[0], [1], [2]]))]
    assert len(result["momenta"].blocks()) == 1
