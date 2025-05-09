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
