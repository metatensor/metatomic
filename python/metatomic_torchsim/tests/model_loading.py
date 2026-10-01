"""Tests for MetatomicModel loading paths."""

import pytest
import torch

import metatomic_lj_test
from metatomic_torchsim import MetatomicModel


DEVICE = torch.device("cpu")


@pytest.fixture
def lj_model():
    return metatomic_lj_test.lennard_jones_model(
        atomic_type=28,
        cutoff=5.0,
        sigma=1.5808,
        epsilon=0.1729,
        length_unit="Angstrom",
        energy_unit="eV",
        with_extension=False,
    )


def test_load_from_pt_file(lj_model, tmp_path):
    """Model loads from a saved .pt file."""
    pt_path = tmp_path / "test_model.pt"
    lj_model.save(str(pt_path))

    model = MetatomicModel(model=str(pt_path), device=DEVICE)
    assert model.device == DEVICE


def test_nonexistent_path_raises_valueerror():
    """ValueError raised for a path that does not exist."""
    message = "given model path '/non/existent/path.pt' does not exist"
    with pytest.raises(ValueError, match=message):
        MetatomicModel(model="/non/existent/path.pt", device=DEVICE)


def test_wrong_model_type_raises_typeerror():
    """TypeError raised when passing an unsupported type."""
    message = "unknown type for model: <class 'int'>"
    with pytest.raises(TypeError, match=message):
        MetatomicModel(model=42, device=DEVICE)


def test_non_atomisticmodel_scriptmodule_raises_typeerror():
    """TypeError raised for a ScriptModule that is not AtomisticModel."""

    class Dummy(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    dummy_scripted = torch.jit.script(Dummy())
    message = "torch model must be 'AtomisticModel', got 'Dummy' instead"
    with pytest.raises(TypeError, match=message):
        MetatomicModel(model=dummy_scripted, device=DEVICE)
