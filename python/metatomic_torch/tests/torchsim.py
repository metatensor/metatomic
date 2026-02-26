"""Tests for SevenNetModel torchsim integration.

This test file is structured to work in the torchsim repository's test directory.
It uses the factory function pattern from torchsim's test infrastructure.
"""

from metatomic.torch.ase_calculator import MetatomicCalculator


import traceback

import pytest
import torch
from metatomic.torch import ase_calculator
from metatrain.utils.io import load_model
from metatomic.torch.torchsim import MetatomicModel

try:
    from torch_sim.models.interface import validate_model_outputs
    from torch_sim.testing import (
        SIMSTATE_GENERATORS,
        assert_model_calculator_consistency,
    )

    TORCH_SIM_AVAILABLE = True
except ImportError:
    TORCH_SIM_AVAILABLE = False  # pyright: ignore[reportConstantRedefinition]
    SIMSTATE_GENERATORS = {}  # pyright: ignore[reportConstantRedefinition]

    def validate_model_outputs(*args, **kwargs):
        return None

    def assert_model_calculator_consistency(*args, **kwargs):
        return None

if not TORCH_SIM_AVAILABLE:
    pytest.skip(
        'torch_sim not installed. Install torch-sim-atomistic separately if needed.',
        allow_module_level=True,
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DTYPE = torch.float32

@pytest.fixture
def metatomic_calculator() -> MetatomicCalculator:
    """Load a pretrained metatomic model for testing."""
    model_url = "https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.1.0/models/pet-mad-v1.1.0.ckpt"
    return ase_calculator.MetatomicCalculator(
        model=load_model(model_url).export(), device=DEVICE
    )


@pytest.fixture
def metatomic_model() -> MetatomicModel:
    """Create an MetatomicModel wrapper for the pretrained model."""
    return MetatomicModel(model="pet-mad", device=DEVICE)


def test_metatomic_initialization() -> None:
    """Test that the metatomic model initializes correctly."""
    model = MetatomicModel(
        model="pet-mad",
        device=DEVICE,
    )
    assert model.device == DEVICE
    assert model.dtype == torch.float32


def test_metatomic_model_output_validation(metatomic_model: MetatomicModel) -> None:
    """Test that a model implementation follows the ModelInterface contract."""
    validate_model_outputs(metatomic_model, DEVICE, DTYPE)


@pytest.mark.parametrize('sim_state_name', SIMSTATE_GENERATORS)
def test_metatomic_model_consistency(
    sim_state_name: str,
    metatomic_model: MetatomicModel,
    metatomic_calculator: MetatomicCalculator,
) -> None:
    """Test consistency between SevenNetModel and SevenNetCalculator.

    NOTE: sevenn is broken for the benzene simstate is ase comparison."""
    sim_state = SIMSTATE_GENERATORS[sim_state_name](DEVICE, DTYPE)
    assert_model_calculator_consistency(metatomic_model, metatomic_calculator, sim_state)
