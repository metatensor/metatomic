import pytest


def test_import():
    message = (
        "Importing MetatomicCalculator from metatomic.torch.ase_calculator is "
        "deprecated and will be removed in a future release. Please import from "
        "metatomic_ase instead."
    )
    with pytest.warns(DeprecationWarning, match=message):
        from metatomic.torch.ase_calculator import MetatomicCalculator  # noqa: F401

    message = (
        "Importing SymmetrizedCalculator from metatomic.torch.ase_calculator is "
        "deprecated and will be removed in a future release. Please import from "
        "metatomic_ase instead."
    )
    with pytest.warns(DeprecationWarning, match=message):
        from metatomic.torch.ase_calculator import SymmetrizedCalculator  # noqa: F401
