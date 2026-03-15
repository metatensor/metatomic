import os

import pytest

import metatomic.torch
from metatomic.torch.utils import version_compatible


def test_cmake_prefix_path():
    assert os.path.exists(metatomic.torch.utils.cmake_prefix_path)


def test_version_compatible():
    # same major.minor is compatible regardless of patch
    assert version_compatible("1.2.3", "1.2.0") is True
    # different minor versions are incompatible
    assert version_compatible("1.2.3", "1.3.0") is False
    # different major versions are incompatible
    assert version_compatible("1.2.3", "2.2.0") is False


def test_lazy_ase_calculator_import():
    mod = metatomic.torch.ase_calculator
    assert hasattr(mod, "MetatomicCalculator")


def test_lazy_import_missing_attribute():
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = metatomic.torch.nonexistent_attribute
