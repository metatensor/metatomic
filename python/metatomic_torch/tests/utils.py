import os

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
