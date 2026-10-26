import os

import metatomic_torch


def test_cmake_prefix_path():
    assert os.path.exists(metatomic_torch.utils.cmake_prefix_path)
