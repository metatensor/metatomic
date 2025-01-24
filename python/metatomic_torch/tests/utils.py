import os

import metatomic.torch


def test_cmake_prefix_path():
    assert os.path.exists(metatomic.torch.utils.cmake_prefix_path)
