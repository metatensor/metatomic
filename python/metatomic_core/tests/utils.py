import os

import pytest

import metatomic as mta


def test_cmake_prefix_path():
    assert os.path.exists(mta.utils.cmake_prefix_path)


def test_library_loading():
    # temporary test to be removed as soon as some other test actually load the library
    import metatomic._c_lib  # noqa: F401

    lib = mta._c_lib._get_library()
    assert lib.mta_version().decode("utf8").replace("-", ".") == mta.__version__


def test_plugin_path():
    path = mta.testing.plugin_path("lj-plugin")
    assert os.path.isfile(path)
    assert os.path.basename(path) == "lj-plugin.so"
    assert os.path.dirname(path) == mta.testing.plugins_directory()
    assert path.startswith(mta.utils._installation_prefix)


def test_plugin_path_missing():
    with pytest.raises(FileNotFoundError, match="not-a-plugin"):
        mta.testing.plugin_path("not-a-plugin")
