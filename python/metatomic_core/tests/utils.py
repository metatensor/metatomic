import os

import pytest

import metatomic as mta


def test_unit_conversion_factor():
    assert mta.unit_conversion_factor("nm", "angstrom") == pytest.approx(10)
    assert mta.unit_conversion_factor("", "angstrom") == 1
    message = "dimension mismatch in unit conversion"
    with pytest.raises(mta.MetatomicError, match=message):
        mta.unit_conversion_factor("eV", "angstrom")
    message = "unknown unit 'unknown'"
    with pytest.raises(mta.MetatomicError, match=message):
        mta.unit_conversion_factor("unknown", "angstrom")


def test_cmake_prefix_path():
    assert os.path.exists(mta.utils.cmake_prefix_path)


def test_library_loading():
    # temporary test to be removed as soon as some other test actually load the library
    import metatomic._c_lib  # noqa: F401

    lib = mta._c_lib._get_library()
    assert lib.mta_version().decode("utf8").replace("-", ".") == mta.__version__


def test_plugin_path():
    path = mta.testing.plugin_path("lj")
    assert os.path.isfile(path)
    assert os.path.basename(path) == "metatomic-lj-plugin.so"
    assert os.path.dirname(path) == mta.testing.plugins_directory()

    message = "Unknown plugin name: 'not-a-plugin'"
    with pytest.raises(ValueError, match=message):
        mta.testing.plugin_path("not-a-plugin")
