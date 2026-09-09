import os

import metatomic as mta


def test_cmake_prefix_path():
    assert os.path.exists(mta.utils.cmake_prefix_path)


def test_library_loading():
    # temporary test to be removed as soon as some other test actually load the library
    import metatomic._c_lib  # noqa: F401

    lib = mta._c_lib._get_library()
    assert lib.mta_version().decode("utf8").replace("-", ".") == mta.__version__
