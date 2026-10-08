import os
import pathlib

import metatensor
import pytest

import metatomic as mta
from metatomic import MetatomicError, load_model, load_plugin


PLUGIN = "metatomic-lj-plugin"
MODEL = "metatomic-lj-model"

OPTIONS = {
    "sigma": "1",
    "epsilon": "1",
    "atomic_type": "1",
    "cutoff": "3.0",
    "length_unit": "A",
    "energy_unit": "kJ/mol",
}


@pytest.fixture(scope="module", autouse=True)
def lj_plugin():
    load_plugin(mta.testing.plugin_path("lj"))

    # Register a data wrapper for C++ arrays, this should be moved to metatensor
    metatensor.register_external_data_wrapper(
        "metatensor::SimpleDataArray", metatensor.ExternalCpuArray
    )


def test_load_plugin_multiple_times(tmp_path):
    path = mta.testing.plugin_path("lj")

    # loading the same file again does nothing, even through a different path
    load_plugin(path)
    load_plugin(pathlib.Path(path))
    load_plugin(os.path.relpath(path))

    symlink = tmp_path / "symlink.so"
    symlink.symlink_to(path)
    load_plugin(symlink)

    # the plugin is still usable
    load_model(MODEL, OPTIONS, plugin_name=PLUGIN)


def test_load_plugin_errors():
    # failed loads are not recorded, and fail again
    message = (
        "invalid parameter: can not load plugin 'not-a-plugin.so': file does not exist"
    )
    for _ in range(2):
        with pytest.raises(MetatomicError, match=message):
            load_plugin("not-a-plugin.so")


def test_load_model_errors():
    message = (
        "failed to load model from 'not-lj': "
        f"plugin '{PLUGIN}' could not load the model"
    )
    with pytest.raises(MetatomicError, match=message):
        load_model("not-lj", plugin_name=PLUGIN)

    message = "no plugin named 'not-a-plugin' is registered"
    with pytest.raises(MetatomicError, match=message):
        load_model(MODEL, OPTIONS, plugin_name="not-a-plugin")

    message = "unknown Lennard-Jones option: 'unknown'"
    with pytest.raises(MetatomicError, match=message):
        load_model(MODEL, {"unknown": "1"}, plugin_name=PLUGIN)

    message = "Lennard-Jones option 'cutoff' must be finite and positive"
    with pytest.raises(MetatomicError, match=message):
        load_model(MODEL, {**OPTIONS, "cutoff": "0"}, plugin_name=PLUGIN)

    message = "JSON option 'sigma' has a non-string value in `mta_load_model`"
    with pytest.raises(MetatomicError, match=message):
        load_model(MODEL, {**OPTIONS, "sigma": 1.0}, plugin_name=PLUGIN)

    message = "Lennard-Jones option 'atomic_type' must be an integer"
    with pytest.raises(MetatomicError, match=message):
        load_model(MODEL, {**OPTIONS, "atomic_type": "1.6"}, plugin_name=PLUGIN)
