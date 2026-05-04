import pytest
import torch

import metatomic.torch as mta


def test_pick_device_basic():
    # basic call should return a torch.device
    res = mta.pick_device(["cpu", "cuda", "mps"], None)
    assert isinstance(res, torch.device)
    # sanity: in typical environments we'll at least see "cpu", "cuda" or "mps"
    assert res.type in ("cpu", "cuda", "mps")


def test_pick_device_requested_if_available():
    # if CUDA is available, requesting it should yield a cuda device
    if torch.cuda.is_available():
        res = mta.pick_device(["cpu", "cuda"], "cuda")
        assert isinstance(res, torch.device)
        assert res.type == "cuda"
    # if MPS is available, requesting it should yield an mps device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        res = mta.pick_device(["cpu", "mps"], "mps")
        assert isinstance(res, torch.device)
        assert res.type == "mps"


def test_pick_device_no_index_when_auto():
    # When no desired_device is given, the result should have no specific index
    res = mta.pick_device(["cpu"], None)
    assert isinstance(res, torch.device)
    assert res.type == "cpu"
    # torch.device("cpu") has index -1 (unset); torch.device("cpu:0") has index 0
    assert res.index is None


def test_pick_device_ignores_unrecognized_and_warns(capfd):
    # Ensure unrecognized device names are ignored and a warning is emitted
    res = mta.pick_device(["cpu", "fooo"], None)
    assert isinstance(res, torch.device)
    # should pick cpu (ignore "fooo")
    assert res.type == "cpu"
    # at least one warning should have been produced about the unrecognized/
    # ignored entry
    captured = capfd.readouterr()
    assert captured.out == ""

    message = "Warning: ignoring unknown device 'fooo' from `model_devices`"
    assert message in captured.err


def test_pick_device_error_on_unavailable_requested():
    # Test if a device is explicitly requested but isn't available/declared"
    if torch.cuda.is_available():
        model_devices = ["cpu"]
    else:
        model_devices = ["cuda"]

    match = (
        "failed to find a valid device. "
        "None of the model-supported devices are available."
    )
    with pytest.raises(ValueError, match=match):
        mta.pick_device(model_devices, "cuda")


def test_pick_device_indexed():
    # Test that indexed device strings like "cpu:0" or "cuda:1" are accepted
    # and that the index is preserved in the returned torch.device.
    res = mta.pick_device(["cpu", "cuda"], "cpu:0")
    assert isinstance(res, torch.device)
    assert res.type == "cpu"
    assert res.index == 0

    if torch.cuda.is_available():
        res = mta.pick_device(["cpu", "cuda"], "cuda:0")
        assert isinstance(res, torch.device)
        assert res.type == "cuda"
        assert res.index == 0

    with pytest.raises(ValueError, match="invalid device string"):
        mta.pick_device(["cpu"], "cpu:invalid")
