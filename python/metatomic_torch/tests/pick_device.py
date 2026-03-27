import pytest
import torch

import metatomic.torch as mta


def test_pick_device_basic():
    # basic call should return a non-empty string describing a device
    res = mta.pick_device(["cpu", "cuda", "mps"], None)
    assert isinstance(res, str)
    assert len(res) > 0
    # sanity: in typical environments we'll at least see "cpu" somewhere
    assert "cpu" == res or "cuda" == res or "mps" == res


def test_pick_device_requested_if_available():
    # if CUDA is available, requesting it should yield a cuda device string
    if torch.cuda.is_available():
        res = mta.pick_device(["cpu", "cuda"], "cuda")
        assert isinstance(res, str)
        assert "cuda" == res
    # if MPS is available, requesting it should yield an mps device string
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        res = mta.pick_device(["cpu", "mps"], "mps")
        assert isinstance(res, str)
        assert "mps" == res


def test_pick_device_ignores_unrecognized_and_warns(capfd):
    # Ensure unrecognized device names are ignored and a warning is emitted
    res = mta.pick_device(["cpu", "fooo"], None)
    assert isinstance(res, str)
    # should pick cpu (ignore "fooo")
    assert "cpu" == res
    # at least one warning should have been produced about the unrecognized/
    # ignored entry
    captured = capfd.readouterr()
    err_output = captured.err.lower()
    assert "fooo" in err_output
    assert "ignoring" in err_output or "unknown" in err_output


def test_pick_device_error_on_unavailable_requested():
    # If a device is explicitly requested but isn't available/declared,
    # the python wrapper is expected to raise a RuntimeError (propagated from C++).
    only_cuda = ["cuda"]
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            mta.pick_device(only_cuda, "cuda")
    else:
        # If CUDA is available, requesting a non-present device should raise
        with pytest.raises(RuntimeError):
            mta.pick_device(["cpu"], "cuda")


def test_pick_device_indexed():
    # Test that indexed device strings like "cpu:0" or "cuda:1" are accepted
    # and preserved.
    res = mta.pick_device(["cpu", "cuda"], "cpu:0")
    assert res == "cpu:0"

    if torch.cuda.is_available():
        res = mta.pick_device(["cpu", "cuda"], "cuda:0")
        assert res == "cuda:0"

    with pytest.raises(RuntimeError, match="invalid device string"):
        mta.pick_device(["cpu"], "cpu:invalid")
