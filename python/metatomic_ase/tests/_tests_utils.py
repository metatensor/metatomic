import os
import re

import pytest
import torch


def _can_use_mps_backend():
    return (
        # Github Actions M1 runners don't have a GPU accessible
        os.environ.get("GITHUB_ACTIONS") is None
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


ALL_DEVICE_DTYPE = [("cpu", "float64"), ("cpu", "float32")]
if torch.cuda.is_available():
    ALL_DEVICE_DTYPE.append(("cuda", "float64"))
    ALL_DEVICE_DTYPE.append(("cuda", "float32"))

if _can_use_mps_backend():
    ALL_DEVICE_DTYPE.append(("mps", "float32"))


STR_TO_DTYPE = {
    "float32": torch.float32,
    "float64": torch.float64,
}


class prints_to_stderr:
    def __init__(self, capfd, match=""):
        self.capfd = capfd
        self.match = match

    def __enter__(self):
        out, err = self.capfd.readouterr()
        if out != "":
            pytest.fail(
                "Expected no output to stdout before prints_to_stderr block, "
                f"got:\n'{out}'"
            )

        if err != "":
            pytest.fail(
                "Expected no output to stderr before prints_to_stderr block, "
                f"got:\n'{err}'"
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        out, err = self.capfd.readouterr()

        if out != "":
            pytest.fail(
                f"Expected no output to stdout in prints_to_stderr block, got:\n'{out}'"
            )

        if re.search(self.match, err) is None:
            pytest.fail(
                "Expected output to stderr matching "
                f"'{self.match}' in prints_to_stderr block, got:\n'{err}'"
            )
