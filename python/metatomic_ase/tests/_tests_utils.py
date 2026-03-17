import os

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
