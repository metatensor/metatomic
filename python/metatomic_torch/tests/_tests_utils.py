import os

import torch


def can_use_mps_backend():
    return (
        # Github Actions M1 runners don't have a GPU accessible
        os.environ.get("GITHUB_ACTIONS") is None
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )
