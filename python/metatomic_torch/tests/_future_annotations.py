from __future__ import annotations

from typing import Optional

import torch
from metatensor.torch import Labels, TensorMap

from metatomic.torch import ModelOutput, System


class FutureAnnotationsModel(torch.nn.Module):
    def forward(
        self,
        systems: list[System],
        outputs: dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> dict[str, TensorMap]:
        return {}
