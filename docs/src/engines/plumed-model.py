from typing import Dict, List, Optional

import metatensor.torch as mts
import torch

import metatomic.torch as mta


class Distance(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def requested_neighbor_lists(self) -> List[mta.NeighborListOptions]:
        # request a neighbor list to be computed and stored in the
        # system passed to `forward`. In this case we return an empty
        # list because we don't use neighbor lists
        return []

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[mts.Labels],
    ) -> Dict[str, mts.TensorMap]:
        if "features" not in outputs:
            return {}

        if outputs["features"].per_atom:
            raise ValueError("per-atoms features are not supported in this model")

        # PLUMED will first call the model with 0 atoms to get the size of the
        # output, so we need to handle this case first
        if len(systems[0]) == 0:
            # prepares an empty TensorMap with the correct metadata
            keys = mts.Labels("_", torch.tensor([[0]]))
            block = mts.TensorBlock(
                # this shape will be used by PLUMED to know that the CV only has 1 entry
                torch.zeros((0, 1), dtype=torch.float64),
                samples=mts.Labels("structure", torch.zeros((0, 1), dtype=torch.int32)),
                components=[],
                properties=mts.Labels(
                    "distance", torch.zeros((1, 1), dtype=torch.int32)
                ),
            )

            return {"features": mts.TensorMap(keys, [block])}

        if selected_atoms is None:
            raise ValueError("this model requires selected_atoms to be set")

        if len(selected_atoms) != 2:
            raise ValueError(
                "the model should be given two atoms to compute a distance"
            )

        atom_ids = selected_atoms.column("atom").squeeze()
        values = []
        for system in systems:
            if len(system.positions) < selected_atoms.values.max():
                raise ValueError(
                    "System size is too small for the selected atoms indices"
                )
            distance = torch.sqrt(
                (
                    (system.positions[atom_ids[0]] - system.positions[atom_ids[1]]) ** 2
                ).sum()
            )
            values.append(distance)

        # creates a tensor map to contain the values and return
        keys = mts.Labels("_", torch.tensor([[0]]))
        block = mts.TensorBlock(
            torch.stack(values, dim=0).reshape(-1, 1),
            samples=mts.Labels(
                "system",
                torch.arange(len(systems), dtype=torch.int32).reshape((-1, 1)),
            ),
            components=[],
            properties=mts.Labels("distance", torch.zeros((1, 1), dtype=torch.int32)),
        )

        return {"features": mts.TensorMap(keys, [block])}


# instantiates the model, describes its metadata, and export
distance = Distance()

# metatdata about the model itself
metadata = mta.ModelMetadata(
    name="Distance",
    description="Computes the distance between two selected atoms",
)

# metatdata about what the model can do
capabilities = mta.ModelCapabilities(
    length_unit="Angstrom",
    outputs={"features": mta.ModelOutput(per_atom=False)},
    atomic_types=[0],
    interaction_range=torch.inf,
    supported_devices=["cpu", "cuda"],
    dtype="float64",
)

model = mta.AtomisticModel(
    module=distance.eval(),
    metadata=metadata,
    capabilities=capabilities,
)

model.save("mta-distance.pt")
