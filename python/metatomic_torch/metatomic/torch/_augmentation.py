from typing import Dict, List, Optional, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import TensorBlock, TensorMap

from . import System, register_autograd_neighbors


def _block_row_indices_by_system(
    block: TensorBlock,
    n_systems: int,
) -> List[torch.Tensor]:
    if "system" not in block.samples.names:
        if n_systems == 1:
            return [torch.arange(block.values.shape[0], device=block.values.device)]
        raise ValueError(
            "Rotational augmentation expects output samples to include a 'system' "
            "dimension when transforming multiple systems."
        )

    system_ids = block.samples.column("system").to(dtype=torch.long)
    if len(system_ids) != 0:
        min_system_id = int(torch.min(system_ids).item())
        max_system_id = int(torch.max(system_ids).item())
        if min_system_id < 0 or max_system_id >= n_systems:
            raise ValueError(
                "Encountered output samples with out-of-range system indices."
            )

    return [
        torch.nonzero(system_ids == system_index, as_tuple=False).reshape(-1)
        for system_index in range(n_systems)
    ]


def _apply_wigner_D_matrices(
    systems: List[System],
    target_tmap: TensorMap,
    transformations: List[torch.Tensor],
    wigner_D_matrices: Dict[int, List[torch.Tensor]],
) -> TensorMap:
    new_blocks: List[TensorBlock] = []
    for key, block in target_tmap.items():
        values = block.values
        row_indices = _block_row_indices_by_system(block, len(systems))

        new_values = values.clone()
        rank = len(block.components)
        if rank == 1:
            ell, sigma = int(key["o3_lambda"]), int(key["o3_sigma"])
            for rows, transformation, wigner_D_matrix in zip(
                row_indices, transformations, wigner_D_matrices[ell], strict=True
            ):
                is_inverted = torch.det(transformation) < 0
                new_v = values[rows].clone()
                if is_inverted:
                    new_v = new_v * (-1) ** ell * sigma
                new_v = new_v.transpose(1, 2)
                new_v = new_v @ wigner_D_matrix.T
                new_v = new_v.transpose(1, 2)
                new_values[rows] = new_v
        elif rank == 2:
            ell1, ell2, sigma1, sigma2 = (
                int(key["o3_lambda_1"]),
                int(key["o3_lambda_2"]),
                int(key["o3_sigma_1"]),
                int(key["o3_sigma_2"]),
            )
            for rows, transformation, wigner_D_matrix1, wigner_D_matrix2 in zip(
                row_indices,
                transformations,
                wigner_D_matrices[ell1],
                wigner_D_matrices[ell2],
                strict=True,
            ):
                is_inverted = torch.det(transformation) < 0
                new_v = values[rows].clone()
                if is_inverted:
                    new_v = new_v * (-1) ** ell1 * sigma1 * (-1) ** ell2 * sigma2
                new_v = torch.einsum(
                    "Aa,iabp,bB->iABp", wigner_D_matrix1, new_v, wigner_D_matrix2.T
                )
                new_values[rows] = new_v
        else:
            raise ValueError(
                f"Unsupported spherical tensor rank {rank} in augmentation helper."
            )
        new_blocks.append(
            TensorBlock(
                values=new_values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )

    return TensorMap(keys=target_tmap.keys, blocks=new_blocks)


def _apply_augmentations(
    systems: List[System],
    targets: Dict[str, TensorMap],
    transformations: List[torch.Tensor],
    wigner_D_matrices: Dict[int, List[torch.Tensor]],
    extra_data: Optional[Dict[str, TensorMap]] = None,
) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
    new_systems: List[System] = []
    for system, transformation in zip(systems, transformations, strict=True):
        new_system = System(
            positions=system.positions @ transformation.T,
            types=system.types,
            cell=system.cell @ transformation.T,
            pbc=system.pbc,
        )
        for data_name in system.known_data():
            data = system.get_data(data_name)
            if len(data) != 1:
                raise ValueError(
                    f"System data '{data_name}' has {len(data)} blocks, which is not "
                    "supported. Only scalar and vector data are supported."
                )
            if len(data.block().components) == 0:
                new_system.add_data(data_name, data)
            elif len(data.block().components) == 1 and data.block().components[
                0
            ].names == ["xyz"]:
                new_system.add_data(
                    data_name,
                    TensorMap(
                        keys=data.keys,
                        blocks=[
                            TensorBlock(
                                values=(
                                    data.block().values.swapaxes(-1, -2)
                                    @ transformation.T
                                ).swapaxes(-1, -2),
                                samples=data.block().samples,
                                components=data.block().components,
                                properties=data.block().properties,
                            )
                        ],
                    ),
                )
            else:
                raise ValueError(
                    f"System data '{data_name}' has components {data.block().components}, "
                    "which are not supported. Only scalar and vector data are supported."
                )
        for options in system.known_neighbor_lists():
            neighbors = mts.detach_block(system.get_neighbor_list(options))
            neighbors.values[:] = (
                neighbors.values.squeeze(-1) @ transformation.T
            ).unsqueeze(-1)
            register_autograd_neighbors(system, neighbors)
            new_system.add_neighbor_list(options, neighbors)
        new_systems.append(new_system)

    new_targets: Dict[str, TensorMap] = {}
    new_extra_data: Dict[str, TensorMap] = {}

    if extra_data is not None:
        mask_keys: List[str] = []
        for key in extra_data.keys():
            if key.endswith("_mask"):
                mask_keys.append(key)
        for key in mask_keys:
            new_extra_data[key] = extra_data.pop(key)

    for tensormap_dict, new_dict in zip(
        [targets, extra_data], [new_targets, new_extra_data], strict=True
    ):
        if tensormap_dict is None:
            continue
        for name, original_tmap in tensormap_dict.items():
            is_scalar = False
            if (
                len(original_tmap.blocks()) == 1
                and len(original_tmap.block().components) == 0
            ):
                is_scalar = True

            is_cartesian = False
            if (
                len(original_tmap.blocks()) == 1
                and len(original_tmap.block().components) > 0
            ):
                if "xyz" in original_tmap.block().components[0].names[0]:
                    is_cartesian = True

            is_spherical = all(
                len(block.components) == 1 and block.components[0].names == ["o3_mu"]
                for block in original_tmap.blocks()
            ) or all(
                len(block.components) == 2
                and block.components[0].names == ["o3_mu_1"]
                and block.components[1].names == ["o3_mu_2"]
                for block in original_tmap.blocks()
            )

            if is_scalar:
                energy_block = TensorBlock(
                    values=original_tmap.block().values,
                    samples=original_tmap.block().samples,
                    components=original_tmap.block().components,
                    properties=original_tmap.block().properties,
                )
                if original_tmap.block().has_gradient("positions"):
                    block = original_tmap.block().gradient("positions")
                    position_gradients = block.values.squeeze(-1)
                    split_sizes = [system.positions.shape[0] for system in systems]
                    split_position_gradients = torch.split(
                        position_gradients, split_sizes
                    )
                    position_gradients = torch.cat(
                        [
                            split_position_gradients[i] @ transformations[i].T
                            for i in range(len(systems))
                        ]
                    )
                    energy_block.add_gradient(
                        "positions",
                        TensorBlock(
                            values=position_gradients.unsqueeze(-1),
                            samples=block.samples,
                            components=block.components,
                            properties=block.properties,
                        ),
                    )
                if original_tmap.block().has_gradient("strain"):
                    block = original_tmap.block().gradient("strain")
                    strain_gradients = block.values.squeeze(-1)
                    split_strain_gradients = torch.split(strain_gradients, 1)
                    new_strain_gradients = torch.stack(
                        [
                            transformations[i]
                            @ split_strain_gradients[i].squeeze(0)
                            @ transformations[i].T
                            for i in range(len(systems))
                        ],
                        dim=0,
                    )
                    energy_block.add_gradient(
                        "strain",
                        TensorBlock(
                            values=new_strain_gradients.unsqueeze(-1),
                            samples=block.samples,
                            components=block.components,
                            properties=block.properties,
                        ),
                    )
                new_dict[name] = TensorMap(
                    keys=original_tmap.keys, blocks=[energy_block]
                )

            elif is_spherical:
                new_dict[name] = _apply_wigner_D_matrices(
                    systems, original_tmap, transformations, wigner_D_matrices
                )

            elif is_cartesian:
                rank = len(original_tmap.block().components)
                block = original_tmap.block()
                row_indices = _block_row_indices_by_system(block, len(systems))
                if rank == 1:
                    new_vectors = block.values.clone()
                    for rows, transformation in zip(
                        row_indices, transformations, strict=True
                    ):
                        v = block.values[rows].clone()
                        new_v = v.transpose(1, 2)
                        new_v = new_v @ transformation.T
                        new_v = new_v.transpose(1, 2)
                        new_vectors[rows] = new_v
                    new_dict[name] = TensorMap(
                        keys=original_tmap.keys,
                        blocks=[
                            TensorBlock(
                                values=new_vectors,
                                samples=block.samples,
                                components=block.components,
                                properties=block.properties,
                            )
                        ],
                    )
                elif rank == 2:
                    new_tensors = block.values.clone()
                    for rows, transformation in zip(
                        row_indices, transformations, strict=True
                    ):
                        tensor_i = block.values[rows].clone()
                        new_tensors[rows] = torch.einsum(
                            "Aa,iabp,bB->iABp",
                            transformation,
                            tensor_i,
                            transformation.T,
                        )
                    new_dict[name] = TensorMap(
                        keys=original_tmap.keys,
                        blocks=[
                            TensorBlock(
                                values=new_tensors,
                                samples=block.samples,
                                components=block.components,
                                properties=block.properties,
                            )
                        ],
                    )
                else:
                    raise ValueError(
                        f"Unsupported Cartesian tensor rank {rank} in augmentation helper."
                    )
            else:
                raise ValueError(
                    f"TensorMap '{name}' is neither scalar, Cartesian, nor spherical in the supported format."
                )

    return new_systems, new_targets, new_extra_data
