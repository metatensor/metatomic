import logging
import time
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import metatensor.torch as mts


if TYPE_CHECKING:

    class TensorBlock: ...

    class System: ...

    class TensorMap: ...

    class ModelOutput: ...

    class Labels: ...

    class ModelInterface: ...

else:
    from metatensor.torch import Labels, TensorBlock, TensorMap

    from metatomic.torch import ModelOutput, System

import numpy as np
import torch

from metatomic.torch import ModelInterface, register_autograd_neighbors
from metatomic.torch._augmentation import _apply_augmentations
from metatomic.torch._wigner import compute_real_wigner_d_matrices


LOGGER = logging.getLogger(__name__)


try:
    from scipy.integrate import lebedev_rule  # noqa: F401
    from scipy.spatial.transform import Rotation  # noqa: F401
except ImportError as e:
    raise ImportError(
        "To perform data augmentation on spherical targets, please "
        "install the `scipy` package with `pip install scipy`."
    ) from e


def _choose_quadrature(L_max: int) -> Tuple[int, int]:
    """
    Choose a Lebedev quadrature order and number of in-plane rotations to integrate
    spherical harmonics up to degree ``L_max``.

    :param L_max: maximum spherical harmonic degree
    :return: (lebedev_order, n_inplane_rotations)
    """
    available = [
        3,
        5,
        7,
        9,
        11,
        13,
        15,
        17,
        19,
        21,
        23,
        25,
        27,
        29,
        31,
        35,
        41,
        47,
        53,
        59,
        65,
        71,
        77,
        83,
        89,
        95,
        101,
        107,
        113,
        119,
        125,
        131,
    ]
    # pick smallest order >= L_max
    n = min(o for o in available if o >= L_max)
    # minimal gamma count
    K = L_max + 1
    return n, K


def get_euler_angles_quadrature(lebedev_order: int, n_rotations: int):
    """
    Get the Euler angles and weights for a Lebedev quadrature combined with in-plane
    rotations for SO(3) integration.

    :param lebedev_order: order of the Lebedev quadrature on the unit sphere
    :param n_rotations: number of in-plane rotations per Lebedev node
    :return: alpha, beta, gamma, w arrays of shape (M,), (M,), (K,), (M,)
        respectively, where M is the number of Lebedev nodes and K is the number of
        in-plane rotations.
    """

    # Lebedev nodes (X: (3, M))
    X, w = lebedev_rule(lebedev_order)  # w sums to 4*pi
    x, y, z = X
    alpha = np.arctan2(y, x)  # (M,)
    beta = np.arccos(np.clip(z, -1.0, 1.0))  # (M,)
    gamma = np.linspace(0.0, 2 * np.pi, n_rotations, endpoint=False)  # (K,)

    w_so3 = np.repeat(w / (4 * np.pi * n_rotations), repeats=gamma.size)  # (M*K,)

    A = np.repeat(alpha, gamma.size)  # (N,)
    B = np.repeat(beta, gamma.size)  # (N,)
    G = np.tile(gamma, alpha.size)  # (N,)

    return A, B, G, w_so3


def _rotations_from_angles(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray
) -> Rotation:
    """
    Compose rotations from ZYZ Euler angles.

    :param alpha: array of alpha angles (M,)
    :param beta: array of beta angles (M,)
    :param gamma: array of gamma angles (K,)
    :return: Rotation object containing all (M*K,) rotations
    """

    # Compose ZYZ rotations in SO(3)
    Rot = (
        Rotation.from_euler("z", alpha.reshape(-1, 1))
        * Rotation.from_euler("y", beta.reshape(-1, 1))
        * Rotation.from_euler("z", gamma.reshape(-1, 1))
    )

    return Rot


def _transform_system(system: System, transformation: torch.Tensor) -> System:
    transformed_system = System(
        positions=system.positions @ transformation.T,
        types=system.types,
        cell=system.cell @ transformation.T,
        pbc=system.pbc,
    )
    for options in system.known_neighbor_lists():
        neighbors = mts.detach_block(system.get_neighbor_list(options))

        neighbors.values[:] = (
            neighbors.values.squeeze(-1) @ transformation.T
        ).unsqueeze(-1)

        register_autograd_neighbors(system, neighbors)
        transformed_system.add_neighbor_list(options, neighbors)
    return transformed_system


def _complex_to_real_spherical_harmonics_transform(ell: int) -> np.ndarray:
    """
    Generate the transformation matrix from complex spherical harmonics
    to real spherical harmonics for a given l.
    Returns a transformation matrix of shape ((2l+1), (2l+1)).
    """
    if ell < 0 or not isinstance(ell, int):
        raise ValueError("l must be a non-negative integer.")

    # The size of the transformation matrix is (2l+1) x (2l+1)
    size = 2 * ell + 1
    T = np.zeros((size, size), dtype=complex)

    for m in range(-ell, ell + 1):
        m_index = m + ell  # Index in the matrix
        if m > 0:
            # Real part of Y_{l}^{m}
            T[m_index, ell + m] = 1 / np.sqrt(2) * (-1) ** m
            T[m_index, ell - m] = 1 / np.sqrt(2)
        elif m < 0:
            # Imaginary part of Y_{l}^{|m|}
            T[m_index, ell + abs(m)] = -1j / np.sqrt(2) * (-1) ** m
            T[m_index, ell - abs(m)] = 1j / np.sqrt(2)
        else:  # m == 0
            # Y_{l}^{0} remains unchanged
            T[m_index, ell] = 1

    # Return the transformation matrix to convert complex to real spherical harmonics
    return T


def _compute_real_wigner_matrices(
    o3_lambda_max: int,
    angles: Tuple[np.ndarray, np.ndarray, np.ndarray],  # alpha, beta, gamma
) -> Dict[int, np.ndarray]:
    complex_to_real = {
        ell: _complex_to_real_spherical_harmonics_transform(ell)
        for ell in range(o3_lambda_max + 1)
    }
    return compute_real_wigner_d_matrices(o3_lambda_max, angles, complex_to_real)


def _angles_from_rotations(
    R: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Z-Y-Z Euler angles (alpha, beta, gamma) from rotation matrices, with
    explicit handling of the gimbal-lock cases (beta≈0 and beta≈pi).
    TODO: This function is extremely sensitive to eps and will be modified.
    Parameters
    ----------
    R : np.ndarray
        Rotation matrices with arbitrary batch shape `(..., 3, 3)`.
    eps : float
        Tolerance used to detect gimbal lock via `sin(beta) < eps`.

    Returns
    -------
    (alphas, betas, gammas) : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Each with the same batch shape as `R[..., 0, 0]` (i.e., `R.shape[:-2]`).

    Notes
    -----
    Conventions:
      - Base convention is Z-Y-Z (Rz(alpha) Ry(beta) Rz(gamma)).
      - For beta≈0: set beta=0, gamma=0, alpha=atan2(R[1,0], R[0,0]).
      - For beta≈pi: set beta=pi, alpha=0,  gamma=atan2(R[1,0], -R[0,0]).
    These conventions ensure a deterministic inverse where the standard formulas
    are ill-conditioned.
    """
    # Accept any batch shape. Flatten to (N, 3, 3) for clarity, then unflatten.
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)

    # Read commonly-used entries with explicit names for readability
    R00 = R_flat[:, 0, 0]
    # R01 = R_flat[:, 0, 1]
    R02 = R_flat[:, 0, 2]
    R10 = R_flat[:, 1, 0]
    # R11 = R_flat[:, 1, 1]
    R12 = R_flat[:, 1, 2]
    R20 = R_flat[:, 2, 0]
    R21 = R_flat[:, 2, 1]
    R22 = R_flat[:, 2, 2]

    # Default (non-singular) extraction
    zz = np.clip(R22, -1.0, 1.0)
    betas = np.arccos(zz)

    # For Z–Y–Z, standard formulas away from the singular set
    alphas = np.arctan2(R12, R02)
    gammas = np.arctan2(R21, -R20)

    # Normalize into [0, 2π)
    two_pi = 2.0 * np.pi
    alphas = np.mod(alphas, two_pi)
    gammas = np.mod(gammas, two_pi)

    # Gimbal-lock detection via sin(beta)
    sinb = np.sin(betas)
    near = np.abs(sinb) < eps
    if np.any(near):
        # Split the two singular bands using zz = cos(beta)
        near_zero = near & (zz > 0)  # beta≈0
        near_pi = near & (zz < 0)  # beta≈pi

        if np.any(near_zero):
            # beta≈0: rotation ≈ Rz(alpha+gamma). Choose gamma=0, recover alpha from
            # 2x2 block.
            betas[near_zero] = 0.0
            gammas[near_zero] = 0.0
            alphas[near_zero] = np.arctan2(R10[near_zero], R00[near_zero])
            alphas[near_zero] = np.mod(alphas[near_zero], two_pi)

        if np.any(near_pi):
            # beta≈pi: choose alpha=0, recover gamma from 2x2 block with sign flip on
            # R00.
            betas[near_pi] = np.pi
            alphas[near_pi] = 0.0
            gammas[near_pi] = np.arctan2(R10[near_pi], -R00[near_pi])
            gammas[near_pi] = np.mod(gammas[near_pi], two_pi)

    # Unflatten back to the original batch shape
    alphas = alphas.reshape(batch_shape)
    betas = betas.reshape(batch_shape)
    gammas = gammas.reshape(batch_shape)
    return alphas, betas, gammas


def _l0_components_from_matrices(A: torch.Tensor) -> torch.Tensor:
    """
    Extract the L=0 components from a (3, 3) tensor.
    """
    # The tensor will have shape (a, 3, 3, b) so we need to move the 3, 3 dimension at
    # the end
    A = A.permute(0, 3, 1, 2)
    assert A.shape[-2:] == (3, 3), "The last two dimensions of A must be (3, 3)."

    # Trace as L=0 component; unsqueeze preserves the autograd graph
    l0_A = (A[..., 0, 0] + A[..., 1, 1] + A[..., 2, 2]).unsqueeze(-1)

    l0_A = l0_A.permute(0, 2, 1)
    return l0_A


def _l2_components_from_matrices(A: torch.Tensor) -> torch.Tensor:
    """
    Extract the L=2 components from a (3, 3) tensor.
    """
    # The tensor will have shape (a, 3, 3, b) so we need to move the 3, 3 dimension at
    # the end
    A = A.permute(0, 3, 1, 2)
    assert A.shape[-2:] == (3, 3), "The last two dimensions of A must be (3, 3)."

    # Use torch.stack to preserve the autograd graph
    l2_A = torch.stack(
        [
            (A[..., 0, 1] + A[..., 1, 0]) / 2.0,
            (A[..., 1, 2] + A[..., 2, 1]) / 2.0,
            (2.0 * A[..., 2, 2] - A[..., 0, 0] - A[..., 1, 1]) / (2.0 * np.sqrt(3.0)),
            (A[..., 0, 2] + A[..., 2, 0]) / 2.0,
            (A[..., 0, 0] - A[..., 1, 1]) / 2.0,
        ],
        dim=-1,
    )

    l2_A = l2_A.permute(0, 2, 1)

    return l2_A


def decompose_energy_tensor(
    tensor_dict: Dict[str, TensorMap],
) -> Dict[str, TensorMap]:
    """
    Decompose energy tensor into its L=0 irreducible representation.

    Energy is a scalar, so it lives entirely in the L=0 sector. This function
    adds an ``o3_mu`` component axis with a single m=0 entry to make the format
    consistent with higher-order decompositions.

    :param tensor_dict: dictionary of TensorMaps (modified in place)
    :return: the same dictionary with ``"energy"`` replaced by ``"energy_l0"``
    """
    for key in ["energy", "energy_total"]:
        if key not in tensor_dict:
            continue

        tensor = tensor_dict[key]
        tensor_dict[key + "_l0"] = TensorMap(
            tensor.keys,
            [
                TensorBlock(
                    values=block.values.unsqueeze(1),
                    samples=block.samples,
                    components=[
                        Labels(
                            names=["o3_mu"],
                            values=torch.tensor(
                                [[0]], device=block.values.device, dtype=torch.int32
                            ),
                        )
                    ],
                    properties=block.properties,
                )
                for block in tensor
            ],
        )
        tensor_dict.pop(key)

    return tensor_dict


def decompose_forces_tensor(
    tensor_dict: Dict[str, TensorMap],
) -> Dict[str, TensorMap]:
    """
    Decompose forces tensors into L=1 irreducible representations.

    Forces are Cartesian vectors (x, y, z). This reorders them to spherical
    component order (y, z, x) → (m=-1, m=0, m=1) via a cyclic roll, and
    labels the component axis as ``o3_mu``.

    Handles both ``"forces"`` (conservative) and ``"non_conservative_forces"`` keys.

    :param tensor_dict: dictionary of TensorMaps (modified in place)
    :return: the same dictionary with forces keys replaced by ``"..._l1"`` variants
    """
    for key in ["forces", "non_conservative_forces"]:
        if key not in tensor_dict:
            continue

        tensor = tensor_dict[key]
        tensor_dict[key + "_l1"] = TensorMap(
            tensor.keys,
            [
                TensorBlock(
                    values=block.values.roll(-1, 1),
                    samples=block.samples,
                    components=[
                        Labels(
                            names="o3_mu",
                            values=torch.tensor(
                                [[mu] for mu in range(-1, 2)],
                                device=block.values.device,
                                dtype=torch.int32,
                            ),
                        )
                    ],
                    properties=block.properties,
                )
                for block in tensor
            ],
        )
        tensor_dict.pop(key)
    return tensor_dict


def decompose_stress_tensor(
    tensor_dict: Dict[str, TensorMap],
) -> Dict[str, TensorMap]:
    """
    Decompose stress tensors into L=0 (trace) and L=2 (symmetric traceless) parts.

    The 3x3 stress tensor decomposes as: trace (L=0 scalar) + symmetric traceless
    (L=2, 5 components). The antisymmetric part (L=1) is zero for physical stress.

    Handles both ``"stress"`` (conservative) and ``"non_conservative_stress"`` keys.

    :param tensor_dict: dictionary of TensorMaps (modified in place)
    :return: the same dictionary with stress keys replaced by ``"..._l0"`` and
        ``"..._l2"`` variants
    """
    for key in ["stress", "non_conservative_stress"]:
        if key not in tensor_dict:
            continue

        tensor = tensor_dict[key]
        blocks_l0 = []
        blocks_l2 = []
        for block in tensor.blocks():
            trace_values = _l0_components_from_matrices(block.values)
            block_l0 = TensorBlock(
                values=trace_values,
                samples=block.samples,
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.tensor(
                            [[0]], device=block.values.device, dtype=torch.int32
                        ),
                    )
                ],
                properties=block.properties,
            )
            blocks_l0.append(block_l0)

            block_l2 = TensorBlock(
                values=_l2_components_from_matrices(block.values),
                samples=block.samples,
                components=[
                    Labels(
                        names="o3_mu",
                        values=torch.tensor(
                            [[mu] for mu in range(-2, 3)],
                            device=block.values.device,
                            dtype=torch.int32,
                        ),
                    )
                ],
                properties=block.properties,
            )
            blocks_l2.append(block_l2)

        tensor_dict[key + "_l0"] = TensorMap(tensor.keys, blocks_l0)
        tensor_dict[key + "_l2"] = TensorMap(tensor.keys, blocks_l2)
        tensor_dict.pop(key)

    return tensor_dict


def decompose_tensors(
    tensor_dict: Dict[str, TensorMap],
    device: torch.device,
) -> Dict[str, TensorMap]:
    """
    Decompose all tensors in the dictionary into irreducible representations of O(3).

    :param tensor_dict: dictionary of TensorMaps to decompose
    :param device: device for label tensors
    :return: dictionary of TensorMaps with decomposed tensors
    """
    tensor_dict = decompose_energy_tensor(tensor_dict)
    tensor_dict = decompose_forces_tensor(tensor_dict)
    tensor_dict = decompose_stress_tensor(tensor_dict)
    return tensor_dict


def _copy_tensor_dict(tensor_dict: Dict[str, TensorMap]) -> Dict[str, TensorMap]:
    return {name: tensor for name, tensor in tensor_dict.items()}


def _maybe_add_energy_total(tensor_dict: Dict[str, TensorMap]) -> Dict[str, TensorMap]:
    tensor_dict = _copy_tensor_dict(tensor_dict)
    if (
        "energy" in tensor_dict
        and "atom" in tensor_dict["energy"].block().samples.names
    ):
        tensor_dict["energy_total"] = mts.sum_over_samples(
            tensor_dict["energy"], ["atom"]
        )
    return tensor_dict


def _normalize_output_tensors(
    name: str,
    tensor: TensorMap,
    device: torch.device,
) -> Dict[str, TensorMap]:
    return decompose_tensors(_maybe_add_energy_total({name: tensor}), device)


def _tensor_map_dtype(tensor: TensorMap) -> torch.dtype:
    return tensor.block().values.dtype


def _key_to_tuple(key_entry) -> Tuple[int, ...]:
    return tuple(int(v) for v in key_entry.values.tolist())


def _prepend_system_to_samples(
    sample_names: List[str],
    sample_values: torch.Tensor,
    system_index: int,
    *,
    device: torch.device,
) -> Labels:
    system_values = torch.full(
        (sample_values.shape[0], 1),
        system_index,
        dtype=torch.int32,
        device=device,
    )
    if len(sample_names) == 0:
        return Labels(["system"], system_values)

    return Labels(
        ["system"] + sample_names,
        torch.cat(
            [system_values, sample_values.to(device=device, dtype=torch.int32)], dim=1
        ),
    )


def _selected_atoms_for_local_systems(
    selected_atoms: Optional[Labels],
    system_index: int,
    n_local_systems: int,
) -> Optional[Labels]:
    if selected_atoms is None:
        return None

    system_mask = selected_atoms.column("system").to(dtype=torch.long) == system_index
    system_selected_atoms = selected_atoms.values[system_mask]
    if system_selected_atoms.shape[0] == 0:
        return Labels(
            list(selected_atoms.names),
            selected_atoms.values.new_empty((0, len(selected_atoms.names))),
        )

    local_selected_atoms: List[torch.Tensor] = []
    for local_system_index in range(n_local_systems):
        local_values = system_selected_atoms.clone()
        local_values[:, 0] = local_system_index
        local_selected_atoms.append(local_values)

    return Labels(
        list(selected_atoms.names),
        torch.cat(local_selected_atoms, dim=0),
    )


def _reshape_block_by_local_system(
    block: TensorBlock, n_local_systems: int
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    local_ids = block.samples.column("system").to(dtype=torch.long)
    if len(local_ids) != 0:
        min_local_id = int(torch.min(local_ids).item())
        max_local_id = int(torch.max(local_ids).item())
        if min_local_id < 0 or max_local_id >= n_local_systems:
            raise ValueError(
                "Encountered output samples with out-of-range system indices."
            )

    split_values: List[torch.Tensor] = []
    base_sample_values: Optional[torch.Tensor] = None
    for local_system_index in range(n_local_systems):
        local_mask = local_ids == local_system_index
        local_values = block.values[local_mask]
        local_sample_values = block.samples.values[local_mask][:, 1:]
        if base_sample_values is None:
            base_sample_values = local_sample_values
        elif not torch.equal(local_sample_values, base_sample_values):
            raise ValueError(
                "Streaming SymmetrizedModel expects each rotated copy of a system to "
                "produce the same sample labels in the same order."
            )
        split_values.append(local_values)

    assert base_sample_values is not None
    stacked_values = torch.stack(split_values, dim=0)
    return stacked_values, list(block.samples.names[1:]), base_sample_values


def _reduce_weighted_batch_tensor(
    tensor: TensorMap,
    weights: torch.Tensor,
    system_index: int,
    *,
    component_norm: bool = False,
    elementwise_square: bool = False,
    half_weight: bool = True,
) -> TensorMap:
    n_local_systems = weights.numel()
    reduced_blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        values, sample_names, sample_values = _reshape_block_by_local_system(
            block, n_local_systems
        )

        components = block.components
        if component_norm:
            component_dims = tuple(range(2, 2 + len(block.components)))
            if len(component_dims) == 0:
                norm_squared = values**2
            else:
                norm_squared = torch.sum(values**2, dim=component_dims)
            values = norm_squared**2 if elementwise_square else norm_squared
            components = []
        elif elementwise_square:
            values = values**2

        weight = weights.to(dtype=values.dtype, device=values.device)
        factor = 0.5 if half_weight else 1.0
        view = [values.shape[0]] + [1] * (values.ndim - 1)
        reduced_values = torch.sum(factor * weight.view(view) * values, dim=0)
        if reduced_values.ndim == 1:
            reduced_values = reduced_values.unsqueeze(0)

        reduced_blocks.append(
            TensorBlock(
                values=reduced_values,
                samples=_prepend_system_to_samples(
                    sample_names,
                    sample_values,
                    system_index,
                    device=block.samples.values.device,
                ),
                components=components,
                properties=block.properties,
            )
        )

    return TensorMap(tensor.keys, reduced_blocks)


def _accumulate_tensormap(
    accumulators: Dict[str, TensorMap], name: str, contribution: TensorMap
) -> None:
    if name in accumulators:
        accumulators[name] = mts.add(accumulators[name], contribution)
    else:
        accumulators[name] = contribution


def _append_tensormap(
    accumulators: Dict[str, List[TensorMap]], name: str, contribution: TensorMap
) -> None:
    accumulators.setdefault(name, []).append(contribution)


def _join_tensormap_list(tensors: List[TensorMap]) -> TensorMap:
    if len(tensors) == 1:
        return tensors[0]
    return mts.join(tensors, "samples", different_keys="union")


def _component_norm_squared(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 3:
        return values
    if values.ndim > 3:
        dims = list(range(2, values.ndim - 1))
        return torch.sum(values**2, dim=dims)
    return values**2


def _mean_norm_squared_tensor(tensor: TensorMap) -> TensorMap:
    blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        if block.values.ndim > 2:
            values = torch.sum(
                block.values**2, dim=tuple(range(1, block.values.ndim - 1))
            )
        else:
            values = block.values**2
        if values.ndim == 1:
            values = values.unsqueeze(0)
        blocks.append(
            TensorBlock(
                values=values,
                samples=block.samples,
                components=[],
                properties=block.properties,
            )
        )
    return TensorMap(tensor.keys, blocks)


def _finalize_variance(
    second_moment: TensorMap,
    mean: TensorMap,
) -> TensorMap:
    mean_norm_sq = _mean_norm_squared_tensor(mean)
    return mts.subtract(second_moment, mean_norm_sq)


def _compute_batch_projection_contributions(
    tensor: TensorMap,
    weights: torch.Tensor,
    wigner_matrices: Dict[int, torch.Tensor],
    max_o3_lambda_character: int,
    *,
    storage_device: Optional[torch.device] = None,
) -> Dict[Tuple[int, ...], Dict[str, object]]:
    n_local_systems = weights.numel()
    block_contributions: Dict[Tuple[int, ...], Dict[str, object]] = {}
    for key, block in tensor.items():
        key_tuple = _key_to_tuple(key)
        values, sample_names, sample_values = _reshape_block_by_local_system(
            block, n_local_systems
        )
        weight = weights.to(dtype=values.dtype, device=values.device)
        weighted_values = (
            weight.view([weight.shape[0]] + [1] * (values.ndim - 1)) * values
        )

        coefficients: Dict[int, torch.Tensor] = {}
        for ell in range(max_o3_lambda_character + 1):
            D = wigner_matrices[ell].to(dtype=values.dtype, device=values.device)
            coefficient = torch.einsum("imn,i...->mn...", D, weighted_values)
            if storage_device is not None and coefficient.device != storage_device:
                coefficient = coefficient.to(device=storage_device)
            coefficients[ell] = coefficient

        key_values = key.values.clone()
        sample_values_out = sample_values.clone()
        components = list(block.components)
        properties = block.properties
        if storage_device is not None:
            key_values = key_values.to(device=storage_device)
            sample_values_out = sample_values_out.to(device=storage_device)
            components = [component.to(device=storage_device) for component in block.components]
            properties = block.properties.to(device=storage_device)

        block_contributions[key_tuple] = {
            "key_names": list(tensor.keys.names),
            "key_values": key_values,
            "sample_names": sample_names,
            "sample_values": sample_values_out,
            "components": components,
            "properties": properties,
            "coefficients": coefficients,
        }

    return block_contributions


def _merge_projection_contributions(
    accumulator: Dict[Tuple[int, ...], Dict[str, object]],
    contribution: Dict[Tuple[int, ...], Dict[str, object]],
) -> None:
    for key_tuple, entry in contribution.items():
        if key_tuple not in accumulator:
            accumulator[key_tuple] = entry
            continue
        existing = accumulator[key_tuple]
        existing_coefficients = existing["coefficients"]
        contribution_coefficients = entry["coefficients"]
        assert isinstance(existing_coefficients, dict)
        assert isinstance(contribution_coefficients, dict)
        for ell, tensor in contribution_coefficients.items():
            if ell in existing_coefficients:
                existing_coefficients[ell] = existing_coefficients[ell] + tensor
            else:
                existing_coefficients[ell] = tensor


def _finalize_projection_tensor(
    positive: Dict[Tuple[int, ...], Dict[str, object]],
    negative: Dict[Tuple[int, ...], Dict[str, object]],
    system_index: int,
    max_o3_lambda_character: int,
) -> Optional[TensorMap]:
    all_keys = list(positive.keys())
    for key in negative.keys():
        if key not in positive:
            all_keys.append(key)

    if len(all_keys) == 0:
        return None

    blocks: List[TensorBlock] = []
    key_values: List[torch.Tensor] = []
    key_names: Optional[List[str]] = None
    for key_tuple in all_keys:
        plus_entry = positive.get(key_tuple)
        minus_entry = negative.get(key_tuple)
        meta = plus_entry if plus_entry is not None else minus_entry
        assert meta is not None

        key_names = list(meta["key_names"])
        key_tensor = meta["key_values"]
        sample_names = meta["sample_names"]
        sample_values = meta["sample_values"]
        components = meta["components"]
        properties = meta["properties"]
        plus_coeffs = plus_entry["coefficients"] if plus_entry is not None else {}
        minus_coeffs = minus_entry["coefficients"] if minus_entry is not None else {}

        for ell in range(max_o3_lambda_character + 1):
            plus_tensor = plus_coeffs.get(ell)
            minus_tensor = minus_coeffs.get(ell)
            if plus_tensor is None and minus_tensor is None:
                continue
            if plus_tensor is None:
                plus_tensor = torch.zeros_like(minus_tensor)
            if minus_tensor is None:
                minus_tensor = torch.zeros_like(plus_tensor)

            parity = (-1) ** ell
            for sigma in [1, -1]:
                combined = plus_tensor + sigma * parity * minus_tensor
                values = (
                    0.25 * (2 * ell + 1) * torch.sum(combined * combined, dim=(0, 1))
                )
                blocks.append(
                    TensorBlock(
                        values=values,
                        samples=_prepend_system_to_samples(
                            sample_names,
                            sample_values,
                            system_index,
                            device=values.device,
                        ),
                        components=components,
                        properties=properties,
                    )
                )
                key_values.append(
                    torch.cat(
                        [
                            key_tensor,
                            torch.tensor(
                                [ell, sigma],
                                dtype=key_tensor.dtype,
                                device=key_tensor.device,
                            ),
                        ]
                    )
                )

    assert key_names is not None
    tensor = TensorMap(
        Labels(key_names + ["chi_lambda", "chi_sigma"], torch.stack(key_values)),
        blocks,
    )
    if "_" in tensor.keys.names:
        tensor = mts.remove_dimension(tensor, "keys", "_")
    return tensor


def _slice_angles(
    angles: Tuple[np.ndarray, np.ndarray, np.ndarray],
    start: int,
    stop: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(angle[start:stop] for angle in angles)


def _compute_wigner_batch(
    ell_max: int,
    angles: Tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[int, torch.Tensor]:
    return {
        ell: tensor.to(device=device, dtype=dtype)
        for ell, tensor in _compute_real_wigner_matrices(ell_max, angles).items()
    }


def _compute_wigner_batch_lists(
    ell_max: int,
    angles: Tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[int, List[torch.Tensor]]:
    return {
        ell: list(tensor.to(device=device, dtype=dtype).unbind(0))
        for ell, tensor in _compute_real_wigner_matrices(ell_max, angles).items()
    }


class SymmetrizedModel(torch.nn.Module):
    r"""
    Wrapper around an atomistic model that symmetrizes its outputs over :math:`O(3)`
    and computes equivariance metrics.

    The model is evaluated over a quadrature grid on :math:`O(3)`, constructed from a
    Lebedev grid supplemented by in-plane rotations. For each sampled group element, the
    model outputs are "back-rotated" according to the known :math:`O(3)` action
    appropriate for their tensorial type (scalar, vector, tensor, etc.). Averaging these
    back-rotated predictions over the quadrature grid yields fully
    :math:`O(3)`-symmetrized outputs. In addition, two complementary equivariance
    metrics are computed:

    1. Variance under :math:`O(3)` of the back-rotated outputs.

        For a perfectly equivariant model, the back-rotated output :math:`x(g)` is
        independent of the group element :math:`g`. Deviations from perfect equivariance
        are quantified by the difference between the average squared norm over
        :math:`O(3)` and the squared norm of the :math:`O(3)`-averaged output:

        .. math::

            \mathrm{Var}_{O(3)}[x]
            =
            \left\langle \,\| x(g) \|^{2} \,\right\rangle_{O(3)}
            -
            \left\| \left\langle x(g) \right\rangle_{O(3)} \right\|^{2} .

        Here, :math:`\|\cdot\|` denotes the Euclidean norm over the ``component`` axis,
        and :math:`\langle \cdot \rangle_{O(3)}` denotes averaging over the quadrature
        grid. This quantity is the squared norm of the component orthogonal to the
        perfectly equivariant subspace and therefore provides a scalar measure of the
        deviation from exact equivariance.

    2. Decomposition into isotypical components of :math:`O(3)`.

        Each output component may be viewed as a scalar function on :math:`O(3)`,
        which can be decomposed into isotypical components labeled by the irreducible
        representations :math:`\ell,\sigma` of :math:`O(3)`. The projection onto the
        :math:`(\ell,\sigma)`-th isotypical subspace is computed as a convolution with
        the corresponding character :math:`\chi_{\ell}`:

        .. math::

            (P_{\ell,\sigma} x)(g)
            =
            \int_{O(3)} \chi_{\ell,\sigma}(h^{-1} g)\, x(h)\, \mathrm{d}\mu(h).

        Its squared :math:`L^{2}` norm over :math:`O(3)` is

        .. math::

            \| P_{\ell,\sigma} x \|^{2}
            =
            \left\langle \, | (P_{\ell,\sigma} x)(g) |^{2} \, \right\rangle_{O(3)} .

        These quantities describe how the model output is distributed across the
        different :math:`O(3)` irreducible sectors. The complementary component,
        orthogonal to all isotypical subspaces, is given by

        .. math::

            \| x \|^{2}
            -
            \sum_{\ell,\sigma} \| P_{\ell,\sigma} x \|^{2} ,

        and provides a refined measure of the deviation from lying entirely within any
        prescribed set of :math:`O(3)` irreducible representations.

    :param base_model: atomistic model to symmetrize
    :param max_o3_lambda: maximum O(3) angular momentum the grid integrates exactly
    :param batch_size: number of rotations to evaluate in a single batch
    :param max_o3_lambda_character: maximum O(3) angular momentum for character
        projections. If None, set to ``max_o3_lambda``.
    :param offload_to_cpu: if True, move model outputs to CPU before symmetry
        postprocessing. If None, preserve the historical behavior of offloading only
        when ``compute_gradients=False``.
    """

    def __init__(
        self,
        base_model,
        max_o3_lambda_character: int,
        max_o3_lambda_target: int,
        batch_size: int = 32,
        max_o3_lambda_grid: Optional[int] = None,
        offload_to_cpu: Optional[bool] = None,
    ):
        super().__init__()
        self.base_model = base_model

        try:
            ref_param = next(base_model.parameters())
            device = ref_param.device
            dtype = ref_param.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.get_default_dtype()

        self.max_o3_lambda_target = max_o3_lambda_target
        self.batch_size = batch_size
        self.offload_to_cpu = offload_to_cpu
        if max_o3_lambda_grid is None:
            max_o3_lambda_grid = int(2 * max_o3_lambda_character + 1)
        self.max_o3_lambda_grid = max_o3_lambda_grid
        self.max_o3_lambda_character = max_o3_lambda_character

        # Compute grid (unchanged)
        lebedev_order, n_inplane_rotations = _choose_quadrature(self.max_o3_lambda_grid)
        if lebedev_order < 2 * self.max_o3_lambda_character:
            warnings.warn(
                "Lebedev order may be insufficient for character projections.",
                stacklevel=2,
            )
        alpha, beta, gamma, w_so3 = get_euler_angles_quadrature(
            lebedev_order, n_inplane_rotations
        )
        so3_weights = torch.from_numpy(w_so3).to(device=device, dtype=dtype)
        self.register_buffer("so3_weights", so3_weights)

        so3_rotations = torch.from_numpy(
            _rotations_from_angles(alpha, beta, gamma).as_matrix()
        ).to(device=device, dtype=dtype)
        self.register_buffer("so3_rotations", so3_rotations)
        self.n_so3_rotations = self.so3_rotations.size(0)

        angles_inverse_rotations = (np.pi - gamma, beta, np.pi - alpha)
        so3_inverse_rotations = torch.from_numpy(
            _rotations_from_angles(*angles_inverse_rotations).as_matrix()
        ).to(device=device, dtype=dtype)
        self.register_buffer("so3_inverse_rotations", so3_inverse_rotations)
        self._quadrature_angles = (alpha, beta, gamma)
        self._inverse_quadrature_angles = angles_inverse_rotations

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
        project_tokens: bool = False,
        compute_gradients: bool = False,
    ) -> Dict[str, TensorMap]:
        """
        Symmetrize the model outputs over :math:`O(3)` and compute equivariance
        metrics.

        :param systems: list of systems to evaluate
        :param outputs: dictionary of model outputs to symmetrize
        :param selected_atoms: optional Labels specifying which atoms to consider
        :param project_tokens: if True, also compute character projections
        :param compute_gradients: if True, compute conservative forces and stress
            via autograd. When False (default), the grid evaluation runs under
            ``torch.no_grad()`` to save memory.
        :return: dictionary with symmetrized outputs and equivariance metrics
        """
        device = self.so3_weights.device
        offload = (not compute_gradients) and (
            True if self.offload_to_cpu is None else self.offload_to_cpu
        )
        work_device = torch.device("cpu") if offload else device

        with torch.no_grad() if offload else torch.enable_grad():
            transformed_outputs, backtransformed_outputs = self._eval_over_grid(
                systems,
                outputs,
                selected_atoms,
                return_transformed=project_tokens,
                compute_gradients=compute_gradients,
                offload_to_cpu=offload,
            )

        out_dict: Dict[str, TensorMap] = {
            name: tensor.to(device=work_device)
            for name, tensor in backtransformed_outputs.items()
        }

        if not project_tokens:
            return out_dict

        for name, tensor in transformed_outputs.items():
            if name.endswith("_componentwise_norm_squared"):
                out_dict[name] = tensor.to(device=work_device)
            else:
                out_dict[name] = tensor.to(device=work_device)

        return out_dict

    def _eval_over_grid(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
        return_transformed: bool,
        compute_gradients: bool = False,
        offload_to_cpu: bool = False,
    ) -> Tuple[Dict[str, TensorMap], Dict[str, TensorMap]]:
        """
        Stream the model over the O(3) quadrature, accumulating mean, variance, and
        character projections without ever materializing the full-grid TensorMap.

        :param systems: list of systems to evaluate
        :param outputs: dictionary of model outputs to symmetrize
        :param selected_atoms: optional Labels specifying which atoms to consider
        :param return_transformed: if True, also return un-back-rotated outputs
        :param compute_gradients: if True, compute forces/stress via autograd
        :return: (transformed_outputs, backtransformed_outputs) dictionaries
        """
        eval_start = time.perf_counter()
        n_rotations = self.n_so3_rotations
        requested_output_names = list(outputs.keys())
        if compute_gradients:
            if "energy" not in outputs:
                raise ValueError("compute_gradients=True requires 'energy' in outputs")

            requested_output_names = list(
                dict.fromkeys(requested_output_names + ["forces"])
            )
            if any(bool(torch.any(s.pbc).item()) for s in systems):
                requested_output_names = list(
                    dict.fromkeys(requested_output_names + ["stress"])
                )

        mean_accumulators: Dict[str, List[TensorMap]] = {}
        second_moment_accumulators: Dict[str, List[TensorMap]] = {}
        character_projection_accumulators: Dict[str, List[TensorMap]] = {}
        proj_pos_accumulators: Dict[str, List[TensorMap]] = {}
        proj_neg_accumulators: Dict[str, List[TensorMap]] = {}

        for i_sys, system in enumerate(systems):
            system_start = time.perf_counter()
            system_mean_accumulators: Dict[str, TensorMap] = {}
            system_second_moment_accumulators: Dict[str, TensorMap] = {}
            system_proj_pos_accumulators: Dict[
                str, Dict[Tuple[int, ...], Dict[str, object]]
            ] = {}
            system_proj_neg_accumulators: Dict[
                str, Dict[Tuple[int, ...], Dict[str, object]]
            ] = {}

            for inversion in [1, -1]:
                work_device = system.positions.device
                work_dtype = system.positions.dtype

                if compute_gradients:
                    effective_batch_size = 1
                    batch_starts = list(range(0, n_rotations, 1))
                else:
                    effective_batch_size = self.batch_size
                    batch_starts = list(range(0, n_rotations, self.batch_size))

                for batch_start in batch_starts:
                    batch_stop = min(batch_start + effective_batch_size, n_rotations)
                    n_local_systems = batch_stop - batch_start
                    batch_rotations = self.so3_rotations[batch_start:batch_stop].to(
                        device=work_device, dtype=work_dtype
                    )
                    inversion_batch = inversion * batch_rotations
                    local_selected_atoms = _selected_atoms_for_local_systems(
                        selected_atoms,
                        i_sys,
                        n_local_systems,
                    )

                    if compute_gradients:
                        out = _evaluate_with_gradients(
                            self.base_model,
                            system,
                            inversion_batch.squeeze(0),
                            outputs,
                            local_selected_atoms,
                            work_device,
                            system.positions.dtype,
                        )
                        rotation_outputs = [out]
                    else:
                        transformed_systems = [
                            _transform_system(
                                system,
                                R.to(device=work_device),
                            )
                            for R in inversion_batch
                        ]
                        out = self.base_model(
                            transformed_systems,
                            outputs,
                            local_selected_atoms,
                        )
                        if offload_to_cpu:
                            out = {k: v.to(device="cpu") for k, v in out.items()}
                        rotation_outputs = [out]

                    present_output_names = [
                        name for name in requested_output_names if name in rotation_outputs[0]
                    ]
                    if len(present_output_names) == 0:
                        continue

                    weights = self.so3_weights[batch_start:batch_stop]
                    batch_augmentation_cache = {}

                    for name in present_output_names:
                        tensor = rotation_outputs[0][name]
                        tensor_dtype = _tensor_map_dtype(tensor)
                        tensor_device = tensor.block().values.device
                        cache_key = (str(tensor_device), str(tensor_dtype))
                        if cache_key not in batch_augmentation_cache:
                            augmentation_system = (
                                system
                                if system.positions.device == tensor_device
                                else system.to(
                                    device=tensor_device,
                                    dtype=system.positions.dtype,
                                )
                            )

                            batch_wigner = _compute_wigner_batch(
                                max(
                                    self.max_o3_lambda_target,
                                    self.max_o3_lambda_character,
                                ),
                                _slice_angles(
                                    self._inverse_quadrature_angles,
                                    batch_start,
                                    batch_stop,
                                ),
                                device=tensor_device,
                                dtype=tensor_dtype,
                            )
                            wigner_dict: Dict[int, List[torch.Tensor]] = {
                                ell: list(mat.unbind(0))
                                for ell, mat in batch_wigner.items()
                            }
                            inverse_mats = (
                                inversion
                                * self.so3_inverse_rotations[batch_start:batch_stop]
                            ).to(device=tensor_device, dtype=tensor_dtype)
                            inverse_rotations = list(inverse_mats.unbind(0))
                            batch_augmentation_cache[cache_key] = (
                                augmentation_system,
                                batch_wigner,
                                wigner_dict,
                                inverse_rotations,
                            )

                        (
                            augmentation_system,
                            batch_wigner,
                            wigner_dict,
                            inverse_rotations,
                        ) = batch_augmentation_cache[cache_key]

                        _, backtransformed_batch, _ = _apply_augmentations(
                            [augmentation_system] * n_local_systems,
                            {name: tensor},
                            inverse_rotations,
                            wigner_dict,
                        )
                        for (
                            final_name,
                            backtransformed_tensor,
                        ) in _normalize_output_tensors(
                            name,
                            backtransformed_batch[name],
                            tensor_device,
                        ).items():
                            mean_batch = _reduce_weighted_batch_tensor(
                                backtransformed_tensor,
                                weights,
                                i_sys,
                                component_norm=False,
                                half_weight=True,
                            )
                            _accumulate_tensormap(
                                system_mean_accumulators, final_name, mean_batch
                            )

                            second_moment_batch = _reduce_weighted_batch_tensor(
                                backtransformed_tensor,
                                weights,
                                i_sys,
                                component_norm=True,
                                elementwise_square=False,
                                half_weight=True,
                            )
                            _accumulate_tensormap(
                                system_second_moment_accumulators,
                                final_name,
                                second_moment_batch,
                            )

                            if return_transformed:
                                projection_storage_device = (
                                    torch.device("cpu")
                                    if tensor_device.type != "cpu"
                                    else tensor_device
                                )
                                block_contribution = (
                                    _compute_batch_projection_contributions(
                                        backtransformed_tensor,
                                        weights,
                                        batch_wigner,
                                        self.max_o3_lambda_character,
                                        storage_device=projection_storage_device,
                                    )
                                )
                                accumulators = (
                                    system_proj_pos_accumulators
                                    if inversion == 1
                                    else system_proj_neg_accumulators
                                )
                                accumulators.setdefault(final_name, {})
                                _merge_projection_contributions(
                                    accumulators[final_name], block_contribution
                                )

            if return_transformed:
                projection_names = set(system_proj_pos_accumulators) | set(
                    system_proj_neg_accumulators
                )
                for name in projection_names:
                    char_proj = _finalize_projection_tensor(
                        system_proj_pos_accumulators.get(name, {}),
                        system_proj_neg_accumulators.get(name, {}),
                        i_sys,
                        self.max_o3_lambda_character,
                    )
                    if char_proj is not None:
                        character_projection_accumulators.setdefault(name, []).append(
                            char_proj
                        )

                    proj_pos_final = _finalize_projection_tensor(
                        system_proj_pos_accumulators.get(name, {}),
                        {},
                        i_sys,
                        self.max_o3_lambda_character,
                    )
                    if proj_pos_final is not None:
                        proj_pos_accumulators.setdefault(name, []).append(
                            proj_pos_final
                        )

                    proj_neg_final = _finalize_projection_tensor(
                        {},
                        system_proj_neg_accumulators.get(name, {}),
                        i_sys,
                        self.max_o3_lambda_character,
                    )
                    if proj_neg_final is not None:
                        proj_neg_accumulators.setdefault(name, []).append(
                            proj_neg_final
                        )

            for name, tensor in system_mean_accumulators.items():
                _append_tensormap(mean_accumulators, name, tensor)

            for name, tensor in system_second_moment_accumulators.items():
                _append_tensormap(second_moment_accumulators, name, tensor)

            LOGGER.info(
                "SymmetrizedModel progress: system %s/%s finished in %.2fs "
                "(project_tokens=%s, outputs=%s, batch_size=%s, offload_to_cpu=%s)",
                i_sys + 1,
                len(systems),
                time.perf_counter() - system_start,
                return_transformed,
                len(requested_output_names),
                self.batch_size,
                offload_to_cpu,
            )

        mean_results: Dict[str, TensorMap] = {}
        for name, mean_tensors in mean_accumulators.items():
            mean_tensor = _join_tensormap_list(mean_tensors)
            mean_results[name] = mean_tensor
            mean_results[name + "_mean"] = mean_tensor

            norm_squared = _join_tensormap_list(second_moment_accumulators[name])
            mean_results[name + "_var"] = _finalize_variance(
                norm_squared,
                mean_tensor,
            )
            mean_results[name + "_norm_squared"] = norm_squared

        if not return_transformed:
            return {}, mean_results

        transformed_results: Dict[str, TensorMap] = {}
        backtransformed_results = dict(mean_results)

        for name, tensors in character_projection_accumulators.items():
            if tensors:
                backtransformed_results[name + "_character_projection"] = (
                    _join_tensormap_list(tensors)
                )

        for name, tensors in proj_pos_accumulators.items():
            if tensors:
                transformed_results[name + "_character_projection_plus"] = (
                    tensors[0] if len(tensors) == 1 else _join_tensormap_list(tensors)
                )

        for name, tensors in proj_neg_accumulators.items():
            if tensors:
                transformed_results[name + "_character_projection_minus"] = (
                    tensors[0] if len(tensors) == 1 else _join_tensormap_list(tensors)
                )

        LOGGER.info(
            "SymmetrizedModel finished %s systems in %.2fs "
            "(project_tokens=%s, outputs=%s, batch_size=%s, offload_to_cpu=%s)",
            len(systems),
            time.perf_counter() - eval_start,
            return_transformed,
            len(requested_output_names),
            self.batch_size,
            offload_to_cpu,
        )

        return transformed_results, backtransformed_results


def _evaluate_with_gradients(
    model: ModelInterface,
    system: System,
    rotation: torch.Tensor,
    outputs: Dict[str, ModelOutput],
    selected_atoms: Optional[Labels],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, TensorMap]:
    """
    Evaluate model on a single rotated system and compute conservative forces/stress
    via autograd.

    Forces are ``-dE/d(positions)`` in the rotated frame. Stress is computed via the
    strain trick as ``(1/V) dE/d(strain)`` in the rotated frame. Both are packaged as
    Cartesian TensorMaps suitable for back-rotation by the existing pipeline.

    :param model: atomistic model to evaluate
    :param system: input system (original frame)
    :param rotation: 3x3 rotation matrix (may include inversion)
    :param outputs: model output specifications
    :param selected_atoms: optional atom selection
    :param device: device for tensors
    :param dtype: dtype for tensors
    :return: model output dict with added ``"forces"`` and (if periodic) ``"stress"``
    """
    n_atoms = system.positions.shape[0]
    R = rotation.to(device=device, dtype=dtype)

    # Rotate positions (detached from original graph) and enable grad tracking
    rotated_positions = (system.positions.detach() @ R.T).requires_grad_(True)
    rotated_cell = system.cell.detach() @ R.T

    # Strain trick for stress (applied in the rotated frame)
    has_cell = bool(torch.any(system.pbc).item())
    if has_cell:
        strain = torch.eye(3, requires_grad=True, device=device, dtype=dtype)
        final_positions = rotated_positions @ strain
        final_cell = rotated_cell @ strain
    else:
        strain = None
        final_positions = rotated_positions
        final_cell = rotated_cell

    # Build transformed system
    transformed = System(
        types=system.types,
        positions=final_positions,
        cell=final_cell,
        pbc=system.pbc,
    )

    # Copy and register neighbor lists for autograd
    for options in system.known_neighbor_lists():
        neighbors = mts.detach_block(system.get_neighbor_list(options))
        neighbors.values[:] = (neighbors.values.squeeze(-1) @ R.T).unsqueeze(-1)
        register_autograd_neighbors(transformed, neighbors)
        transformed.add_neighbor_list(options, neighbors)

    # Evaluate model
    out = model([transformed], outputs, selected_atoms)

    if "energy" not in out:
        raise ValueError("compute_gradients=True requires the model to output 'energy'")
    energy_sum = out["energy"].block().values.sum()

    # Compute gradients via autograd
    grad_targets = [rotated_positions]
    if strain is not None:
        grad_targets.append(strain)
    grads = torch.autograd.grad(energy_sum, grad_targets, create_graph=False)

    # Forces: -dE/d(rotated_positions) in the rotated frame
    forces_values = -grads[0]  # (n_atoms, 3)

    key_labels = Labels(
        names=["_"],
        values=torch.tensor([[0]], dtype=torch.int64, device=device),
    )

    forces_block = TensorBlock(
        values=forces_values.unsqueeze(-1),  # (n_atoms, 3, 1)
        samples=Labels(
            names=["system", "atom"],
            values=torch.stack(
                [
                    torch.zeros(n_atoms, dtype=torch.int64, device=device),
                    torch.arange(n_atoms, dtype=torch.int64, device=device),
                ],
                dim=1,
            ),
        ),
        components=[
            Labels(
                "xyz",
                torch.arange(3, dtype=torch.int64, device=device).reshape(-1, 1),
            )
        ],
        properties=Labels(
            names=["energy"],
            values=torch.tensor([[0]], dtype=torch.int64, device=device),
        ),
    )
    forces_tmap = TensorMap(key_labels, [forces_block])
    if selected_atoms is not None:
        forces_tmap = mts.slice(forces_tmap, axis="samples", selection=selected_atoms)
    out["forces"] = forces_tmap

    # Stress: (1/V) dE/d(strain) in the rotated frame
    if strain is not None:
        volume = torch.abs(torch.linalg.det(system.cell.detach()))
        stress_values = grads[1] / volume  # (3, 3)

        stress_block = TensorBlock(
            values=stress_values.unsqueeze(0).unsqueeze(-1),  # (1, 3, 3, 1)
            samples=Labels(
                names=["system"],
                values=torch.tensor([[0]], dtype=torch.int64, device=device),
            ),
            components=[
                Labels(
                    "xyz_1",
                    torch.arange(3, dtype=torch.int64, device=device).reshape(-1, 1),
                ),
                Labels(
                    "xyz_2",
                    torch.arange(3, dtype=torch.int64, device=device).reshape(-1, 1),
                ),
            ],
            properties=Labels(
                names=["energy"],
                values=torch.tensor([[0]], dtype=torch.int64, device=device),
            ),
        )
        out["stress"] = TensorMap(key_labels, [stress_block])

    return out
