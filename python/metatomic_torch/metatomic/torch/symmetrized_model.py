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
            (2.0 * A[..., 2, 2] - A[..., 0, 0] - A[..., 1, 1])
            / (2.0 * np.sqrt(3.0)),
            (A[..., 0, 2] + A[..., 2, 0]) / 2.0,
            (A[..., 0, 0] - A[..., 1, 1]) / 2.0,
        ],
        dim=-1,
    )

    l2_A = l2_A.permute(0, 2, 1)

    return l2_A


def _euler_angles_of_combined_rotation(
    angles1: Tuple[np.ndarray, np.ndarray, np.ndarray],
    angles2: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given two sets of Euler angles (alpha, beta, gamma), returns the Euler angles
    of all pairwise compositions
    """

    R1 = _rotations_from_angles(*angles1).as_matrix()  # (N1, 3, 3)
    R2 = _rotations_from_angles(*angles2).as_matrix()  # (N2, 3, 3)

    # Broadcasted pairwise multiplication to shape (N1, N2, 3, 3): R1[p] @ R2[a]
    R_product = R1[:, None, :, :] @ R2[None, :, :, :]

    # Extract Euler angles from the combined rotation matrices (robust to gimbal lock)
    alpha, beta, gamma = _angles_from_rotations(R_product, eps=1e-6)
    return alpha, beta, gamma


def _get_so3_character(
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    o3_lambda: int,
    tol: float = 1e-7,
) -> np.ndarray:
    """
    Numerically stable evaluation of the character function χ_{o3_lambda}(R) over SO(3).

    Uses a small-angle Taylor expansion for χ_l(ω) = sin((2l+1)t)/sin(t) with t = ω/2
    when |t| is very small, and a guarded ratio otherwise.
    """
    # Compute half-angle t = ω/2 via Z–Y–Z relation: cos t = cos(β/2) cos((α+γ)/2)
    cos_t = np.cos(betas / 2.0) * np.cos((alphas + gammas) / 2.0)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    t = np.arccos(cos_t)

    # Output array
    chi = np.empty_like(t)

    # Parameters for χ
    L = o3_lambda
    a = 2 * L + 1
    ll1 = L * (L + 1)

    small = np.abs(t) < tol
    if np.any(small):
        # Series up to t^4: χ ≈ a [1 - (2/3) ℓ(ℓ+1) t^2 + (1/45) ℓ(ℓ+1)(3ℓ^2+3ℓ-1) t^4]
        ts = t[small]
        t2 = ts * ts
        coeff4 = ll1 * (3 * L * L + 3 * L - 1)
        chi[small] = a * (
            1.0 - (2.0 / 3.0) * ll1 * t2 + (1.0 / 45.0) * coeff4 * t2 * t2
        )

    # Large-angle (or not-so-small) branch: safe ratio with guard
    large = ~small
    if np.any(large):
        tl = t[large]
        sin_t = np.sin(tl)
        numer = np.sin(a * tl)
        mask = np.abs(sin_t) >= tol
        out = np.empty_like(tl)
        np.divide(numer, sin_t, out=out, where=mask)
        out[~mask] = a  # exact limit as t -> 0
        chi[large] = out

    return chi


def compute_characters(
    o3_lambda_max: int,
    angles: Tuple[np.ndarray, np.ndarray, np.ndarray],
    inverse_angles: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[Dict[int, torch.Tensor], Dict[str, torch.Tensor]]:
    alpha, beta, gamma = _euler_angles_of_combined_rotation(angles, inverse_angles)

    so3_characters = {
        o3_lambda: _get_so3_character(alpha, beta, gamma, o3_lambda)
        for o3_lambda in range(o3_lambda_max + 1)
    }

    pso3_characters = {}
    for o3_lambda in range(o3_lambda_max + 1):
        for o3_sigma in [-1, +1]:
            pso3_characters[f"{o3_lambda}_{o3_sigma}"] = (
                o3_sigma * ((-1) ** o3_lambda) * so3_characters[o3_lambda]
            )

    so3_characters = {
        key: torch.from_numpy(value) for key, value in so3_characters.items()
    }
    pso3_characters = {
        key: torch.from_numpy(value) for key, value in pso3_characters.items()
    }

    return so3_characters, pso3_characters


def _character_convolution(
    chi: torch.Tensor, block1: TensorBlock, block2: TensorBlock, w: torch.Tensor
) -> TensorBlock:
    """
    Compute the character convolution of a block containing SO(3)-sampled tensors.
    Then contract with another block.
    """
    samples = block1.samples
    assert samples.names[0] == "so3_rotation"
    n_rot = chi.size(0)
    components = block1.components
    properties = block1.properties
    values = block1.values
    chi = chi.to(dtype=values.dtype, device=values.device)
    n_rot = chi.size(1)
    weight = w.to(dtype=values.dtype, device=values.device)

    split_sizes = torch.bincount(samples.values[:, 1]).tolist()
    split_by_system = torch.split(values, split_sizes, dim=0)
    tensor_list: List[torch.Tensor] = []
    for split_tensor, size in zip(split_by_system, split_sizes, strict=True):
        split_size = [size // n_rot] * n_rot
        split_by_rotation = torch.stack(torch.split(split_tensor, split_size, dim=0))
        tensor_list.append(split_by_rotation)
    split_by_rotation = torch.cat(tensor_list, dim=1)
    reshaped_values = split_by_rotation

    # broadcast weights to match reshaped_values
    view: List[int] = []
    view.append(-1)
    for _ in range(reshaped_values.ndim - 1):
        view.append(1)
    weighted_values = weight.view(view) * reshaped_values

    # broadcast characters to match reshaped_values
    contracted_shape: List[int] = [chi.shape[0]] + list(weighted_values.shape[1:])
    contracted_values = (
        chi @ weighted_values.reshape(weighted_values.shape[0], -1)
    ).reshape(contracted_shape)

    values2 = block2.values
    split_sizes = torch.bincount(block2.samples.values[:, 1]).tolist()
    split_by_system = torch.split(values2, split_sizes, dim=0)
    tensor_list: List[torch.Tensor] = []
    for split_tensor, size in zip(split_by_system, split_sizes, strict=True):
        split_size = [size // n_rot] * n_rot
        split_by_rotation = torch.stack(torch.split(split_tensor, split_size, dim=0))
        tensor_list.append(split_by_rotation)
    split_by_rotation = torch.cat(tensor_list, dim=1)
    reshaped_values2 = split_by_rotation

    # broadcast weights to match reshaped_values2
    view: List[int] = []
    view.append(-1)
    for _ in range(reshaped_values2.ndim - 1):
        view.append(1)
    weighted_values2 = weight.view(view) * reshaped_values2

    # contract weighted_values2 with contracted_values
    contracted_values = torch.einsum(
        "i...,i...->...",
        weighted_values2,
        contracted_values,
    )

    names: List[str] = []
    for name in samples.names:
        if name != "so3_rotation":
            names.append(name)
    new_block = TensorBlock(
        samples=Labels(names, samples.values[samples.values[:, 0] == 0][:, 1:]),
        components=components,
        properties=properties,
        values=contracted_values,
    )

    return new_block


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
    if "energy" not in tensor_dict:
        return tensor_dict

    tensor = tensor_dict["energy"]
    tensor_dict["energy_l0"] = TensorMap(
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
    tensor_dict.pop("energy")
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
    if "energy" in tensor_dict and "atom" in tensor_dict["energy"].block().samples.names:
        tensor_dict["energy_total"] = mts.sum_over_samples(tensor_dict["energy"], ["atom"])
    return tensor_dict


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
        torch.cat([system_values, sample_values.to(device=device, dtype=torch.int32)], dim=1),
    )


def _reshape_block_by_local_system(
    block: TensorBlock, n_local_systems: int
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    local_ids = block.samples.column("system").to(dtype=torch.long)
    split_sizes = torch.bincount(local_ids, minlength=n_local_systems).tolist()
    if len(set(split_sizes)) != 1:
        raise ValueError(
            "Streaming SymmetrizedModel expects each rotated copy of a system to "
            "produce the same sample layout."
        )
    if split_sizes[0] == 0:
        raise ValueError("Encountered an output block with no samples for any rotation.")

    split_values = torch.split(block.values, split_sizes, dim=0)
    stacked_values = torch.stack(split_values, dim=0)
    base_sample_values = block.samples.values[local_ids == 0][:, 1:]
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

        if elementwise_square:
            values = values**2

        components = block.components
        if component_norm:
            values = _component_norm_squared(values)
            components = []

        weight = weights.to(dtype=values.dtype, device=values.device)
        factor = 0.5 if half_weight else 1.0
        view = [values.shape[0]] + [1] * (values.ndim - 1)
        reduced_values = torch.sum(factor * weight.view(view) * values, dim=0)

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


def _join_tensormap_list(tensors: List[TensorMap]) -> TensorMap:
    if len(tensors) == 1:
        return tensors[0]
    return mts.join(tensors, "samples", different_keys="union")


def _mean_norm_squared_tensor(tensor: TensorMap) -> TensorMap:
    blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        blocks.append(
            TensorBlock(
                values=_component_norm_squared(block.values),
                samples=block.samples,
                components=[],
                properties=block.properties,
            )
        )
    return TensorMap(tensor.keys, blocks)


def _compute_batch_projection_contributions(
    tensor: TensorMap,
    weights: torch.Tensor,
    wigner_matrices: Dict[int, torch.Tensor],
    max_o3_lambda_character: int,
) -> Dict[Tuple[int, ...], Dict[str, object]]:
    n_local_systems = weights.numel()
    block_contributions: Dict[Tuple[int, ...], Dict[str, object]] = {}
    for key, block in tensor.items():
        key_tuple = _key_to_tuple(key)
        values, sample_names, sample_values = _reshape_block_by_local_system(
            block, n_local_systems
        )
        weight = weights.to(dtype=values.dtype, device=values.device)
        weighted_values = weight.view([weight.shape[0]] + [1] * (values.ndim - 1)) * values

        coefficients: Dict[int, torch.Tensor] = {}
        for ell in range(max_o3_lambda_character + 1):
            D = wigner_matrices[ell].to(dtype=values.dtype, device=values.device)
            coefficients[ell] = torch.einsum("imn,i...->mn...", D, weighted_values)

        block_contributions[key_tuple] = {
            "key_names": list(tensor.keys.names),
            "key_values": key.values.clone(),
            "sample_names": sample_names,
            "sample_values": sample_values.clone(),
            "components": block.components,
            "properties": block.properties,
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
                values = 0.25 * (2 * ell + 1) * torch.sum(combined * combined, dim=(0, 1))
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


def _combine_single_rotation_outputs(
    rotation_outputs: List[Dict[str, TensorMap]],
) -> Dict[str, TensorMap]:
    if len(rotation_outputs) == 1:
        return rotation_outputs[0]

    output_names = set()
    for output in rotation_outputs:
        output_names.update(output.keys())

    combined_outputs: Dict[str, TensorMap] = {}
    for name in output_names:
        blocks = [output[name] for output in rotation_outputs if name in output]
        combined = mts.join(blocks, "samples", add_dimension="batch_rotation")
        if "batch_rotation" in combined[0].samples.names:
            new_blocks: List[TensorBlock] = []
            for block in combined.blocks():
                batch_id = block.samples.column("batch_rotation")
                rot_id = block.samples.column("system")
                new_sample_values = block.samples.values[:, :-1].clone()
                new_sample_values[:, 0] = batch_id + rot_id
                new_blocks.append(
                    TensorBlock(
                        values=block.values,
                        samples=Labels(block.samples.names[:-1], new_sample_values),
                        components=block.components,
                        properties=block.properties,
                    )
                )
            combined = TensorMap(combined.keys, new_blocks)
        combined_outputs[name] = combined
    return combined_outputs


def _weight_by_quadrature(
    so3_weights: torch.Tensor, rot_ids: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Apply SO(3) quadrature weights to values, broadcasting over all dims
    except the first (samples)."""
    view = [values.size(0)] + [1] * (values.ndim - 1)
    return 0.5 * so3_weights[rot_ids].view(view) * values


def _component_norm_squared(values: torch.Tensor) -> torch.Tensor:
    """Sum of squares over component dimensions (all dims except first and last)."""
    if values.ndim > 2:
        dims = list(range(1, values.ndim - 1))
        return torch.sum(values**2, dim=dims)
    return values**2


def compute_norm_per_property(
    tensor_dict: Dict[str, TensorMap],
    so3_weights: torch.Tensor,
) -> Dict[str, TensorMap]:
    """
    Compute the weighted squared norm per property of each tensor.

    For each output, computes the quadrature-weighted sum of squared values
    over the O(3) grid, giving the squared norm in each irrep sector per property.

    :param tensor_dict: dictionary of TensorMaps with ``so3_rotation`` in samples
    :param so3_weights: quadrature weights, shape ``(n_rotations,)``
    :return: dictionary of TensorMaps with componentwise squared norms
    """
    norms: Dict[str, TensorMap] = {}
    for name in tensor_dict:
        tensor = tensor_dict[name]
        norm_blocks: List[TensorBlock] = []
        for block in tensor.blocks():
            rot_ids = block.samples.column("so3_rotation")
            values_squared = _weight_by_quadrature(
                so3_weights, rot_ids, block.values**2
            )

            norm_blocks.append(
                TensorBlock(
                    values=values_squared,
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
            )

        tensor_norm = TensorMap(tensor.keys, norm_blocks)
        tensor_norm = mts.sum_over_samples(
            tensor_norm.keys_to_samples("inversion"), ["inversion", "so3_rotation"]
        )

        norms[name + "_componentwise_norm_squared"] = tensor_norm
    return norms


def compute_conv_integral(
    tensor_dict: Dict[str, TensorMap],
    so3_weights: torch.Tensor,
    so3_characters: Dict[int, torch.Tensor],
    pso3_characters: Dict[str, torch.Tensor],
    max_o3_lambda_character: int,
) -> Dict[str, TensorMap]:
    """
    Compute character convolution integrals over O(3) for each tensor.

    Projects each output onto O(3) irrep sectors by convolving with
    the characters chi_{l,sigma}. The result measures how much of the
    output's variance lives in each (l, sigma) sector.

    :param tensor_dict: dictionary of TensorMaps with rotation samples
    :param so3_weights: quadrature weights
    :param so3_characters: SO(3) characters, mapping l → tensor of shape (N_rot, N_rot)
    :param pso3_characters: P*SO(3) characters, mapping "l_sigma" → tensor
    :param max_o3_lambda_character: maximum angular momentum for projection
    :return: dictionary of TensorMaps with character projections
    """
    new_tensors: Dict[str, TensorMap] = {}
    for name, tensor in tensor_dict.items():
        keys = tensor.keys
        remaining_keys = Labels(
            keys.names[:-1],
            keys.values[keys.column("inversion") == 1][:, :-1],
        )
        new_blocks: List[TensorBlock] = []
        new_keys: List[torch.Tensor] = []
        for key_values in remaining_keys.values:
            key_to_match_plus: Dict[str, int] = {}
            key_to_match_minus: Dict[str, int] = {}
            for k, v in zip(remaining_keys.names, key_values, strict=True):
                key_to_match_plus[k] = int(v)
                key_to_match_minus[k] = int(v)
            key_to_match_plus["inversion"] = 1
            key_to_match_minus["inversion"] = -1
            so3_block = tensor.block(key_to_match_plus)
            pso3_block = tensor.block(key_to_match_minus)

            for o3_lambda in range(max_o3_lambda_character + 1):
                so3_chi = so3_characters[o3_lambda]
                first_term = _character_convolution(
                    so3_chi, so3_block, so3_block, so3_weights
                )
                second_term = _character_convolution(
                    so3_chi, pso3_block, pso3_block, so3_weights
                )
                for o3_sigma in [1, -1]:
                    label = str(o3_lambda) + "_" + str(o3_sigma)
                    pso3_chi = pso3_characters[label]
                    third_term = _character_convolution(
                        pso3_chi, pso3_block, so3_block, so3_weights
                    )
                    block = TensorBlock(
                        samples=first_term.samples,
                        components=first_term.components,
                        properties=first_term.properties,
                        values=(
                            0.25 * (first_term.values + second_term.values)
                            + 0.5 * third_term.values
                        )
                        * (2 * o3_lambda + 1),
                    )
                    new_blocks.append(block)
                    new_keys.append(
                        torch.cat(
                            [
                                key_values,
                                torch.tensor(
                                    [o3_lambda, o3_sigma],
                                    device=key_values.device,
                                    dtype=key_values.dtype,
                                ),
                            ]
                        )
                    )
        key_names: List[str] = []
        for key_name in tensor.keys.names:
            if key_name != "inversion":
                key_names.append(key_name)
        new_tensor = TensorMap(
            Labels(
                key_names + ["chi_lambda", "chi_sigma"],
                torch.stack(new_keys),
            ),
            new_blocks,
        )
        if "_" in new_tensor.keys.names:
            new_tensor = mts.remove_dimension(new_tensor, "keys", "_")
        new_tensors[name + "_character_projection"] = new_tensor
    return new_tensors


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
    """

    def __init__(
        self,
        base_model,
        max_o3_lambda_character: int,
        max_o3_lambda_target: int,
        batch_size: int = 32,
        max_o3_lambda_grid: Optional[int] = None,
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

    @torch.jit.ignore
    def _wigner_D_inverse_dict(self) -> Dict[int, torch.Tensor]:
        try:
            ref = next(self.base_model.parameters())
            device = ref.device
            dtype = ref.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.get_default_dtype()

        return {
            ell: tensor.to(device=device, dtype=dtype)
            for ell, tensor in _compute_real_wigner_matrices(
                self.max_o3_lambda_target, self._inverse_quadrature_angles
            ).items()
        }

    @property
    def wigner_D_inverse_rotations(self) -> Dict[int, torch.Tensor]:
        # Python-only nice view
        return self._wigner_D_inverse_dict()

    @torch.jit.ignore
    def _so3_characters_dict(self) -> Dict[int, torch.Tensor]:
        try:
            ref = next(self.base_model.parameters())
            device = ref.device
            dtype = ref.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.get_default_dtype()

        so3_characters, _ = compute_characters(
            self.max_o3_lambda_character,
            self._quadrature_angles,
            self._inverse_quadrature_angles,
        )
        return {ell: tensor.to(device=device, dtype=dtype) for ell, tensor in so3_characters.items()}

    @property
    def so3_characters(self) -> Dict[int, torch.Tensor]:
        # Python-only nice view
        return self._so3_characters_dict()

    @torch.jit.ignore
    def _pso3_characters_dict(self) -> Dict[str, torch.Tensor]:
        try:
            ref = next(self.base_model.parameters())
            device = ref.device
            dtype = ref.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.get_default_dtype()

        _, pso3_characters = compute_characters(
            self.max_o3_lambda_character,
            self._quadrature_angles,
            self._inverse_quadrature_angles,
        )
        return {
            label: tensor.to(device=device, dtype=dtype)
            for label, tensor in pso3_characters.items()
        }

    @property
    def pso3_characters(self) -> Dict[str, torch.Tensor]:
        # Python-only nice view
        return self._pso3_characters_dict()

    def _get_wigner_D_inverse(self, ell: int) -> torch.Tensor:
        return self.wigner_D_inverse_rotations[ell]

    def _get_so3_character(self, o3_lambda: int) -> torch.Tensor:
        return self.so3_characters[o3_lambda]

    def _get_pso3_character(self, o3_lambda: int, o3_sigma: int) -> torch.Tensor:
        return self.pso3_characters[str(o3_lambda) + "_" + str(o3_sigma)]

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

        # When not computing energy gradients (forces/stress), there is no
        # autograd graph to preserve, so we offload to CPU to save GPU memory.
        offload = not compute_gradients
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

        transformed_outputs = decompose_tensors(transformed_outputs, work_device)
        backtransformed_outputs = decompose_tensors(
            backtransformed_outputs, work_device
        )

        out_dict: Dict[str, TensorMap] = {}

        so3_weights = self.so3_weights.to(device=work_device)

        mean_var = symmetrize_over_grid(backtransformed_outputs, so3_weights)
        for name, tensor in mean_var.items():
            out_dict[name] = tensor

        if not project_tokens:
            return out_dict

        norms = compute_norm_per_property(transformed_outputs, so3_weights)
        for name, tensor in norms.items():
            out_dict[name] = tensor

        so3_chars = {k: v.to(device=work_device) for k, v in self.so3_characters.items()}
        pso3_chars = {
            k: v.to(device=work_device) for k, v in self.pso3_characters.items()
        }
        convolution_integrals = compute_conv_integral(
            transformed_outputs,
            so3_weights,
            so3_chars,
            pso3_chars,
            self.max_o3_lambda_character,
        )
        for name, integral in convolution_integrals.items():
            out_dict[name] = integral

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
        Sample the model on the O(3) quadrature.

        :param systems: list of systems to evaluate
        :param outputs: dictionary of model outputs to symmetrize
        :param selected_atoms: optional Labels specifying which atoms to consider
        :param return_transformed: if True, also return un-back-rotated outputs
        :param compute_gradients: if True, compute forces/stress via autograd
        :return: (transformed_outputs, backtransformed_outputs) dictionaries
        """

        results = evaluate_model_over_grid(
            self.base_model,
            self.batch_size,
            self.so3_rotations,
            self.so3_inverse_rotations,
            self._wigner_D_inverse_jit,
            return_transformed,
            systems,
            outputs,
            selected_atoms,
            compute_gradients=compute_gradients,
            offload_to_cpu=offload_to_cpu,
        )

        if return_transformed:
            transformed_outputs_tensor, backtransformed_outputs_tensor = results
        else:
            backtransformed_outputs_tensor = results
            transformed_outputs_tensor: Dict[str, TensorMap] = {}

        # TODO: possibly remove
        if "energy" in transformed_outputs_tensor:
            energy_tm = transformed_outputs_tensor["energy"]
            if "atom" in energy_tm[0].samples.names:
                # Sum over atoms while keeping system and rotation indices.
                energy_total_tm = mts.sum_over_samples(energy_tm, ["atom"])
                transformed_outputs_tensor["energy_total"] = energy_total_tm

        if "energy" in backtransformed_outputs_tensor:
            energy_tm_bt = backtransformed_outputs_tensor["energy"]
            if "atom" in energy_tm_bt[0].samples.names:
                energy_total_tm_bt = mts.sum_over_samples(energy_tm_bt, ["atom"])
                backtransformed_outputs_tensor["energy_total"] = energy_total_tm_bt
        return transformed_outputs_tensor, backtransformed_outputs_tensor


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
    out["forces"] = TensorMap(key_labels, [forces_block])

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


def evaluate_model_over_grid(
    model: ModelInterface,
    batch_size: int,
    so3_rotations: torch.Tensor,
    so3_rotations_inverse: torch.Tensor,
    wigner_D_inverse: Dict[int, torch.Tensor],
    return_transformed: bool,
    systems: List[System],
    outputs: Dict[str, ModelOutput],
    selected_atoms: Optional[Labels] = None,
    compute_gradients: bool = False,
    offload_to_cpu: bool = False,
) -> Dict[str, TensorMap] | Tuple[Dict[str, TensorMap], Dict[str, TensorMap]]:
    """
    Evaluate the model on rotated copies of the input systems over an O(3) quadrature
    grid, and optionally back-rotate the outputs.

    This function does **not** manage gradient context (``torch.no_grad`` etc.).
    Callers are responsible for wrapping in the appropriate context.

    :param model: atomistic model to evaluate
    :param batch_size: number of rotations to evaluate in a single batch
    :param so3_rotations: SO(3) rotation matrices, shape ``(N, 3, 3)``
    :param so3_rotations_inverse: inverse rotation matrices, shape ``(N, 3, 3)``
    :param wigner_D_inverse: Wigner D matrices for back-rotation, mapping l to tensor
    :param return_transformed: if True, also return un-back-rotated outputs
    :param systems: list of systems to evaluate
    :param outputs: dictionary of model outputs to compute
    :param selected_atoms: optional Labels specifying which atoms to consider
    :param compute_gradients: if True, compute conservative forces and stress via
        autograd on each rotated evaluation. Results are added as ``"forces"`` and
        ``"stress"`` keys (distinct from any ``"non_conservative_*"`` model outputs).
    :param offload_to_cpu: if True, move intermediate outputs to CPU after model
        evaluation to save GPU memory. Only safe when gradients are not needed
        downstream (set to False if computing gradients through decompose/symmetrize
        operations). Default False to preserve gradient flow.
    :return: back-rotated outputs, or (transformed, back-rotated) if
        ``return_transformed=True``
    """

    if compute_gradients and offload_to_cpu:
        raise ValueError(
            "Cannot offload to CPU when computing gradients (forces/stress via "
            "autograd). Set offload_to_cpu=False to preserve the autograd graph."
        )

    device = systems[0].positions.device
    dtype = systems[0].positions.dtype

    transformed_outputs: Dict[str, List[Dict[int, TensorMap]]] = {}
    output_names = list(outputs.keys())
    if compute_gradients:
        output_names = list(set(output_names + ["forces"]))
        if any(bool(torch.any(s.pbc).item()) for s in systems):
            output_names = list(set(output_names + ["stress"]))
    for name in output_names:
        lst: List[Dict[int, TensorMap]] = []
        for _ in systems:
            d: Dict[int, TensorMap] = {}
            lst.append(d)
        transformed_outputs[name] = lst
    for i_sys, system in enumerate(systems):
        for inversion in [-1, 1]:
            rotation_outputs: List[Dict[str, TensorMap]] = []

            if compute_gradients:
                # Process one rotation at a time for per-rotation autograd
                for R in so3_rotations:
                    rotation = inversion * R.to(device=device, dtype=dtype)
                    out = _evaluate_with_gradients(
                        model,
                        system,
                        rotation,
                        outputs,
                        selected_atoms,
                        device,
                        dtype,
                    )
                    rotation_outputs.append(out)
                effective_batch_size = 1
            else:
                for batch_start in range(0, len(so3_rotations), batch_size):
                    transformed_systems = [
                        _transform_system(
                            system,
                            inversion * R.to(device=device, dtype=dtype),
                        )
                        for R in so3_rotations[batch_start : batch_start + batch_size]
                    ]
                    out = model(
                        transformed_systems,
                        outputs,
                        selected_atoms,
                    )
                    if offload_to_cpu:
                        out = {k: v.to(device="cpu") for k, v in out.items()}
                    rotation_outputs.append(out)
                effective_batch_size = batch_size

            # Combine batch outputs
            for name in output_names:
                if name not in rotation_outputs[0]:
                    continue
                combined_: List[TensorMap] = [r[name] for r in rotation_outputs]
                combined = mts.join(
                    combined_,
                    "samples",
                    add_dimension="batch_rotation",
                )
                if "batch_rotation" in combined[0].samples.names:
                    blocks: List[TensorBlock] = []
                    for block in combined.blocks():
                        batch_id = block.samples.column("batch_rotation")
                        rot_id = block.samples.column("system")
                        new_sample_values = block.samples.values[:, :-1]
                        new_sample_values[:, 0] = (
                            batch_id * effective_batch_size + rot_id
                        )
                        blocks.append(
                            TensorBlock(
                                values=block.values,
                                samples=Labels(
                                    block.samples.names[:-1],
                                    new_sample_values,
                                ),
                                components=block.components,
                                properties=block.properties,
                            )
                        )
                    combined = TensorMap(combined.keys, blocks)
                transformed_outputs[name][i_sys][inversion] = combined

    # When offloading, move everything to CPU before backtransform so that
    # backtransform_outputs and to_metatensor operate on a single device.
    if offload_to_cpu:
        systems = [s.to(device="cpu") for s in systems]
        so3_rotations_inverse = so3_rotations_inverse.to(device="cpu")
        wigner_D_inverse = {k: v.to(device="cpu") for k, v in wigner_D_inverse.items()}

    backtransformed_outputs = backtransform_outputs(
        transformed_outputs, systems, so3_rotations_inverse, wigner_D_inverse
    )
    backtransformed_outputs_tensor = to_metatensor(backtransformed_outputs, systems)

    if return_transformed:
        transformed_outputs_tensor = to_metatensor(transformed_outputs, systems)
        return transformed_outputs_tensor, backtransformed_outputs_tensor
    else:
        transformed_outputs_tensor: Dict[str, TensorMap] = {}
        return backtransformed_outputs_tensor


def to_metatensor(
    tensor_dict: Dict[str, TensorMap], systems: List[System]
) -> Dict[str, TensorMap]:
    """
    Convert the outputs of the model evaluated on rotated systems to a single
    TensorMap per property, with appropriate dimensions for O(3) symmetrization.
    """

    out_tensor_dict: Dict[str, TensorMap] = {}
    # Massage outputs to have desired shape
    for name in tensor_dict:
        joined_plus = mts.join(
            [tensor_dict[name][i_sys][1] for i_sys in range(len(systems))],
            "samples",
            add_dimension="phys_system",
        )
        joined_minus = mts.join(
            [tensor_dict[name][i_sys][-1] for i_sys in range(len(systems))],
            "samples",
            add_dimension="phys_system",
        )
        joined = mts.join(
            [
                mts.append_dimension(joined_plus, "keys", "inversion", 1),
                mts.append_dimension(joined_minus, "keys", "inversion", -1),
            ],
            "samples",
            different_keys="union",
        )
        joined = mts.rename_dimension(joined, "samples", "system", "so3_rotation")

        if "phys_system" in joined[0].samples.names:
            joined = mts.rename_dimension(joined, "samples", "phys_system", "system")
        else:
            joined = mts.insert_dimension(
                joined,
                "samples",
                1,
                "system",
                torch.zeros(
                    joined[0].samples.values.shape[0],
                    dtype=torch.long,
                    device=joined[0].samples.values.device,
                ),
            )
        if "atom" in joined[0].samples.names or "first_atom" in joined[0].samples.names:
            perm = _permute_system_before_atom(joined[0].samples.names)
            joined = mts.permute_dimensions(joined, "samples", perm)
        out_tensor_dict[name] = joined

    return out_tensor_dict


def backtransform_outputs(
    tensor_dict: Dict[str, List[Dict[int, TensorMap]]],
    systems: List[System],
    so3_rotations_inverse: torch.Tensor,
    wigner_D_inverse: Dict[int, torch.Tensor],
) -> Dict[str, List[Dict[int, TensorMap]]]:
    """
    Given the outputs of the model evaluated on rotated systems, backtransform them to
    the original frame according to the equivariance labels in the TensorMap keys.
    """

    device = systems[0].positions.device
    dtype = systems[0].positions.dtype

    backtransformed_tensor_dict: Dict[str, List[Dict[int, TensorMap]]] = {}
    for name in tensor_dict:
        lst: List[Dict[int, TensorMap]] = []
        for _ in systems:
            d: Dict[int, TensorMap] = {}
            lst.append(d)
        backtransformed_tensor_dict[name] = lst

    n_rot = so3_rotations_inverse.size(0)
    for name in tensor_dict:
        for i_sys, system in enumerate(systems):
            for inversion in [-1, 1]:
                tensor = tensor_dict[name][i_sys][inversion]
                wigner_dict: Dict[int, List[torch.Tensor]] = {}
                for ell in wigner_D_inverse:
                    wigner_dict[ell] = (
                        wigner_D_inverse[ell].to(device=device, dtype=dtype).unbind(0)
                    )

                _, backtransformed, _ = _apply_augmentations(
                    [system] * n_rot,
                    {name: tensor},
                    list(
                        (
                            so3_rotations_inverse.to(device=device, dtype=dtype)
                            * inversion
                        ).unbind(0)
                    ),
                    wigner_dict,
                )
                backtransformed_tensor_dict[name][i_sys][inversion] = backtransformed[
                    name
                ]
    return backtransformed_tensor_dict


def _permute_system_before_atom(labels: List[str]) -> List[int]:
    # find positions
    sys_idx = -1
    atom_idx = -1
    for i in range(len(labels)):
        if labels[i] == "system":
            sys_idx = i
        elif labels[i] == "atom":
            atom_idx = i
        elif labels[i] == "first_atom":
            atom_idx = i

    # identity permutation
    perm = list(range(len(labels)))

    # reorder only if both present and system is after atom
    if sys_idx != -1 and atom_idx != -1 and sys_idx > atom_idx:
        v = perm[sys_idx]
        # remove system
        for k in range(sys_idx, len(perm) - 1):
            perm[k] = perm[k + 1]
        perm.pop()
        # insert before atom
        perm.insert(atom_idx, v)

    return perm


def symmetrize_over_grid(
    tensor_dict: Dict[str, TensorMap],
    so3_weights: torch.Tensor,
) -> Dict[str, TensorMap]:
    """
    Compute the mean and variance of the outputs over O(3).

    :param tensor_dict: dictionary of TensorMaps with rotated and backtransformed
        outputs to compute mean, variance, and norm squared for
    :param so3_weights: weights of the SO(3) quadrature
    :return: dictionary of TensorMaps with mean, variance, and norm squared
    """
    mean_var: Dict[str, TensorMap] = {}
    for name in tensor_dict:
        # cannot compute a mean or variance as these have no known behaviour under
        # rotations
        if "features" in name:
            continue
        tensor = tensor_dict[name]
        mean_blocks: List[TensorBlock] = []
        second_moment_blocks: List[TensorBlock] = []
        for block in tensor.blocks():
            rot_ids = block.samples.column("so3_rotation")
            values_squared = _component_norm_squared(block.values)

            weighted_values = _weight_by_quadrature(
                so3_weights, rot_ids, block.values
            )
            weighted_sq = _weight_by_quadrature(
                so3_weights, rot_ids, values_squared
            )

            mean_blocks.append(
                TensorBlock(
                    values=weighted_values,
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
            )
            second_moment_blocks.append(
                TensorBlock(
                    values=weighted_sq,
                    samples=block.samples,
                    components=[],
                    properties=block.properties,
                )
            )

        # Mean
        tensor_mean = TensorMap(tensor.keys, mean_blocks)
        tensor_mean = mts.sum_over_samples(
            tensor_mean.keys_to_samples("inversion"), ["inversion", "so3_rotation"]
        )

        # ||<x>||^2
        mean_norm_squared_blocks: List[TensorBlock] = []
        for block in tensor_mean.blocks():
            mean_norm_squared_blocks.append(
                TensorBlock(
                    values=_component_norm_squared(block.values),
                    samples=block.samples,
                    components=[],
                    properties=block.properties,
                )
            )
        tensor_mean_norm_squared = TensorMap(tensor_mean.keys, mean_norm_squared_blocks)

        # Second moment
        tensor_second_moment = TensorMap(tensor.keys, second_moment_blocks)
        tensor_second_moment = mts.sum_over_samples(
            tensor_second_moment.keys_to_samples("inversion"),
            ["inversion", "so3_rotation"],
        )

        # Variance
        tensor_variance = mts.subtract(tensor_second_moment, tensor_mean_norm_squared)

        mean_var[name + "_mean"] = tensor_mean
        mean_var[name + "_norm_squared"] = tensor_second_moment
        mean_var[name + "_var"] = tensor_variance
    return mean_var
