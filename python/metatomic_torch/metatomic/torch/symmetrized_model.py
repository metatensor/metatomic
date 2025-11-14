from typing import Dict, List, Optional, Tuple

import metatensor.torch as mts
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatrain.utils.augmentation import _apply_augmentations

from metatomic.torch import ModelOutput, System, register_autograd_neighbors


try:
    from scipy.integrate import lebedev_rule  # noqa: F401
    from scipy.spatial.transform import Rotation  # noqa: F401
except ImportError as e:
    raise ImportError(
        "To perform data augmentation on spherical targets, please "
        "install the `scipy` package with `pip install scipy`."
    ) from e
try:
    import spherical  # noqa: F401
except ImportError as e:
    raise ImportError(
        "To perform data augmentation on spherical targets, please "
        "install the `spherical` package with `pip install spherical`."
    ) from e
try:
    import quaternionic  # noqa: F401
except ImportError as e:
    raise ImportError(
        "To perform data augmentation on spherical targets, please "
        "install the `quaternionic` package with `pip install quaternionic`."
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
        Rotation.from_euler("z", alpha)
        * Rotation.from_euler("y", beta)
        * Rotation.from_euler("z", gamma)
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
    wigner = spherical.Wigner(o3_lambda_max)
    R = quaternionic.array.from_euler_angles(*angles)
    D = wigner.D(R)
    wigner_D_matrices = {}
    for ell in range(o3_lambda_max + 1):
        wigner_D_matrices[ell] = np.zeros(
            angles[0].shape + (2 * ell + 1, 2 * ell + 1), dtype=np.complex128
        )
        for mp in range(-ell, ell + 1):
            for m in range(-ell, ell + 1):
                # There is an unexplained conjugation factor in the definition given in
                # the quaternionic library.
                wigner_D_matrices[ell][..., mp + ell, m + ell] = (
                    D[..., wigner.Dindex(ell, mp, m)]
                ).conj()
        U = _complex_to_real_spherical_harmonics_transform(ell)
        wigner_D_matrices[ell] = np.einsum(
            "ij,...jk,kl->...il", U.conj(), wigner_D_matrices[ell], U.T
        )
        assert np.allclose(wigner_D_matrices[ell].imag, 0)
        wigner_D_matrices[ell] = torch.from_numpy(wigner_D_matrices[ell].real)

    return wigner_D_matrices


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
    R01 = R_flat[:, 0, 1]
    R02 = R_flat[:, 0, 2]
    R10 = R_flat[:, 1, 0]
    R11 = R_flat[:, 1, 1]
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


def _get_o3_character(
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    o3_lambda: int,
    o3_sigma: int,
    tol: float = 1e-13,
) -> np.ndarray:
    """
    Numerically stable evaluation of the character function χ_{o3_lambda}(R) over O(3).
    """
    return (
        o3_sigma
        * ((-1) ** o3_lambda)
        * _get_so3_character(alphas, betas, gammas, o3_lambda, tol)
    )


def compute_characters(
    o3_lambda_max: int,
    angles: Tuple[np.ndarray, np.ndarray, np.ndarray],
    inverse_angles: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict[int, torch.Tensor]:
    alpha, beta, gamma = _euler_angles_of_combined_rotation(angles, inverse_angles)

    so3_characters = {
        o3_lambda: _get_so3_character(alpha, beta, gamma, o3_lambda)
        for o3_lambda in range(o3_lambda_max + 1)
    }

    pso3_characters = {}
    for o3_lambda in range(o3_lambda_max + 1):
        for o3_sigma in [-1, +1]:
            pso3_characters[(o3_lambda, o3_sigma)] = (
                o3_sigma * ((-1) ** o3_lambda) * so3_characters[o3_lambda]
            )

    so3_characters = {
        key: torch.from_numpy(value) for key, value in so3_characters.items()
    }
    pso3_characters = {
        key: torch.from_numpy(value) for key, value in pso3_characters.items()
    }

    return so3_characters, pso3_characters


def _integrate_with_character(
    tensor_so3: torch.Tensor,
    tensor_pso3: torch.Tensor,
    so3_characters: Dict[int, torch.Tensor],
    pso3_characters: Dict[Tuple[int, int], torch.Tensor],
    o3_lambda_max: int,
):
    integral = {}
    for o3_lambda in range(o3_lambda_max + 1):
        so3_character = so3_characters[o3_lambda]
        for o3_sigma in [-1, 1]:
            pso3_character = pso3_characters[o3_lambda, o3_sigma]
            integral[o3_lambda, o3_sigma] = (1 / 4) * (
                torch.einsum(
                    "i...,i...->...",
                    tensor_so3,
                    torch.einsum("ij,j...->i...", so3_character, tensor_so3),
                )
                + torch.einsum(
                    "i...,i...->...",
                    tensor_pso3,
                    torch.einsum("ij,j...->i...", pso3_character, tensor_pso3),
                )
            ) + (1 / 2) * (
                torch.einsum(
                    "i...,i...->...",
                    tensor_so3,
                    torch.einsum("ij,j...->i...", pso3_character, tensor_pso3),
                )
            )

            # Normalize by Haar measure
            integral[(o3_lambda, o3_sigma)] *= (2 * o3_lambda + 1) / (
                8 * torch.pi**2
            ) ** 2
    return integral


class SymmetrizedModel(torch.nn.Module):
    def __init__(self, base_model, max_o3_lambda, batch_size: int = 32):
        super().__init__()
        self.base_model = base_model
        self.max_o3_lambda = max_o3_lambda
        self.batch_size = batch_size

        # Compute grid
        lebedev_order, n_inplane_rotations = _choose_quadrature(self.max_o3_lambda)
        alpha, beta, gamma, w_so3 = get_euler_angles_quadrature(
            lebedev_order, n_inplane_rotations
        )
        self.so3_weights = torch.from_numpy(w_so3)

        # Active rotations
        self.so3_rotations = torch.from_numpy(
            _rotations_from_angles(alpha, beta, gamma).as_matrix()
        )
        self.n_so3_rotations = self.so3_rotations.size(0)

        # Compute inverse Wigner D representations
        angles_inverse_rotations = (np.pi - gamma, beta, np.pi - alpha)
        self.so3_inverse_rotations = torch.from_numpy(
            _rotations_from_angles(*angles_inverse_rotations).as_matrix()
        )
        self.wigner_D_inverse_rotations = _compute_real_wigner_matrices(
            self.max_o3_lambda, angles_inverse_rotations
        )

        # Compute characters
        self.so3_characters, self.pso3_characters = compute_characters(
            self.max_o3_lambda,
            (alpha, beta, gamma),
            angles_inverse_rotations,
        )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        # Evaluate the model over the grid
        transformed_outputs, backtransformed_outputs = self._eval_over_grid(
            systems, outputs, selected_atoms
        )

        mean_var = self._compute_mean_and_variance(backtransformed_outputs)
        character_projections = self._compute_character_projections(
            transformed_outputs, mean_var, systems
        )
        return mean_var, character_projections

    def _compute_character_projections(self, transformed_outputs, mean_var, systems):
        integrals = {}
        for name in transformed_outputs:
            integrals[name] = []
            for i_sys, tensor_dict in enumerate(transformed_outputs[name]):
                integrals[name].append({})
                for (key, block_so3), block_pso3 in zip(
                    tensor_dict[1].items(),
                    tensor_dict[-1],
                    strict=True,
                ):
                    split_by_transformation = torch.bincount(
                        block_so3.samples.values[:, 0]
                    )
                    w = torch.repeat_interleave(
                        self.so3_weights, split_by_transformation
                    )
                    w = w.view(w.shape[0], *[1] * (block_so3.values.ndim - 1))

                    integral = _integrate_with_character(
                        block_so3.values * w,
                        block_pso3.values * w,
                        self.so3_characters,
                        self.pso3_characters,
                        self.max_o3_lambda,
                    )
                    key_dict = tuple(int(k) for k in key.values)
                    integrals[name][i_sys][key_dict] = integral

        tensors = {}
        for name in integrals:
            tensors[name] = []
            original_keys = mean_var[name + "_mean"].keys
            sample_names = mean_var[name + "_mean"][0].samples.names
            for i_sys, integral_per_system in enumerate(integrals[name]):
                if "atom" in sample_names:
                    samples = torch.cartesian_prod(
                        torch.tensor([i_sys]),
                        torch.arange(len(systems[i_sys].positions)),
                    )
                else:
                    samples = torch.tensor([[i_sys]])
                blocks = {}
                for old_key, integral_dict in integral_per_system.items():
                    for new_key, integral_values in integral_dict.items():
                        full_key = old_key + new_key
                        blocks[full_key] = integral_values
                blocks = TensorMap(
                    Labels(
                        original_keys.names + ["ell", "sigma"],
                        torch.tensor(list(blocks.keys())),
                    ),
                    [
                        TensorBlock(
                            values=blocks[key].unsqueeze(0),
                            samples=Labels(sample_names, samples),
                            components=mean_var[name + "_mean"].block(
                                {_k: key[i] for i, _k in enumerate(original_keys.names)}
                            )
                            # .block({"o3_lambda": key[0], "o3_sigma": key[1]})
                            .components,
                            properties=mean_var[name + "_mean"]
                            .block(
                                {_k: key[i] for i, _k in enumerate(original_keys.names)}
                            )
                            .properties,
                        )
                        for key in blocks
                    ],
                )
                tensors[name].append(blocks)
            tensors[name] = mts.join(tensors[name], "samples")

        return tensors

    def _compute_mean_and_variance(
        self, tensor_dict: Dict[str, TensorMap], contract_components: Dict[str, bool]
    ) -> Tuple[Dict[str, TensorMap], Dict[str, TensorMap]]:
        mean_var = {}
        for name in contract_components:
            tensor = tensor_dict[name]
            mean_blocks = []
            second_moment_blocks = []
            mean_norm_blocks = []
            for block in tensor:
                rot_ids = block.samples.column("so3_rotation")

                values = block.values
                values_norm = (
                    torch.norm(values, dim=tuple(range(1, values.ndim - 1)))
                    if values.ndim > 2
                    else torch.abs(values)
                )
                values_squared = values_norm**2

                view = (values.size(0), *[1] * (values.ndim - 1))
                values = 0.5 * self.so3_weights[rot_ids].view(view) * values

                view = (values_squared.size(0), *[1] * (values_squared.ndim - 1))
                values_squared = (
                    0.5 * self.so3_weights[rot_ids].view(view) * values_squared
                )

                view = (values_norm.size(0), *[1] * (values_norm.ndim - 1))
                values_norm = 0.5 * self.so3_weights[rot_ids].view(view) * values_norm

                mean_blocks.append(
                    TensorBlock(
                        values=values,
                        samples=block.samples,
                        components=block.components,
                        properties=block.properties,
                    )
                )
                mean_norm_blocks.append(
                    TensorBlock(
                        values=values_norm,
                        samples=block.samples,
                        components=[],
                        properties=block.properties,
                    )
                )
                second_moment_blocks.append(
                    TensorBlock(
                        values=values_squared,
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

            # Mean norm
            tensor_mean_norm = TensorMap(tensor.keys, mean_norm_blocks)
            tensor_mean_norm = mts.sum_over_samples(
                tensor_mean_norm.keys_to_samples("inversion"),
                ["inversion", "so3_rotation"],
            )

            # Second moment
            tensor_second_moment = TensorMap(tensor.keys, second_moment_blocks)
            tensor_second_moment = mts.sum_over_samples(
                tensor_second_moment.keys_to_samples("inversion"),
                ["inversion", "so3_rotation"],
            )

            # Variance
            tensor_variance = mts.subtract(
                tensor_second_moment, mts.pow(tensor_mean_norm, 2)
            )

            mean_var[name + "_mean"] = tensor_mean
            mean_var[name + "_mean_norm"] = tensor_mean_norm
            mean_var[name + "_var"] = tensor_variance
        return mean_var

    def _eval_over_grid(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
        return_tensormaps: bool = True,
    ):
        """
        Sample the model on the O(3) quadrature.

        :param systems: list of systems to evaluate
        :param model: atomistic model to evaluate
        :param device: device to use for computation
        :return: list of list of model outputs, shape (len(systems), N)
            where N is the number of quadrature points
        """

        device = systems[0].positions.device
        dtype = systems[0].positions.dtype

        transformed_outputs: Dict[str, List[Dict[int, Optional[TensorMap]]]] = {
            name: [{-1: None, 1: None} for _ in systems] for name in outputs
        }
        backtransformed_outputs: Dict[str, List[Dict[int, Optional[TensorMap]]]] = {
            name: [{-1: None, 1: None} for _ in systems] for name in outputs
        }
        for i_sys, system in enumerate(systems):
            for inversion in [-1, 1]:
                rotation_outputs: List[Dict[str, TensorMap]] = []
                for batch in range(0, len(self.so3_rotations), self.batch_size):
                    transformed_systems = [
                        _transform_system(
                            system, inversion * R.to(device=device, dtype=dtype)
                        )
                        for R in self.so3_rotations[batch : batch + self.batch_size]
                    ]
                    out = self.base_model(
                        transformed_systems,
                        outputs,
                        selected_atoms,
                    )
                    rotation_outputs.append(out)

                # Combine batch outputs
                for name in transformed_outputs:
                    combined: List[TensorMap] = [r[name] for r in rotation_outputs]
                    combined = mts.join(
                        combined,
                        "samples",
                        add_dimension="batch_rotation",
                    )
                    if "batch_rotation" in combined[0].samples.names:
                        # Reindex
                        blocks = []
                        for block in combined:
                            batch_id = block.samples.column("batch_rotation")
                            rot_id = block.samples.column("system")
                            new_sample_values = block.samples.values[:, :-1]
                            new_sample_values[:, 0] = (
                                batch_id * self.batch_size + rot_id
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

        n_rot = self.so3_rotations.size(0)
        for name in transformed_outputs:
            for i_sys, system in enumerate(systems):
                for inversion in [-1, 1]:
                    tensor = transformed_outputs[name][i_sys][inversion]
                    _, backtransformed, _ = _apply_augmentations(
                        [system] * n_rot,
                        {name: tensor},
                        list(
                            (
                                self.so3_inverse_rotations.to(
                                    device=device, dtype=dtype
                                )
                                * inversion
                            ).unbind(0)
                        ),
                        {
                            ell: self.wigner_D_inverse_rotations[ell]
                            .to(device=device, dtype=dtype)
                            .unbind(0)
                            for ell in self.wigner_D_inverse_rotations
                        },
                    )
                    backtransformed_outputs[name][i_sys][inversion] = backtransformed[
                        name
                    ]

        if return_tensormaps:
            # Massage outputs to have desired shape
            for name in transformed_outputs:
                joined_plus = mts.join(
                    [
                        transformed_outputs[name][i_sys][1]
                        for i_sys in range(len(systems))
                    ],
                    "samples",
                    add_dimension="phys_system",
                )
                joined_minus = mts.join(
                    [
                        transformed_outputs[name][i_sys][1]
                        for i_sys in range(len(systems))
                    ],
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
                joined = mts.rename_dimension(
                    joined, "samples", "system", "so3_rotation"
                )

                if "phys_system" in joined[0].samples.names:
                    joined = mts.rename_dimension(
                        joined, "samples", "phys_system", "system"
                    )
                else:
                    joined = mts.insert_dimension(
                        joined,
                        "samples",
                        0,
                        "system",
                        torch.zeros(
                            joined[0].samples.values.shape[0], dtype=torch.long
                        ),
                    )
                transformed_outputs[name] = mts.permute_dimensions(
                    joined,
                    "samples",
                    [
                        len(joined[0].samples.names) - 1,
                        *range(len(joined[0].samples.names) - 1),
                    ],
                )

                joined_plus = mts.join(
                    [
                        backtransformed_outputs[name][i_sys][1]
                        for i_sys in range(len(systems))
                    ],
                    "samples",
                    add_dimension="phys_system",
                )
                joined_minus = mts.join(
                    [
                        backtransformed_outputs[name][i_sys][1]
                        for i_sys in range(len(systems))
                    ],
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
                joined = mts.rename_dimension(
                    joined, "samples", "system", "so3_rotation"
                )
                if "phys_system" in joined[0].samples.names:
                    joined = mts.rename_dimension(
                        joined, "samples", "phys_system", "system"
                    )
                else:
                    joined = mts.insert_dimension(
                        joined,
                        "samples",
                        0,
                        "system",
                        torch.zeros(
                            joined[0].samples.values.shape[0], dtype=torch.long
                        ),
                    )
                backtransformed_outputs[name] = mts.permute_dimensions(
                    joined,
                    "samples",
                    [
                        len(joined[0].samples.names) - 1,
                        *range(len(joined[0].samples.names) - 1),
                    ],
                )

        return transformed_outputs, backtransformed_outputs

    # def _compute_mean_and_variance(self, backtransformed_outputs):
    #     mean_var_outputs: Dict[str, TensorMap] = {}
    #     # Iterate over targets
    #     for target_name in backtransformed_outputs:
    #         mean_tensors: List[TensorMap] = []
    #         var_tensors: List[TensorMap] = []
    #         # Iterate over systems
    #         for i_sys in range(len(backtransformed_outputs[target_name])):
    #             tensor_so3 = backtransformed_outputs[target_name][i_sys][1]
    #             tensor_pso3 = backtransformed_outputs[target_name][i_sys][-1]

    #             mean_blocks: List[TensorBlock] = []
    #             var_blocks: List[TensorBlock] = []
    #             # Iterate over blocks
    #             for block_so3, block_pso3 in zip(tensor_so3, tensor_pso3, strict=True):
    #                 split_by_transformation = torch.bincount(
    #                     block_so3.samples.values[:, 0]
    #                 )
    #                 w = torch.repeat_interleave(
    #                     self.so3_weights, split_by_transformation
    #                 )
    #                 w = w.view(w.shape[0], *[1] * (block_so3.values.ndim - 1))
    #                 mean_block = (block_so3.values + block_pso3.values) * 0.5 * w
    #                 second_moment_block = (
    #                     (block_so3.values**2 + block_pso3.values**2) * 0.5 * w
    #                 )
    #                 mean_blocks.append(
    #                     TensorBlock(
    #                         samples=block_so3.samples,
    #                         components=block_so3.components,
    #                         properties=block_so3.properties,
    #                         values=mean_block,
    #                     )
    #                 )
    #                 var_blocks.append(
    #                     TensorBlock(
    #                         samples=block_so3.samples,
    #                         components=block_so3.components,
    #                         properties=block_so3.properties,
    #                         values=second_moment_block,
    #                     )
    #                 )
    #             mean_tensor = mts.sum_over_samples(
    #                 TensorMap(tensor_so3.keys, mean_blocks), "system"
    #             )
    #             second_moment_tensor = mts.sum_over_samples(
    #                 TensorMap(tensor_so3.keys, var_blocks), "system"
    #             )
    #             var_tensor = mts.subtract(second_moment_tensor, mts.pow(mean_tensor, 2))
    #             mean_tensors.append(mean_tensor)
    #             var_tensors.append(var_tensor)

    #         mean = mts.join(mean_tensors, "samples", add_dimension="system")
    #         var = mts.join(var_tensors, "samples", add_dimension="system")

    #         if "system" not in mean[0].samples.names:
    #             mean = mts.insert_dimension(
    #                 mean,
    #                 "samples",
    #                 0,
    #                 "system",
    #                 torch.zeros(mean[0].samples.values.shape[0], dtype=torch.long),
    #             )
    #             var = mts.insert_dimension(
    #                 var,
    #                 "samples",
    #                 0,
    #                 "system",
    #                 torch.zeros(var[0].samples.values.shape[0], dtype=torch.long),
    #             )
    #         else:
    #             num_dims = len(mean[0].samples.names)
    #             mean = mts.permute_dimensions(
    #                 mean,
    #                 "samples",
    #                 [num_dims - 1] + list(range(num_dims - 1)),
    #             )
    #             var = mts.permute_dimensions(
    #                 var,
    #                 "samples",
    #                 [num_dims - 1] + list(range(num_dims - 1)),
    #             )
    #         if "_" in mean[0].samples.names:
    #             mean = mts.remove_dimension(mean, "samples", "_")
    #             var = mts.remove_dimension(var, "samples", "_")

    #         # Store results
    #         mean_var_outputs[target_name + "_mean"] = mean
    #         ncomp = len(var[0].components)
    #         var = TensorMap(
    #             var.keys,
    #             [
    #                 TensorBlock(
    #                     samples=block.samples,
    #                     components=[],
    #                     properties=block.properties,
    #                     values=block.values.sum(dim=list(range(1, ncomp + 1))),
    #                 )
    #                 for block in var
    #             ],
    #         )
    #         mean_var_outputs[target_name + "_var"] = var

    #     return mean_var_outputs

    # def _eval_over_grid(
    #     self,
    #     systems: List[System],
    #     outputs: Dict[str, ModelOutput],
    #     selected_atoms: Optional[Labels],
    # ):
    #     """
    #     Sample the model on the O(3) quadrature.

    #     :param systems: list of systems to evaluate
    #     :param model: atomistic model to evaluate
    #     :param device: device to use for computation
    #     :return: list of list of model outputs, shape (len(systems), N)
    #         where N is the number of quadrature points
    #     """

    #     device = systems[0].positions.device
    #     dtype = systems[0].positions.dtype

    #     transformed_outputs: Dict[str, List[Dict[int, Optional[TensorMap]]]] = {
    #         name: [{-1: None, 1: None} for _ in systems] for name in outputs
    #     }
    #     backtransformed_outputs: Dict[str, List[Dict[int, Optional[TensorMap]]]] = {
    #         name: [{-1: None, 1: None} for _ in systems] for name in outputs
    #     }
    #     for i_sys, system in enumerate(systems):
    #         for inversion in [-1, 1]:
    #             rotation_outputs: List[Dict[str, TensorMap]] = []
    #             for batch in range(0, len(self.so3_rotations), self.batch_size):
    #                 transformed_systems = [
    #                     _transform_system(
    #                         system, inversion * R.to(device=device, dtype=dtype)
    #                     )
    #                     for R in self.so3_rotations[batch : batch + self.batch_size]
    #                 ]
    #                 out = self.base_model(
    #                     transformed_systems,
    #                     outputs,
    #                     selected_atoms,
    #                 )
    #                 rotation_outputs.append(out)

    #             # Combine batch outputs
    #             for name in transformed_outputs:
    #                 combined: List[TensorMap] = [r[name] for r in rotation_outputs]
    #                 transformed_outputs[name][i_sys][inversion] = mts.join(
    #                     combined,
    #                     "samples",
    #                     add_dimension="batch_rotation",
    #                 )

    #     n_rot = self.so3_rotations.size(0)
    #     for name in transformed_outputs:
    #         for i_sys, system in enumerate(systems):
    #             for inversion in [-1, 1]:
    #                 tensor = transformed_outputs[name][i_sys][inversion]
    #                 _, backtransformed, _ = _apply_augmentations(
    #                     [system] * n_rot,
    #                     {name: tensor},
    #                     list(
    #                         (
    #                             self.so3_inverse_rotations.to(
    #                                 device=device, dtype=dtype
    #                             )
    #                             * inversion
    #                         ).unbind(0)
    #                     ),
    #                     {
    #                         ell: self.wigner_D_inverse_rotations[ell]
    #                         .to(device=device, dtype=dtype)
    #                         .unbind(0)
    #                         for ell in self.wigner_D_inverse_rotations
    #                     },
    #                 )
    #                 backtransformed_outputs[name][i_sys][inversion] = backtransformed[
    #                     name
    #                 ]

    #     return transformed_outputs, backtransformed_outputs
