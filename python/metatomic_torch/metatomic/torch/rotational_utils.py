"""
Utilities for diagnosing rotational equivariance of models and for enforcing
rotational symmetry in data augmentation and model evaluation.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import metatensor.torch as mts
import numpy as np
import torch
from metatensor.torch import TensorMap
from metatrain.utils.augmentation import (
    _apply_random_augmentations,
    _complex_to_real_spherical_harmonics_transform,
    _scipy_quaternion_to_quaternionic,
)

from metatomic.torch import ModelEvaluationOptions, System, register_autograd_neighbors
from metatomic.torch.model import AtomisticModel


try:
    from scipy.spatial.transform import Rotation
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
    K = 2 * L_max + 1
    return n, K


def get_euler_angles_quadrature(lebedev_order: int, n_rotations: int):
    """
    Get the Euler angles and weights for Lebedev quadrature.

    :param lebedev_order: order of the Lebedev quadrature on the unit sphere
    :param n_rotations: number of in-plane rotations per Lebedev node
    :return: alpha, beta, gamma, w arrays of shape (M,), (M,), (K,), (M,)
        respectively, where M is the number of Lebedev nodes and K is the number of
        in-plane rotations.
    """
    from scipy.integrate import lebedev_rule

    # Lebedev nodes (X: (3, M))
    X, w = lebedev_rule(lebedev_order)  # w sums to 4*pi
    x, y, z = X
    alpha = np.arctan2(y, x)  # (M,)
    beta = np.arccos(np.clip(z, -1.0, 1.0))  # (M,)
    gamma = np.linspace(0.0, 2 * np.pi, n_rotations, endpoint=False)  # (n_rotations,)

    w_so3 = np.repeat(w / (4 * np.pi * n_rotations), repeats=gamma.size)  # (N,)

    return alpha, beta, gamma, w_so3


def _rotations_from_angles(alpha, beta, gamma):
    from scipy.spatial.transform import Rotation

    # Build all combinations (alpha_i, beta_i, gamma_j)
    A = np.repeat(alpha, gamma.size)  # (N,)
    B = np.repeat(beta, gamma.size)  # (N,)
    G = np.tile(gamma, alpha.size)  # (N,)

    # Compose ZYZ rotations in SO(3)
    Rot = (
        Rotation.from_euler("z", A)
        * Rotation.from_euler("y", B)
        * Rotation.from_euler("z", G)
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


def evaluate_model_on_quadrature(model, systems, L_max: int, device="cpu"):
    pass


############


def _extract_euler_zyz(
    R: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    zz = torch.clip(R22, -1.0, 1.0)
    betas = torch.arccos(zz)

    # For Z-Y-Z, standard formulas away from the singular set
    alphas = torch.arctan2(R12, R02)
    gammas = torch.arctan2(R21, -R20)

    # Normalize into [0, 2π)
    two_pi = 2.0 * torch.pi
    alphas = torch.remainder(alphas, two_pi)
    gammas = torch.remainder(gammas, two_pi)

    # Gimbal-lock detection via sin(beta)
    sinb = torch.sin(betas)
    near = torch.abs(sinb) < eps
    if torch.any(near):
        # Split the two singular bands using zz = cos(beta)
        near_zero = near & (zz > 0)  # beta≈0
        near_pi = near & (zz < 0)  # beta≈pi

        if torch.any(near_zero):
            # beta≈0: rotation ≈ Rz(alpha+gamma). Choose gamma=0, recover alpha from 2x2 block.
            betas[near_zero] = 0.0
            gammas[near_zero] = 0.0
            alphas[near_zero] = torch.arctan2(R10[near_zero], R00[near_zero])
            alphas[near_zero] = torch.remainder(alphas[near_zero], two_pi)

        if torch.any(near_pi):
            # beta≈pi: choose alpha=0, recover gamma from 2x2 block with sign flip on R00.
            betas[near_pi] = torch.pi
            alphas[near_pi] = 0.0
            gammas[near_pi] = torch.arctan2(R10[near_pi], -R00[near_pi])
            gammas[near_pi] = torch.remainder(gammas[near_pi], two_pi)

    # Unflatten back to the original batch shape
    alphas = alphas.reshape(batch_shape)
    betas = betas.reshape(batch_shape)
    gammas = gammas.reshape(batch_shape)
    return alphas, betas, gammas


def get_so3_character(
    alphas: torch.Tensor,
    betas: torch.Tensor,
    gammas: torch.Tensor,
    o3_lambda: int,
    tol: float = 1e-7,
) -> torch.Tensor:
    """
    Numerically stable evaluation of the character function χ_{o3_lambda}(R) over SO(3).

    Uses a small-angle Taylor expansion for χ_ℓ(ω) = sin((2ℓ+1)t)/sin(t) with t = ω/2
    when |t| is very small, and a guarded ratio otherwise.
    """
    # Compute half-angle t = ω/2 via Z–Y–Z relation: cos t = cos(β/2) cos((α+γ)/2)
    cos_t = torch.cos(betas / 2.0) * torch.cos((alphas + gammas) / 2.0)
    cos_t = torch.clip(cos_t, -1.0, 1.0)
    t = torch.arccos(cos_t)

    # Output array
    chi = torch.empty_like(t)

    # Parameters for χ
    L = o3_lambda
    a = 2 * L + 1
    ll1 = L * (L + 1)

    small = torch.abs(t) < tol
    if torch.any(small):
        # Series up to t^4: χ ≈ a [1 - (2/3) ℓ(ℓ+1) t^2 + (1/45) ℓ(ℓ+1)(3ℓ^2+3ℓ-1) t^4]
        ts = t[small]
        t2 = ts * ts
        coeff4 = ll1 * (3 * L * L + 3 * L - 1)
        chi[small] = a * (
            1.0 - (2.0 / 3.0) * ll1 * t2 + (1.0 / 45.0) * coeff4 * t2 * t2
        )

    # Large-angle (or not-so-small) branch: safe ratio with guard
    large = ~small
    if torch.any(large):
        tl = t[large]
        sin_t = torch.sin(tl)
        numer = torch.sin(a * tl)
        mask = torch.abs(sin_t) >= tol
        out = torch.empty_like(tl)
        torch.div(numer, sin_t, out=out)  # TODO figure out with numpy divide
        out[~mask] = a  # exact limit as t -> 0
        chi[large] = out

    return chi


def get_so3_characters_dict(
    alphas: torch.Tensor, betas: torch.Tensor, gammas: torch.Tensor, o3_lambda_max: int
) -> Dict[int, torch.Tensor]:
    """
    Returns a dictionary of the SO(3) characters for all o3_lambda in [0, o3_lambda_max].
    """
    characters = {}
    for o3_lambda in range(o3_lambda_max + 1):
        characters[o3_lambda] = get_so3_character(alphas, betas, gammas, o3_lambda)
    return characters


def get_pso3_characters_dict(
    so3_character: Dict[int, torch.Tensor], o3_lambda_max: int
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Returns a dictionary of the P⋅SO(3) characters for all (o3_lambda, o3_sigma) pairs
    with o3_lambda in [0, o3_lambda_max] and o3_sigma in {-1, +1}.
    Requires a pre-computed dictionary of SO(3) characters.
    """
    characters = {}
    for o3_lambda in range(o3_lambda_max + 1):
        for o3_sigma in [-1, +1]:
            characters[(o3_lambda, o3_sigma)] = (
                o3_sigma * ((-1) ** o3_lambda) * so3_character[o3_lambda]
            )
    return characters


############


class O3Sampler:
    """
    Compute model predictions on a quadrature over the O(3) group.

    :param quad_l_max: maximum spherical harmonic degree for quadrature
    :param project_l_max: maximum spherical harmonic degree to project onto
    """

    def __init__(self, quad_l_max: int, project_l_max: int, batch_size: int = 1):
        try:
            from scipy.spatial.transform import Rotation
        except ImportError as e:
            raise ImportError(
                "To perform data augmentation on spherical targets, please "
                "install the `scipy` package with `pip install scipy`."
            ) from e

        self.quad_l_max = quad_l_max
        """Maximum spherical harmonic degree for quadrature."""

        self.project_l_max = project_l_max
        """Maximum spherical harmonic degree to project onto."""
        if self.project_l_max + 2 > self.quad_l_max:
            warnings.warn(
                (
                    f"Projecting up to l={self.project_l_max} with quadrature up "
                    f"to l={self.quad_l_max} may be inaccurate."
                ),
                stacklevel=2,
            )

        # Get the quadrature
        self.lebedev_order: int
        """Number of Lebedev nodes on the unit sphere."""

        self.n_inplane_rotations: int
        """Number of in-plane rotations per Lebedev node."""
        self.lebedev_order, self.n_inplane_rotations = _choose_quadrature(
            self.quad_l_max
        )

        self.w_so3: torch.Tensor
        """Weights associated to each rotation in the SO(3) Haar measure."""

        alpha, beta, gamma, self.w_so3 = get_euler_angles_quadrature(
            self.lebedev_order, self.n_inplane_rotations
        )
        self.w_so3 = torch.from_numpy(self.w_so3)

        # For active rotation of systems
        self.R_so3 = torch.from_numpy(
            _rotations_from_angles(alpha, beta, gamma).as_matrix()
        )
        """Rotation matrices."""

        self.n_rotations = self.R_so3.size(0)

        # For inverse rotation of tensors
        R_pso3 = _rotations_from_angles(np.pi - alpha, beta, np.pi - gamma)
        self.wigner_D: Dict[int, torch.Tensor] = _compute_wigner_D_matrices(
            self.project_l_max, R_pso3
        )
        """Dict mapping l to (N, (2l+1), (2l+1)) torch.Tensor of Wigner D matrices."""

        self.R_pso3 = torch.from_numpy(R_pso3.as_matrix())
        """Inverse rotation matrices."""

        self.batch_size = batch_size

        _external_product_euler_angles = _extract_euler_zyz(
            (self.R_so3[:, None, :, :] @ self.R_pso3[None, :, :, :]), eps=1e-6
        )
        self.so3_characters = get_so3_characters_dict(
            *_external_product_euler_angles, self.project_l_max
        )
        self.pso3_characters = get_pso3_characters_dict(
            self.so3_characters, self.project_l_max
        )

    def evaluate(
        self,
        model: AtomisticModel,
        systems: List[System],
        options: ModelEvaluationOptions,
        check_consistency: bool = False,
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

        transformed_outputs = {
            name: [{-1: None, 1: None} for _ in systems]
            for name in options.outputs.keys()
        }
        backtransformed_outputs = {
            name: [{-1: None, 1: None} for _ in systems]
            for name in options.outputs.keys()
        }
        for i_sys, system in enumerate(systems):
            for inversion in [-1, 1]:
                rotation_outputs = []
                for batch in range(0, len(self.R_so3), self.batch_size):
                    transformed_systems = [
                        _transform_system(
                            system, inversion * R.to(device=device, dtype=dtype)
                        )
                        for R in self.R_so3[batch : batch + self.batch_size]
                    ]
                    outputs = model(
                        transformed_systems,
                        options=options,
                        check_consistency=check_consistency,
                    )
                    rotation_outputs.append(outputs)

                for name in transformed_outputs:
                    tensor = mts.join(
                        [r[name] for r in rotation_outputs],
                        "samples",
                        remove_tensor_name=True,
                    )
                    transformed_outputs[name][i_sys][inversion] = mts.rename_dimension(
                        tensor, "samples", "tensor", "o3_sample"
                    )

        n_rot = self.R_so3.size(0)
        for name in transformed_outputs:
            for i_sys, system in enumerate(systems):
                for inversion in [-1, 1]:
                    tensor = transformed_outputs[name][i_sys][inversion]
                    _, backtransformed, _ = _apply_random_augmentations(
                        [system] * n_rot,
                        {name: tensor},
                        list(
                            (
                                self.R_pso3.to(device=device, dtype=dtype) * inversion
                            ).unbind(0)
                        ),
                        self.wigner_D,
                    )
                    backtransformed_outputs[name][i_sys][inversion] = backtransformed[
                        name
                    ]

        return transformed_outputs, backtransformed_outputs


class TokenProjector(torch.nn.Module):
    """
    Wrap an atomistic model to project its predictions onto spherical sectors.

    :param model: atomistic model to wrap
    :param quad_l_max: maximum spherical harmonic degree for quadrature
    :param project_l_max: maximum spherical harmonic degree to project onto
    :param batch_size: number of rotations to process in a single batch
    """

    def __init__(
        self,
        model: AtomisticModel,
        quad_l_max: int,
        project_l_max: int,
        batch_size: Optional[int] = None,
    ) -> None:
        super().__init__("TokenProjector")
        self.model = model
        """The underlying atomistic model."""
        self.o3_sampler = O3Sampler(quad_l_max, project_l_max, batch_size=batch_size)
        """The projector onto spherical sectors."""

    def forward(
        self,
        systems: List[System],
        options: ModelEvaluationOptions,
        check_consistency: bool = False,
    ) -> torch.Tensor:
        """
        :param systems: list of systems to evaluate
        :param options: model evaluation options
        :param check_consistency: whether to check model consistency
        :return: TODO
        """

        transformed_outputs, _ = self.o3_sampler.evaluate(
            systems, self.model, options, check_consistency
        )

        # TODO do projection operations
        pass


class SymmetrizedAtomisticModel(torch.nn.Module):
    """
    Wrap an atomistic model to symmetrize its predictions over a quadrature and compute
    O(3) averages, variances, and equivariance score.

    :param model: atomistic model to wrap
    :param quad_l_max: maximum spherical harmonic degree for quadrature
    :param project_l_max: maximum spherical harmonic degree to project onto
    :param batch_size: number of rotations to process in a single batch
    """

    def __init__(
        self,
        model: AtomisticModel,
        quad_l_max: int,
        project_l_max: int,
        batch_size: Optional[int] = None,
    ):
        super().__init__("SymmetrizedAtomisticModel")
        self.model = model
        """The underlying atomistic model."""
        self.o3_sampler = O3Sampler(quad_l_max, project_l_max, batch_size=batch_size)
        """The projector onto spherical sectors."""

    def forward(
        self,
        systems: List[System],
        options: ModelEvaluationOptions,
        check_consistency: bool = False,
    ) -> torch.Tensor:
        """
        :param systems: list of systems to evaluate
        :param options: model evaluation options
        :param check_consistency: whether to check model consistency
        :return:
        """

        transformed_outputs, _ = self.o3_sampler.evaluate(
            systems, self.model, options, check_consistency
        )

        return compute_projections(
            self.o3_sampler.project_l_max,
            systems,
            transformed_outputs,
            self.o3_sampler.w_so3,
            self.o3_sampler.so3_characters,
            self.o3_sampler.pso3_characters,
        )


def _compute_wigner_D_matrices(
    l_max: int,
    rotations: List["Rotation"],
    complex_to_real: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute Wigner D matrices for all l <= project_l_max.

    :param l_max: maximum spherical harmonic degree
    :param rotations: list of scipy Rotation objects
    :param complex_to_real: optional dict mapping l to (2l+1, (2l+1)) array to convert
        complex spherical harmonics to real spherical harmonics
    :return: dict mapping l to (N, (2l+1), (2l+1)) array of Wigner D matrices
    """

    try:
        import spherical
    except ImportError as e:
        # quaternionic (used below) is a dependency of spherical
        raise ImportError(
            "To perform data augmentation on spherical targets, please "
            "install the `spherical` package with `pip install spherical`."
        ) from e

    wigner = spherical.Wigner(l_max)
    scipy_quaternions = [r.as_quat() for r in rotations]
    quaternionic_quaternions = [
        _scipy_quaternion_to_quaternionic(q) for q in scipy_quaternions
    ]
    wigner_D_matrices_complex = [wigner.D(q) for q in quaternionic_quaternions]

    if complex_to_real is None:
        complex_to_real = {
            ell: _complex_to_real_spherical_harmonics_transform(ell)
            for ell in range(l_max + 1)
        }

    wigner_D_matrices = {}
    for ell in range(l_max + 1):
        U = complex_to_real[ell]
        wigner_D_matrices_l = []
        for wigner_D_matrix_complex in wigner_D_matrices_complex:
            wigner_D_matrix = np.zeros((2 * ell + 1, 2 * ell + 1), dtype=np.complex128)
            for mp in range(-ell, ell + 1):
                for m in range(-ell, ell + 1):
                    wigner_D_matrix[m + ell, mp + ell] = (
                        wigner_D_matrix_complex[wigner.Dindex(ell, m, mp)]
                    ).conj()

            wigner_D_matrix = U.conj() @ wigner_D_matrix @ U.T
            assert np.allclose(wigner_D_matrix.imag, 0.0)
            wigner_D_matrix = wigner_D_matrix.real
            wigner_D_matrices_l.append(torch.from_numpy(wigner_D_matrix))
        wigner_D_matrices[ell] = wigner_D_matrices_l

    return wigner_D_matrices


# O3-integrals utilities


def compute_projections(
    max_l: int,
    systems: List[System],
    transformed_outputs: Dict[str, List[TensorMap]],
    weights: torch.Tensor,
    so3_characters: Dict[int, torch.Tensor],
    pso3_characters: Dict[Tuple[int, int], torch.Tensor],
) -> Tuple[
    Dict[str, List[Dict[int, TensorMap]]],
    Dict[str, List[Dict[Tuple[int, int], TensorMap]]],
    Dict[str, List[Dict[Tuple[int, int], TensorMap]]],
]:
    """

    TODO docstring, check type annotations

    - Take model outputs on a quadrature
    - Manipulate dimensions
    - Compute some integrals
    - Return projections

    """

    device = systems[0].positions.device
    dtype = systems[0].positions.dtype

    weights = weights.to(device, dtype)
    so3_characters = {k: v.to(device, dtype) for k, v in so3_characters.items()}
    pso3_characters = {k: v.to(device, dtype) for k, v in pso3_characters.items()}

    n_rotations = len(weights)
    norms = {}
    convolution_integrals = {}
    normalized_convolution_integrals = {}
    # Loop over targets
    for name, transformed_output in transformed_outputs.items():
        norms[name] = []
        convolution_integrals[name] = []
        normalized_convolution_integrals[name] = []
        for o3_output_for_system in transformed_output:
            proper = o3_output_for_system[1]
            improper = o3_output_for_system[-1]

            # Weighting the tensors
            broadcasted_w = (
                weights[proper[0].samples.column("o3_sample")] / 16 / torch.pi**2
            )
            proper_weighted = proper.copy()
            improper_weighted = improper.copy()
            for k in proper_weighted.keys:
                proper_block = proper_weighted[k]
                improper_block = improper_weighted[k]
                proper_block.values[:] *= broadcasted_w.view(
                    -1, *[1] * (proper_block.values.ndim - 1)
                )
                improper_block.values[:] *= broadcasted_w.view(
                    -1, *[1] * (improper_block.values.ndim - 1)
                )

            # Compute norms
            proper_norm = mts.multiply(proper, proper_weighted)
            improper_norm = mts.multiply(improper, improper_weighted)
            norm = mts.add(proper_norm, improper_norm)
            norm = mts.sum_over_samples(norm, "o3_sample")
            norms[name].append(norm)

            # Compute convolution integrals
            convolution_integral = {}
            normalized_convolution_integral = {}
            for ell in range(max_l + 1):
                so3_char = so3_characters[ell]
                for sigma in [-1, 1]:
                    pso3_char = pso3_characters[(ell, sigma)]

                    integral_blocks = []
                    for k in proper.keys:
                        proper_block = proper[k].values.reshape(
                            -1, n_rotations, *proper[k].shape[1:]
                        )
                        improper_block = improper[k].values.reshape(
                            -1, n_rotations, *improper[k].shape[1:]
                        )
                        integral_values = (
                            (
                                0.25
                                * torch.einsum(
                                    "ij...,nij...->n...",
                                    so3_char,
                                    proper_block[:, :, None, ...]
                                    * proper_block[:, None, :, ...]
                                    + improper_block[:, :, None, ...]
                                    * improper_block[:, None, :, ...],
                                )
                                + 0.5
                                * torch.einsum(
                                    "ij...,nij...->n...",
                                    pso3_char,
                                    proper_block[:, :, None, ...]
                                    * improper_block[:, None, :, ...],
                                )
                            )
                            * (2 * ell + 1)
                            / (8 * torch.pi**2) ** 2
                        )
                        integral_blocks.append(
                            mts.TensorBlock(
                                samples=norm[k].samples,
                                components=norm[k].components,
                                properties=norm[k].properties,
                                values=integral_values,
                            )
                        )
                    convolution_integral[(ell, sigma)] = mts.TensorMap(
                        keys=norm.keys, blocks=integral_blocks
                    )
                    normalized_convolution_integral[(ell, sigma)] = mts.divide(
                        convolution_integral[(ell, sigma)], norm
                    )
            convolution_integrals[name].append(convolution_integral)
            normalized_convolution_integrals[name].append(
                normalized_convolution_integral
            )

    return norms, convolution_integrals, normalized_convolution_integrals
