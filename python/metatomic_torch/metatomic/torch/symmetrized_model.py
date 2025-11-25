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
    Compute the character convolution between a block containing SO(3)-sampled tensors.
    Then contract with another block.
    """
    samples = block1.samples
    assert samples.names[0] == "so3_rotation"
    components = block1.components
    properties = block1.properties
    values = block1.values
    chi = chi.to(dtype=values.dtype, device=values.device)
    n_rot = chi.size(1)
    weight = w.to(dtype=values.dtype, device=values.device)

    # reshape the values to separate rotations from the other samples
    new_shape = [n_rot, -1] + list(values.shape[1:])
    reshaped_values = values.reshape(new_shape)

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
    # reshape the values to separate rotations from the other samples
    new_shape = [n_rot, -1] + list(values2.shape[1:])
    reshaped_values2 = values2.reshape(new_shape)

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


class SymmetrizedModel(torch.nn.Module):
    """
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
        max_o3_lambda_grid,
        max_o3_lambda_target,
        batch_size: int = 32,
        max_o3_lambda_character: Optional[int] = None,
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

        self.max_o3_lambda_grid = max_o3_lambda_grid
        self.max_o3_lambda_target = max_o3_lambda_target
        self.batch_size = batch_size
        if max_o3_lambda_character is None:
            max_o3_lambda_character = max_o3_lambda_grid
        self.max_o3_lambda_character = max_o3_lambda_character

        # Compute grid (unchanged)
        lebedev_order, n_inplane_rotations = _choose_quadrature(self.max_o3_lambda_grid)
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

        self._wigner_D_inverse_jit: Dict[int, torch.Tensor] = {}
        self._so3_characters_jit: Dict[int, torch.Tensor] = {}
        self._pso3_characters_jit: Dict[str, torch.Tensor] = {}
        # Since Wigner D matrices are stored in dicts, we need a bit of gymnastics to
        # register the buffers
        raw_wigner = _compute_real_wigner_matrices(
            self.max_o3_lambda_target, angles_inverse_rotations
        )
        self._wigner_D_inverse_names: Dict[int, str] = {}
        for ell, D in raw_wigner.items():
            if isinstance(D, np.ndarray):
                D = torch.from_numpy(D)
            D = D.to(dtype=dtype, device=device)
            name = f"wigner_D_inverse_rotations_l{ell}"
            self.register_buffer(name, D)
            self._wigner_D_inverse_names[ell] = name
            # TorchScript dict view uses the same tensor
            self._wigner_D_inverse_jit[ell] = D

        # Compute characters
        so3_characters, pso3_characters = compute_characters(
            self.max_o3_lambda_character,
            (alpha, beta, gamma),
            angles_inverse_rotations,
        )
        self._so3_char_names: Dict[int, str] = {}
        self._pso3_char_names: Dict[str, str] = {}

        # Since characters are stored in dicts, we need a bit of gymnastics to
        # register the buffers
        for ell, ch in so3_characters.items():
            if isinstance(ch, np.ndarray):
                ch = torch.from_numpy(ch)

            ch = ch.to(dtype=dtype, device="cpu")   # stay on CPU
            name = f"so3_characters_l{ell}"
            self.register_buffer(name, ch)
            self._so3_char_names[ell] = name

        self._so3_characters_jit = {}  # kill the CUDA dict cache

        for ell, ch in pso3_characters.items():
            if isinstance(ch, np.ndarray):
                ch = torch.from_numpy(ch)

            ch = ch.to(dtype=dtype, device="cpu")   # stay on CPU
            name = f"pso3_characters_l{ell}"
            self.register_buffer(name, ch)
            self._pso3_char_names[ell] = name

        self._pso3_characters_jit = {}

    @torch.jit.ignore
    def _wigner_D_inverse_dict(self) -> Dict[int, torch.Tensor]:
        return {
            ell: getattr(self, name)
            for ell, name in self._wigner_D_inverse_names.items()
        }

    @property
    def wigner_D_inverse_rotations(self) -> Dict[int, torch.Tensor]:
        # Python-only nice view
        return self._wigner_D_inverse_dict()

    @torch.jit.ignore
    def _so3_characters_dict(self) -> Dict[int, torch.Tensor]:
        return {ell: getattr(self, name) for ell, name in self._so3_char_names.items()}

    @property
    def so3_characters(self) -> Dict[int, torch.Tensor]:
        # Python-only nice view
        return self._so3_characters_dict()

    @torch.jit.ignore
    def _pso3_characters_dict(self) -> Dict[str, torch.Tensor]:
        return {key: getattr(self, name) for key, name in self._pso3_char_names.items()}

    @property
    def pso3_characters(self) -> Dict[str, torch.Tensor]:
        # Python-only nice view
        return self._pso3_characters_dict()

    def _get_wigner_D_inverse(self, ell: int) -> torch.Tensor:
        return self._wigner_D_inverse_jit[ell]

    def _get_so3_character(self, o3_lambda: int) -> torch.Tensor:
        name = self._so3_char_names[o3_lambda]
        ch_cpu = getattr(self, name)

        # follow the base model device/dtype
        try:
            ref = next(self.base_model.parameters())
            device = ref.device
            dtype = ref.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.get_default_dtype()

        return ch_cpu.to(device=device, dtype=dtype, non_blocking=True)

    def _get_pso3_character(self, o3_lambda: int, o3_sigma: int) -> torch.Tensor:
        label = str(o3_lambda) + "_" + str(o3_sigma)
        name = self._pso3_char_names[label]
        ch_cpu = getattr(self, name)

        try:
            ref = next(self.base_model.parameters())
            device = ref.device
            dtype = ref.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.get_default_dtype()

        return ch_cpu.to(device=device, dtype=dtype, non_blocking=True)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """
        Symmetrize the model outputs over :math:`O(3)` and compute equivariance
        metrics.

        :param systems: list of systems to evaluate
        :param outputs: dictionary of model outputs to symmetrize
        :param selected_atoms: optional Labels specifying which atoms to consider
        :return: dictionary with symmetrized outputs and equivariance metrics
        """
        # Evaluate the model over the grid
        transformed_outputs, backtransformed_outputs = self._eval_over_grid(
            systems, outputs, selected_atoms
        )

        # Compute norms
        norms = self._compute_norm_per_property(transformed_outputs)

        # Compute the O(3) mean and variance
        mean_var = self._compute_mean_and_variance(backtransformed_outputs)

        # Compute the character projections
        convolution_integrals = self._compute_conv_integral(transformed_outputs)

        out_dict: Dict[str, TensorMap] = {}
        for name, tensor in norms.items():
            out_dict[name] = tensor
        for name, tensor in mean_var.items():
            out_dict[name] = tensor
        for name, integral in convolution_integrals.items():
            out_dict[name] = integral

        return out_dict

    def _compute_norm_per_property(
        self, tensor_dict: Dict[str, TensorMap]
    ) -> Dict[str, TensorMap]:
        """
        Compute the norm per property of each tensor in ``tensor_dict``.

        :param tensor_dict: dictionary of TensorMaps to compute norms for
        :return: dictionary of TensorMaps with norms per property
        """
        norms: Dict[str, TensorMap] = {}
        for name in tensor_dict:
            tensor = tensor_dict[name]
            norm_blocks: List[TensorBlock] = []
            for block in tensor.blocks():
                rot_ids = block.samples.column("so3_rotation")

                values_squared = block.values**2

                view: List[int] = []
                view.append(values_squared.size(0))
                for _ in range(values_squared.ndim - 1):
                    view.append(1)
                values_squared = (
                    0.5 * self.so3_weights[rot_ids].view(view) * values_squared
                )

                norm_blocks.append(
                    TensorBlock(
                        values=values_squared,  # /(8 * torch.pi**2),
                        samples=block.samples,
                        components=block.components,
                        properties=block.properties,
                    )
                )

            tensor_norm = TensorMap(tensor.keys, norm_blocks)
            tensor_norm = mts.sum_over_samples(
                tensor_norm.keys_to_samples("inversion"), ["inversion", "so3_rotation"]
            )

            norms[name + "_squared_norm"] = tensor_norm
        return norms

    def _compute_conv_integral(
        self, tensor_dict: Dict[str, TensorMap]
    ) -> Dict[str, TensorMap]:
        """
        Compute the O(3)-convolution of each tensor in ``tensor_dict`` with O(3)
        characters.

        :param tensor_dict: dictionary of TensorMaps to compute convolution integral for
        :return: dictionary of TensorMaps with convolution integrals
        """

        new_tensors: Dict[str, TensorMap] = {}
        # loop over tensormaps
        for name, tensor in tensor_dict.items():
            keys = tensor.keys
            remaining_keys = Labels(
                keys.names[:-1],
                keys.values[keys.column("inversion") == 1][:, :-1],
            )
            new_blocks: List[TensorBlock] = []
            new_keys: List[torch.Tensor] = []
            # loop over keys in the final tensormap
            for key_values in remaining_keys.values:
                key_to_match_plus: Dict[str, int] = {}
                key_to_match_minus: Dict[str, int] = {}
                for k, v in zip(remaining_keys.names, key_values, strict=True):
                    key_to_match_plus[k] = int(v)
                    key_to_match_minus[k] = int(v)
                key_to_match_plus["inversion"] = 1
                key_to_match_minus["inversion"] = -1
                # get the corresponding blocks for proper and improper rotations
                so3_block = tensor.block(key_to_match_plus)
                pso3_block = tensor.block(key_to_match_minus)

                # loop over SO(3) irreps
                for o3_lambda in range(self.max_o3_lambda_character + 1):
                    so3_chi = self._get_so3_character(o3_lambda)
                    first_term = _character_convolution(
                        so3_chi, so3_block, so3_block, self.so3_weights
                    )
                    second_term = _character_convolution(
                        so3_chi, pso3_block, pso3_block, self.so3_weights
                    )
                    for o3_sigma in [1, -1]:
                        pso3_chi = self._get_pso3_character(o3_lambda, o3_sigma)
                        third_term = _character_convolution(
                            pso3_chi, pso3_block, so3_block, self.so3_weights
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
                            # / (8 * torch.pi**2) ** 2,
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

    def _compute_mean_and_variance(
        self,
        tensor_dict: Dict[str, TensorMap],
    ) -> Dict[str, TensorMap]:
        """
        Compute the mean and variance of the outputs over O(3).

        :param tensor_dict: dictionary of TensorMaps to compute mean and variance for
        :return: dictionary of TensorMaps with mean and variance
        """
        mean_var: Dict[str, TensorMap] = {}
        for name in tensor_dict:
            if "features" in name:
                continue
            tensor = tensor_dict[name]
            mean_blocks: List[TensorBlock] = []
            second_moment_blocks: List[TensorBlock] = []
            for block in tensor.blocks():
                rot_ids = block.samples.column("so3_rotation")

                values = block.values
                if values.ndim > 2:
                    dims: List[int] = []
                    for i in range(1, values.ndim - 1):
                        dims.append(i)
                    values_squared = torch.sum(values**2, dim=dims)
                else:
                    values_squared = values**2

                view: List[int] = []
                view.append(values.size(0))
                for _ in range(values.ndim - 1):
                    view.append(1)
                values = 0.5 * self.so3_weights[rot_ids].view(view) * values

                view: List[int] = []
                view.append(values_squared.size(0))
                for _ in range(values_squared.ndim - 1):
                    view.append(1)
                values_squared = (
                    0.5 * self.so3_weights[rot_ids].view(view) * values_squared
                )

                mean_blocks.append(
                    TensorBlock(
                        values=values,
                        samples=block.samples,
                        components=block.components,
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
            mean_norm_squared_blocks: List[TensorBlock] = []
            for block in tensor_mean.blocks():
                vals = block.values
                if vals.ndim > 2:
                    dims: List[int] = []
                    for i in range(1, vals.ndim - 1):
                        dims.append(i)
                    vals = torch.sum(vals**2, dim=dims)
                else:
                    vals = vals**2
                mean_norm_squared_blocks.append(
                    TensorBlock(
                        values=vals,
                        samples=block.samples,
                        components=[],
                        properties=block.properties,
                    )
                )
            tensor_mean_norm_squared = TensorMap(
                tensor_mean.keys, mean_norm_squared_blocks
            )

            # Second moment
            tensor_second_moment = TensorMap(tensor.keys, second_moment_blocks)
            tensor_second_moment = mts.sum_over_samples(
                tensor_second_moment.keys_to_samples("inversion"),
                ["inversion", "so3_rotation"],
            )

            # Variance
            tensor_variance = mts.subtract(
                tensor_second_moment, tensor_mean_norm_squared
            )

            mean_var[name + "_mean"] = tensor_mean
            mean_var[name + "_norm_squared"] = tensor_second_moment
            mean_var[name + "_var"] = tensor_variance
        return mean_var

    def _eval_over_grid(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Tuple[Dict[str, TensorMap], Dict[str, TensorMap]]:
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

        transformed_outputs = torch.jit.annotate(
            Dict[str, List[Dict[int, TensorMap]]], {}
        )
        for name in outputs:
            lst = torch.jit.annotate(List[Dict[int, TensorMap]], [])
            for _ in systems:
                d = torch.jit.annotate(Dict[int, TensorMap], {})
                lst.append(d)
            transformed_outputs[name] = lst
        backtransformed_outputs = torch.jit.annotate(
            Dict[str, List[Dict[int, TensorMap]]], {}
        )
        for name in outputs:
            lst = torch.jit.annotate(List[Dict[int, TensorMap]], [])
            for _ in systems:
                d = torch.jit.annotate(Dict[int, TensorMap], {})
                lst.append(d)
            backtransformed_outputs[name] = lst

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
                    with torch.no_grad():
                        out = self.base_model(
                            transformed_systems,
                            outputs,
                            selected_atoms,
                        )
                    rotation_outputs.append(out)

                # Combine batch outputs
                for name in outputs:
                    combined_: List[TensorMap] = [r[name] for r in rotation_outputs]
                    combined = mts.join(
                        combined_,
                        "samples",
                        add_dimension="batch_rotation",
                    )
                    if "batch_rotation" in combined[0].samples.names:
                        # Reindex
                        blocks: List[TensorBlock] = []
                        for block in combined.blocks():
                            batch_id = block.samples.column("batch_rotation")
                            rot_id = block.samples.column("system")
                            new_sample_values = block.samples.values[:, :-1]
                            new_sample_values[:, 0] = (
                                batch_id * self.batch_size + rot_id
                            )
                            blocks.append(
                                TensorBlock(
                                    values=block.values.detach(),
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
                    wigner_dict = torch.jit.annotate(Dict[int, List[torch.Tensor]], {})
                    for ell in self._wigner_D_inverse_jit:
                        wigner_dict[ell] = (
                            self._get_wigner_D_inverse(ell)
                            .to(device=device, dtype=dtype)
                            .unbind(0)
                        )

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
                        wigner_dict,
                    )
                    backtransformed_outputs[name][i_sys][inversion] = backtransformed[
                        name
                    ]

        transformed_outputs_tensor: Dict[str, TensorMap] = {}
        backtransformed_outputs_tensor: Dict[str, TensorMap] = {}
        # Massage outputs to have desired shape
        for name in transformed_outputs:
            joined_plus = mts.join(
                [transformed_outputs[name][i_sys][1] for i_sys in range(len(systems))],
                "samples",
                add_dimension="phys_system",
            )
            joined_minus = mts.join(
                [transformed_outputs[name][i_sys][-1] for i_sys in range(len(systems))],
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
                joined = mts.rename_dimension(
                    joined, "samples", "phys_system", "system"
                )
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
            if "atom" in joined[0].samples.names:
                perm = _permute_system_before_atom(joined[0].samples.names)
                joined = mts.permute_dimensions(joined, "samples", perm)
            transformed_outputs_tensor[name] = joined

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
                    backtransformed_outputs[name][i_sys][-1]
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
            joined = mts.rename_dimension(joined, "samples", "system", "so3_rotation")
            if "phys_system" in joined[0].samples.names:
                joined = mts.rename_dimension(
                    joined, "samples", "phys_system", "system"
                )
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
            if "atom" in joined[0].samples.names:
                perm = _permute_system_before_atom(joined[0].samples.names)
                joined = mts.permute_dimensions(joined, "samples", perm)
            backtransformed_outputs_tensor[name] = joined

        return transformed_outputs_tensor, backtransformed_outputs_tensor


def _permute_system_before_atom(labels: List[str]) -> List[int]:
    # find positions
    sys_idx = -1
    atom_idx = -1
    for i in range(len(labels)):
        if labels[i] == "system":
            sys_idx = i
        elif labels[i] == "atom":
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
