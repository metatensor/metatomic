"""Private helpers to build real Wigner-D matrices for augmentation.

Complex Wigner-D matrices are delegated to :func:`wigners.wigner_D_array` using
ZYZ Euler angles and then converted to the real spherical-harmonic basis.

The indexing convention used here matches ``wigners``:
``D[mp + ell, m + ell] == D^ell_{mp,m}``.
"""

import functools

import numpy as np
import torch
from wigners import wigner_D_array


def _compute_real_wigner_d_matrices(
    ell_max: int,
    angles: tuple[float, float, float],
    complex_to_real: dict[int, np.ndarray],
) -> dict[int, torch.Tensor]:
    """Convert complex Wigner-D matrices to real ones using the provided
    change-of-basis.

    :param ell_max: maximum angular-momentum order
    :param angles: ``(alpha, beta, gamma)`` ZYZ Euler-angles
    :param complex_to_real: ``{ell: (2*ell+1, 2*ell+1)}`` unitary transform from complex
        to real spherical harmonics
    :return: ``{ell: real tensor of shape (2*ell+1, 2*ell+1)}``
    """
    alpha, beta, gamma = angles
    real_matrices: dict[int, torch.Tensor] = {}
    for ell, matrix in enumerate(wigner_D_array(ell_max, alpha, beta, gamma)):
        transform = complex_to_real[ell]
        matrix = np.einsum("ij,...jk,kl->...il", transform.conj(), matrix, transform.T)
        # The complex-to-real basis change can leave tiny imaginary residuals;
        # scale the tolerance by matrix magnitude instead of using a fixed atol.
        scale = float(np.max(np.abs(matrix.real))) if matrix.size else 1.0
        atol = max(1e-9, scale * 1e-10)
        if not np.allclose(matrix.imag, 0.0, atol=atol):
            raise ValueError("real Wigner matrix conversion produced complex values")
        real_matrices[ell] = torch.from_numpy(matrix.real)
    return real_matrices


@functools.lru_cache(maxsize=None)
def _complex_to_real_spherical_harmonics_transform(ell: int) -> np.ndarray:
    """Return the complex-to-real spherical-harmonics transform for one ``ell``.

    The returned matrix has shape ``(2*ell+1, 2*ell+1)``.
    """
    if ell < 0 or not isinstance(ell, int):
        raise ValueError("ell must be a non-negative integer.")

    size = 2 * ell + 1
    T = np.zeros((size, size), dtype=complex)

    for m in range(-ell, ell + 1):
        m_index = m + ell
        if m > 0:
            T[m_index, ell + m] = 1 / np.sqrt(2) * (-1) ** m
            T[m_index, ell - m] = 1 / np.sqrt(2)
        elif m < 0:
            T[m_index, ell + abs(m)] = -1j / np.sqrt(2) * (-1) ** m
            T[m_index, ell - abs(m)] = 1j / np.sqrt(2)
        else:
            T[m_index, ell] = 1

    return T


def _rotation_to_angles(
    rotation: torch.Tensor,
) -> tuple[float, float, float]:
    """
    Decompose an O(3) rotation matrix into ZYZ Euler angles :math:`(\\alpha, \\beta,
    \\gamma)`.

    For improper rotations (det < 0) the proper part ``-R`` is decomposed; the inversion
    parity factor is handled separately when applying Wigner-D matrices.
    """

    rotation = rotation if torch.det(rotation) > 0 else -rotation
    # R = Rz(alpha) Ry(beta) Rz(gamma): element [2,2] = cos(beta), while
    # hypot(R[0,2], R[1,2]) = sin(beta) for beta in [0, pi]. Recover beta
    # with atan2 instead of arccos: close to a pole, arccos amplifies a one-ulp
    # perturbation of R[2,2] into an O(sqrt(eps)) angle and can incorrectly
    # bypass the gimbal-lock branch.
    cos_beta = rotation[2, 2].clamp(-1.0, 1.0)
    sin_beta = torch.sqrt(rotation[0, 2] ** 2 + rotation[1, 2] ** 2)
    beta = torch.atan2(sin_beta, cos_beta)
    beta = float(beta)
    pole_tolerance = 8.0 * torch.finfo(rotation.dtype).eps
    if float(sin_beta) <= pole_tolerance:
        # Gimbal lock: only alpha +/- gamma is defined; fix gamma=0
        if cos_beta > 0:
            alpha = float(torch.atan2(rotation[1, 0], rotation[0, 0]))
        else:
            alpha = float(torch.atan2(-rotation[1, 0], -rotation[0, 0]))
        gamma = 0.0
    else:
        # R[0,2]=cos(alpha)*sin(beta), R[1,2]=sin(alpha)*sin(beta): alpha via atan2
        # R[2,1]=sin(beta)*sin(gamma), R[2,0]=-sin(beta)*cos(gamma): gamma via atan2
        alpha = float(torch.atan2(rotation[1, 2], rotation[0, 2]))
        gamma = float(torch.atan2(rotation[2, 1], -rotation[2, 0]))

    return alpha, beta, gamma


def build_wigner_D_cache(
    o3_lambda_max: int,
    matrix: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    """
    Build the cache of real Wigner-D matrices for ``ell = 0..o3_lambda_max`` at the ZYZ
    Euler angles corresponding to the given rotation matrix, using cached
    complex-to-real transform per ell.
    """
    angles = _rotation_to_angles(matrix)
    complex_to_real = {
        ell: _complex_to_real_spherical_harmonics_transform(ell)
        for ell in range(o3_lambda_max + 1)
    }
    cache = _compute_real_wigner_d_matrices(o3_lambda_max, angles, complex_to_real)

    return {ell: tensor.to(device=device, dtype=dtype) for ell, tensor in cache.items()}
