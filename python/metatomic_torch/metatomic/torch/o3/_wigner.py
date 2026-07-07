"""Private helpers to build real Wigner-D matrices for augmentation.

Complex Wigner-D matrices are delegated to :func:`wigners.wigner_D_array` using
ZYZ Euler angles. This module then reshapes/indexes these matrices into the flat
layout expected by the augmentation code and applies the complex-to-real
change-of-basis.

The indexing convention used here matches ``wigners``:
``D[mp + ell, m + ell] == D^ell_{mp,m}``.
"""

import functools

import numpy as np
import torch
from wigners import wigner_D_array


def _wigner_d_size(ell_min: int, mp_max: int, ell_max: int) -> int:
    if mp_max >= ell_max:
        return (
            ell_max * (ell_max * (4 * ell_max + 12) + 11)
            + ell_min * (1 - 4 * ell_min**2)
            + 3
        ) // 3
    if mp_max > ell_min:
        return (
            3 * ell_max * (ell_max + 2)
            + ell_min * (1 - 4 * ell_min**2)
            + mp_max
            * (3 * ell_max * (2 * ell_max + 4) + mp_max * (-2 * mp_max - 3) + 5)
            + 3
        ) // 3

    return (ell_max * (ell_max + 2) - ell_min**2) * (1 + 2 * mp_max) + 2 * mp_max + 1


def _wigner_d_index(ell: int, mp: int, m: int, ell_min: int, mp_max: int) -> int:
    idx = 0
    for ell_prev in range(ell_min, ell):
        local_mp_max = mp_max if mp_max < ell_prev else ell_prev
        idx += (2 * local_mp_max + 1) * (2 * ell_prev + 1)

    local_mp_max = mp_max if mp_max < ell else ell
    idx += (mp + local_mp_max) * (2 * ell + 1)
    idx += m + ell
    return idx


def _compute_wigner_d_complex(
    ell_max: int, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Compute complex Wigner-D matrix elements for all ell in ``[0, ell_max]``.

    Calls :func:`wigners.wigner_D_array` for each input angle triplet, then packs
    values into a flat output array indexed by :func:`_wigner_d_index`.

    :param ell_max: maximum angular-momentum order
    :param alpha: ZYZ first rotation angle, arbitrary shape
    :param beta: ZYZ second rotation angle, same shape as ``alpha``
    :param gamma: ZYZ third rotation angle, same shape as ``alpha``
    :return: complex array of shape ``(*alpha.shape, dsize)``
    """

    mp_max = ell_max
    dsize = _wigner_d_size(0, mp_max, ell_max)
    result = np.zeros(dsize, dtype=np.complex128)

    _wigner_D_array = wigner_D_array(ell_max, alpha, beta, gamma)

    for ell in range(0, ell_max + 1):
        for mp in range(-ell, ell + 1):
            i_d = _wigner_d_index(ell, mp, -ell, 0, mp_max)
            for m in range(-ell, ell + 1):
                result[i_d] = _wigner_D_array[ell][mp + ell, m + ell]
                i_d += 1

    return result


def _compute_complex_wigner_d_matrices(
    ell_max: int,
    angles: tuple[float, float, float],
) -> dict[int, np.ndarray]:
    """Return complex Wigner-D matrices for ell in ``[0, ell_max]`` at the given ZYZ
    angles.

    The returned matrices follow the standard ``wigners`` indexing convention:
    ``matrix[..., mp + ell, m + ell] == D^ell_{mp,m}``.

    :param ell_max: maximum angular-momentum order
    :param angles: ``(alpha, beta, gamma)`` ZYZ Euler-angles
    :return: ``{ell: array of shape (2*ell+1, 2*ell+1)}``
    """
    alpha, beta, gamma = angles
    raw = _compute_wigner_d_complex(ell_max, alpha, beta, gamma)
    matrices: dict[int, np.ndarray] = {}
    for ell in range(ell_max + 1):
        block = np.zeros((2 * ell + 1, 2 * ell + 1), dtype=np.complex128)
        for mp in range(-ell, ell + 1):
            for m in range(-ell, ell + 1):
                block[mp + ell, m + ell] = raw[_wigner_d_index(ell, mp, m, 0, ell_max)]
        matrices[ell] = block
    return matrices


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
    :return: ``{ell: real tensor of shape (*angles[0].shape, 2*ell+1, 2*ell+1)}``
    """
    complex_matrices = _compute_complex_wigner_d_matrices(ell_max, angles)
    real_matrices: dict[int, torch.Tensor] = {}
    for ell, matrix in complex_matrices.items():
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
    # R = Rz(alpha) Ry(beta) Rz(gamma): element [2,2] = cos(beta)
    cos_beta = rotation[2, 2].clamp(-1.0, 1.0)
    beta = torch.arccos(cos_beta)
    sin_beta = torch.sin(beta)
    beta = float(beta)
    if abs(sin_beta) < 1e-10:
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
