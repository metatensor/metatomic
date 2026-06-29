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
    ell_max: int, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray
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
    if not (alpha.shape == beta.shape == gamma.shape):
        raise ValueError("alpha, beta, and gamma must have identical shapes")

    mp_max = ell_max
    dsize = _wigner_d_size(0, mp_max, ell_max)
    result = np.zeros(alpha.shape + (dsize,), dtype=np.complex128)

    for index in np.ndindex(alpha.shape):
        a = float(alpha[index])
        b = float(beta[index])
        g = float(gamma[index])
        _wigner_D_array = wigner_D_array(ell_max, a, b, g)
        out = result[index]

        for ell in range(0, ell_max + 1):
            for mp in range(-ell, ell + 1):
                i_d = _wigner_d_index(ell, mp, -ell, 0, mp_max)
                for m in range(-ell, ell + 1):
                    out[i_d] = _wigner_D_array[ell][mp + ell, m + ell]
                    i_d += 1

    return result


def compute_complex_wigner_d_matrices(
    ell_max: int,
    angles: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[int, np.ndarray]:
    """Return complex Wigner-D matrices for ell in ``[0, ell_max]`` at the given ZYZ
    angles.

    The returned matrices follow the standard ``wigners`` indexing convention:
    ``matrix[..., mp + ell, m + ell] == D^ell_{mp,m}``.

    :param ell_max: maximum angular-momentum order
    :param angles: ``(alpha, beta, gamma)`` ZYZ Euler-angle arrays of matching shape
    :return: ``{ell: array of shape (*angles[0].shape, 2*ell+1, 2*ell+1)}``
    """
    alpha, beta, gamma = angles
    raw = _compute_wigner_d_complex(ell_max, alpha, beta, gamma)
    matrices: dict[int, np.ndarray] = {}
    for ell in range(ell_max + 1):
        shape = alpha.shape + (2 * ell + 1, 2 * ell + 1)
        block = np.zeros(shape, dtype=np.complex128)
        for mp in range(-ell, ell + 1):
            for m in range(-ell, ell + 1):
                block[..., mp + ell, m + ell] = raw[
                    ..., _wigner_d_index(ell, mp, m, 0, ell_max)
                ]
        matrices[ell] = block
    return matrices


def compute_real_wigner_d_matrices(
    ell_max: int,
    angles: tuple[np.ndarray, np.ndarray, np.ndarray],
    complex_to_real: dict[int, np.ndarray],
) -> dict[int, torch.Tensor]:
    """Convert complex Wigner-D matrices to real ones using the provided
    change-of-basis.

    :param ell_max: maximum angular-momentum order
    :param angles: ``(alpha, beta, gamma)`` ZYZ Euler-angle arrays
    :param complex_to_real: ``{ell: (2*ell+1, 2*ell+1)}`` unitary transform from complex
        to real spherical harmonics
    :return: ``{ell: real tensor of shape (*angles[0].shape, 2*ell+1, 2*ell+1)}``
    :raises ValueError: if imaginary residuals exceed numerical tolerance
    """
    complex_matrices = compute_complex_wigner_d_matrices(ell_max, angles)
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


def compute_real_wigner_matrices(
    o3_lambda_max: int,
    angles: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[int, torch.Tensor]:
    """Build the real Wigner-D matrices for ``ell = 0..o3_lambda_max`` at the given
    ZYZ Euler angles, using the cached complex-to-real transform per ell."""
    complex_to_real = {
        ell: _complex_to_real_spherical_harmonics_transform(ell)
        for ell in range(o3_lambda_max + 1)
    }
    return compute_real_wigner_d_matrices(o3_lambda_max, angles, complex_to_real)


def compute_wigner_batch(
    ell_max: int,
    angles: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    """Real Wigner-D matrices for ``ell = 0..ell_max`` at the given angles, cast to
    the requested device and dtype."""
    return {
        ell: tensor.to(device=device, dtype=dtype)
        for ell, tensor in compute_real_wigner_matrices(ell_max, angles).items()
    }
