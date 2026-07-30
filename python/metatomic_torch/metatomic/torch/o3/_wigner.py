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
    """Compute real Wigner-D matrices using the provided change of basis.

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
        # The basis change should be real up to numerical roundoff.
        if not np.allclose(matrix.imag, 0.0, rtol=0.0, atol=1e-9):
            raise ValueError("real Wigner matrix conversion produced complex values")
        real_matrices[ell] = torch.from_numpy(matrix.real)
    return real_matrices


@functools.lru_cache(maxsize=None)
def _complex_to_real_spherical_harmonics_transform(ell: int) -> np.ndarray:
    """Return the unitary transform ``T`` with ``Y_real = T @ Y_complex``.

    Both axes are ordered from ``m=-ell`` through ``m=ell``.
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
    # Recover beta from both sine and cosine to remain stable at the poles.
    cos_beta = rotation[2, 2].clamp(-1.0, 1.0)
    sin_beta = torch.sqrt(rotation[0, 2] ** 2 + rotation[1, 2] ** 2)
    beta = float(torch.atan2(sin_beta, cos_beta))
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
    """Return real Wigner-D matrices for ``ell = 0, ..., o3_lambda_max``.

    If ``matrix`` has negative determinant, ``-matrix`` is a proper rotation.
    Build the Wigner-D matrices for this proper rotation; the caller restores
    the inversion action on spherical values with ``sigma * (-1) ** ell``.
    Every returned tensor uses ``device`` and ``dtype``.
    """
    angles = _rotation_to_angles(matrix)
    complex_to_real = {
        ell: _complex_to_real_spherical_harmonics_transform(ell)
        for ell in range(o3_lambda_max + 1)
    }
    cache = _compute_real_wigner_d_matrices(o3_lambda_max, angles, complex_to_real)

    return {ell: tensor.to(device=device, dtype=dtype) for ell, tensor in cache.items()}


def build_packed_wigner_matrices(
    matrices: torch.Tensor,
    max_o3_lambda: int,
) -> torch.Tensor:
    """Build and pack proper Wigner-D matrices through ``max_o3_lambda``.

    :param matrices: ``(n_matrices, 3, 3)`` stack of O(3) matrices
    :param max_o3_lambda: maximum angular momentum to include
    :return: flat tensor holding every Wigner-D matrix, laid out for
        :py:func:`wigner_matrices_for_lambda`, with the dtype and device of
        ``matrices``
    """
    output_device = matrices.device
    output_dtype = matrices.dtype
    calculation_matrices = matrices.detach().to(device="cpu")
    cpu = torch.device("cpu")
    n_matrices = matrices.size(0)
    n_elements_per_matrix = (
        (max_o3_lambda + 1) * (2 * max_o3_lambda + 1) * (2 * max_o3_lambda + 3) // 3
    )
    packed = torch.empty(
        n_matrices * n_elements_per_matrix,
        dtype=output_dtype,
        device="cpu",
    )

    for matrix_index, matrix in enumerate(calculation_matrices.unbind(0)):
        cache = build_wigner_D_cache(
            max_o3_lambda,
            matrix,
            device=cpu,
            dtype=output_dtype,
        )
        for o3_lambda in range(max_o3_lambda + 1):
            dimension = 2 * o3_lambda + 1
            elements_before = o3_lambda * (4 * o3_lambda * o3_lambda - 1) // 3
            offset = n_matrices * elements_before + matrix_index * dimension * dimension
            packed[offset : offset + dimension * dimension].copy_(
                cache[o3_lambda].reshape(-1)
            )

    return packed.to(
        device=output_device,
        dtype=output_dtype,
    )


def wigner_matrices_for_lambda(
    packed: torch.Tensor,
    n_matrices: int,
    o3_lambda: int,
) -> torch.Tensor:
    """Return the packed Wigner-D stack for one ``o3_lambda`` as a view."""
    # the packed layout is rank-major then matrix-major: all matrices for
    # o3_lambda=0 come first, then all matrices for o3_lambda=1, and so on
    dimension = 2 * o3_lambda + 1
    elements_before = o3_lambda * (4 * o3_lambda * o3_lambda - 1) // 3
    offset = n_matrices * elements_before
    length = n_matrices * dimension * dimension
    if offset + length > packed.numel():
        raise ValueError("o3_lambda exceeds the packed Wigner-D storage")

    return packed[offset : offset + length].view(
        n_matrices,
        dimension,
        dimension,
    )
