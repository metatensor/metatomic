"""Private helpers to build real Wigner-D matrices for augmentation.

Complex Wigner-D matrices are delegated to :func:`wigners.wigner_D_array` using
ZYZ Euler angles and then converted to the real spherical-harmonic basis.
:py:class:`EulerWignerD` builds the same real matrices from Euler angles inside
TorchScript code.

The indexing convention used here matches ``wigners``:
``D[mp + ell, m + ell] == D^ell_{mp,m}``.
"""

import functools
import math
from typing import List, Optional

import numpy as np
import torch
from wigners import wigner_D_array

from ._utils import validate_integer


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
    max_angular_momentum: int,
    matrix: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    """Return real Wigner-D matrices for ``ell = 0, ..., max_angular_momentum``.

    If ``matrix`` has negative determinant, ``-matrix`` is a proper rotation.
    Build the Wigner-D matrices for this proper rotation; the caller restores
    the inversion action on spherical values with ``sigma * (-1) ** ell``.
    Every returned tensor uses ``device`` and ``dtype``.
    """
    angles = _rotation_to_angles(matrix)
    complex_to_real = {
        ell: _complex_to_real_spherical_harmonics_transform(ell)
        for ell in range(max_angular_momentum + 1)
    }
    cache = _compute_real_wigner_d_matrices(
        max_angular_momentum, angles, complex_to_real
    )

    return {ell: tensor.to(device=device, dtype=dtype) for ell, tensor in cache.items()}


def _packed_elements_before(o3_lambda: int) -> int:
    """Number of packed elements per matrix below ``o3_lambda`` (sum of (2l+1)^2)."""
    return o3_lambda * (4 * o3_lambda * o3_lambda - 1) // 3


def build_packed_wigner_matrices(
    matrices: torch.Tensor,
    max_angular_momentum: int,
) -> torch.Tensor:
    """Build and pack proper Wigner-D matrices through ``max_angular_momentum``.

    :param matrices: ``(n_matrices, 3, 3)`` stack of O(3) matrices
    :param max_angular_momentum: maximum angular momentum to include
    :return: flat tensor holding every Wigner-D matrix, laid out for
        :py:func:`wigner_matrices_for_lambda`, with the dtype and device of
        ``matrices``
    """
    output_device = matrices.device
    output_dtype = matrices.dtype
    calculation_matrices = matrices.detach().to(device="cpu")
    cpu = torch.device("cpu")
    n_matrices = matrices.size(0)
    n_elements_per_matrix = _packed_elements_before(max_angular_momentum + 1)
    packed = torch.empty(
        n_matrices * n_elements_per_matrix,
        dtype=output_dtype,
        device="cpu",
    )

    for matrix_index, matrix in enumerate(calculation_matrices.unbind(0)):
        cache = build_wigner_D_cache(
            max_angular_momentum,
            matrix,
            device=cpu,
            dtype=output_dtype,
        )
        for o3_lambda in range(max_angular_momentum + 1):
            dimension = 2 * o3_lambda + 1
            elements_before = _packed_elements_before(o3_lambda)
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
    offset = n_matrices * _packed_elements_before(o3_lambda)
    length = n_matrices * dimension * dimension
    # callers validate o3_lambda against their own maximum angular momentum
    assert offset + length <= packed.numel()

    return packed[offset : offset + length].view(
        n_matrices,
        dimension,
        dimension,
    )


def _z_rotation_matrices(angles: torch.Tensor) -> torch.Tensor:
    """Return the ``(N, 3, 3)`` rotations by ``angles`` around the z axis."""
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    zeros = torch.zeros_like(angles)
    ones = torch.ones_like(angles)
    return torch.stack(
        [cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones],
        dim=1,
    ).reshape(-1, 3, 3)


def _y_rotation_matrices(angles: torch.Tensor) -> torch.Tensor:
    """Return the ``(N, 3, 3)`` rotations by ``angles`` around the y axis."""
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    zeros = torch.zeros_like(angles)
    ones = torch.ones_like(angles)
    return torch.stack(
        [cos, zeros, sin, zeros, ones, zeros, -sin, zeros, cos],
        dim=1,
    ).reshape(-1, 3, 3)


def _z_rotation_wigner_matrices(o3_lambda: int, angles: torch.Tensor) -> torch.Tensor:
    """Return the ``(N, 2l+1, 2l+1)`` real Wigner-D matrices of z rotations."""
    m = torch.arange(
        -o3_lambda,
        o3_lambda + 1,
        dtype=angles.dtype,
        device=angles.device,
    )
    m_angles = angles.reshape(-1, 1) * m.reshape(1, -1)
    diagonal = torch.diag_embed(torch.cos(m_angles))
    # D[m, -m] = -sin(m * angle), on the anti-diagonal
    anti_diagonal = torch.diag_embed(-torch.sin(m_angles)).flip(-1)
    return diagonal + anti_diagonal


class EulerWignerD(torch.nn.Module):
    """
    Rotation matrices and real Wigner-D matrices of rotations given by ZYZ Euler
    angles, computed with TorchScript-compatible operations.

    The rotation for the angles ``(alpha, beta, gamma)`` is ``Rz(alpha) @ Ry(beta) @
    Rz(gamma)``, and its Wigner-D matrices follow the conventions of
    :py:func:`build_packed_wigner_matrices`. They are computed as ``Z(alpha) @ J @
    Z(beta) @ J.T @ Z(gamma)``, where ``Z`` are the Wigner-D matrices of rotations
    around the z axis and ``J`` the Wigner-D matrices of the fixed rotation mapping
    the z axis onto the y axis, which are built in ``__init__``.
    """

    _max_angular_momentum: int

    def __init__(
        self,
        max_angular_momentum: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        """
        :param max_angular_momentum: non-negative maximum angular momentum for which
            Wigner-D matrices are computed
        :param dtype: dtype of the computed matrices
        :param device: device of the computed matrices, defaults to the CPU
        """
        super().__init__()
        self._max_angular_momentum = validate_integer(
            "max_angular_momentum", max_angular_momentum, 0
        )

        # rotation by -pi/2 around the x axis
        z_to_y = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]],
            dtype=torch.float64,
        )
        packed = build_packed_wigner_matrices(z_to_y, self._max_angular_momentum)
        if device is None:
            device = torch.device("cpu")
        self.register_buffer("_z_to_y_wigner", packed.to(dtype=dtype, device=device))

    @property
    def max_angular_momentum(self) -> int:
        """The maximum angular momentum of the computed Wigner-D matrices."""
        return self._max_angular_momentum

    def _check_angles(self, angles: torch.Tensor) -> None:
        if angles.dim() != 2 or angles.size(1) != 3:
            raise ValueError(
                f"Euler angles have shape {list(angles.shape)}; expected (N, 3)"
            )
        if (
            angles.dtype != self._z_to_y_wigner.dtype
            or angles.device != self._z_to_y_wigner.device
        ):
            raise ValueError(
                "Euler angles and EulerWignerD must have the same dtype and device"
            )

    @torch.jit.export
    def sample_angles(self, n: int) -> torch.Tensor:
        """
        Sample Euler angles of rotations distributed uniformly (according to the Haar
        measure) over SO(3), using the global random number generator.

        :param n: number of rotations to sample
        :return: ``(n, 3)`` tensor of ``(alpha, beta, gamma)`` angles
        """
        uniform = torch.rand(
            (n, 3),
            dtype=self._z_to_y_wigner.dtype,
            device=self._z_to_y_wigner.device,
        )
        alpha = 2.0 * math.pi * uniform[:, 0]
        beta = torch.arccos(1.0 - 2.0 * uniform[:, 1])
        gamma = 2.0 * math.pi * uniform[:, 2]
        return torch.stack([alpha, beta, gamma], dim=1)

    @torch.jit.export
    def rotation_matrices(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Compute the rotation matrices for the given Euler angles.

        :param angles: ``(N, 3)`` tensor of ``(alpha, beta, gamma)`` angles
        :return: ``(N, 3, 3)`` rotation matrices
        """
        self._check_angles(angles)
        return (
            _z_rotation_matrices(angles[:, 0])
            @ _y_rotation_matrices(angles[:, 1])
            @ _z_rotation_matrices(angles[:, 2])
        )

    @torch.jit.export
    def wigner_D_matrices(self, angles: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute the real Wigner-D matrices for the given Euler angles.

        :param angles: ``(N, 3)`` tensor of ``(alpha, beta, gamma)`` angles
        :return: list of ``(N, 2*ell+1, 2*ell+1)`` Wigner-D matrices, for ``ell`` from
            0 to ``max_angular_momentum``
        """
        self._check_angles(angles)
        matrices: List[torch.Tensor] = []
        for o3_lambda in range(self._max_angular_momentum + 1):
            z_to_y = wigner_matrices_for_lambda(self._z_to_y_wigner, 1, o3_lambda)[0]
            y_rotation = (
                z_to_y
                @ _z_rotation_wigner_matrices(o3_lambda, angles[:, 1])
                @ z_to_y.transpose(0, 1)
            )
            matrices.append(
                _z_rotation_wigner_matrices(o3_lambda, angles[:, 0])
                @ y_rotation
                @ _z_rotation_wigner_matrices(o3_lambda, angles[:, 2])
            )
        return matrices
