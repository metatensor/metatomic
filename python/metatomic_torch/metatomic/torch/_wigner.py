"""Private Wigner-d/Wigner-D helpers for symmetry operations.

Adapted from the `spherical` project (MIT license), primarily from
`spherical/recursions/wignerH.py`, `spherical/utilities/indexing.py`, and the
Wigner-D assembly logic in `spherical/wigner.py`.

This reduced metatomic copy keeps only the recurrence-based pieces needed to build
real Wigner-D matrices from ZYZ Euler angles. It intentionally does not depend on
`spinsfast`, `quaternionic`, or the public `spherical` package.
"""

from typing import Dict, Tuple

import numpy as np
import torch

from ._jit_compat import jit


@jit
def _epsilon(m: int) -> int:
    if m <= 0:
        return 1
    if m % 2:
        return -1
    return 1


@jit
def _nm_index(n: int, m: int) -> int:
    return m + n * (n + 1)


@jit
def _nabsm_index(n: int, absm: int) -> int:
    return absm + (n * (n + 1)) // 2


@jit
def _wigner_h_size(mp_max: int, ell_max: int) -> int:
    if ell_max < 0:
        return 0
    if mp_max >= ell_max:
        return (ell_max + 1) * (ell_max + 2) * (2 * ell_max + 3) // 6

    return (
        (ell_max + 1) * (ell_max + 2) * (2 * ell_max + 3)
        - 2 * (ell_max - mp_max) * (ell_max - mp_max + 1) * (ell_max - mp_max + 2)
    ) // 6


@jit
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


@jit
def _wigner_h_index_base(ell: int, mp: int, m: int, mp_max: int) -> int:
    local_mp_max = mp_max
    if local_mp_max > ell:
        local_mp_max = ell
    idx = _wigner_h_size(local_mp_max, ell - 1)
    if mp < 1:
        idx += (local_mp_max + mp) * (2 * ell - local_mp_max + mp + 1) // 2
    else:
        idx += (local_mp_max + 1) * (2 * ell - local_mp_max + 2) // 2
        idx += (mp - 1) * (2 * ell - mp + 2) // 2
    idx += m - abs(mp)
    return idx


@jit
def _wigner_h_index(ell: int, mp: int, m: int, mp_max: int) -> int:
    if ell == 0:
        return 0

    local_mp_max = mp_max
    if local_mp_max > ell:
        local_mp_max = ell

    if m < -mp:
        if m < mp:
            return _wigner_h_index_base(ell, -mp, -m, local_mp_max)
        return _wigner_h_index_base(ell, -m, -mp, local_mp_max)

    if m < mp:
        return _wigner_h_index_base(ell, m, mp, local_mp_max)
    return _wigner_h_index_base(ell, mp, m, local_mp_max)


@jit
def _wigner_d_index(ell: int, mp: int, m: int, ell_min: int, mp_max: int) -> int:
    idx = 0
    for ell_prev in range(ell_min, ell):
        local_mp_max = mp_max if mp_max < ell_prev else ell_prev
        idx += (2 * local_mp_max + 1) * (2 * ell_prev + 1)

    local_mp_max = mp_max if mp_max < ell else ell
    idx += (mp + local_mp_max) * (2 * ell + 1)
    idx += m + ell
    return idx


@jit
def _step_1(hwedge):
    hwedge[0] = 1.0


@jit
def _step_2(g, h, n_max, mp_max, hwedge, hextra, hv, expi_beta):
    cos_beta = expi_beta.real
    sin_beta = expi_beta.imag
    sqrt3 = np.sqrt(3.0)
    inverse_sqrt2 = 1.0 / np.sqrt(2.0)
    if n_max > 0:
        n0n_index = _wigner_h_index(1, 0, 1, mp_max)
        nn_index = _nm_index(1, 1)
        hwedge[n0n_index] = sqrt3
        hwedge[n0n_index - 1] = (g[nn_index - 1] * cos_beta) * inverse_sqrt2
        for n in range(2, n_max + 2):
            if n <= n_max:
                n0n_index = _wigner_h_index(n, 0, n, mp_max)
                out = hwedge
            else:
                n0n_index = n
                out = hextra
            prev_index = _wigner_h_index(n - 1, 0, n - 1, mp_max)
            nn_index = _nm_index(n, n)
            const = np.sqrt(1.0 + 0.5 / n)
            g_i = g[nn_index - 1]
            out[n0n_index] = const * hwedge[prev_index]
            out[n0n_index - 1] = g_i * cos_beta * out[n0n_index]
            for i in range(2, n):
                g_i = g[nn_index - i]
                h_i = h[nn_index - i]
                out[n0n_index - i] = (
                    g_i * cos_beta * out[n0n_index - i + 1]
                    - h_i * sin_beta**2 * out[n0n_index - i + 2]
                )
            const = 1.0 / np.sqrt(4 * n + 2)
            g_i = g[nn_index - n]
            h_i = h[nn_index - n]
            out[n0n_index - n] = (
                g_i * cos_beta * out[n0n_index - n + 1]
                - h_i * sin_beta**2 * out[n0n_index - n + 2]
            ) * const
            prefactor = const
            for i in range(1, n):
                prefactor *= sin_beta
                out[n0n_index - n + i] *= prefactor
            if n <= n_max:
                hv[_nm_index(n, 1)] = hwedge[_wigner_h_index(n, 0, 1, mp_max)]
                hv[_nm_index(n, 0)] = hwedge[_wigner_h_index(n, 0, 1, mp_max)]
        prefactor = 1.0
        for n in range(1, n_max + 1):
            prefactor *= sin_beta
            hwedge[_wigner_h_index(n, 0, n, mp_max)] *= prefactor / np.sqrt(4 * n + 2)
        prefactor *= sin_beta
        hextra[n_max + 1] *= prefactor / np.sqrt(4 * (n_max + 1) + 2)
        hv[_nm_index(1, 1)] = hwedge[_wigner_h_index(1, 0, 1, mp_max)]
        hv[_nm_index(1, 0)] = hwedge[_wigner_h_index(1, 0, 1, mp_max)]


@jit
def _step_3(a, b, n_max, mp_max, hwedge, hextra, expi_beta):
    cos_beta = expi_beta.real
    sin_beta = expi_beta.imag
    if n_max > 0 and mp_max > 0:
        for n in range(1, n_max + 1):
            i1 = _wigner_h_index(n, 1, 1, mp_max)
            if n + 1 <= n_max:
                i2 = _wigner_h_index(n + 1, 0, 0, mp_max)
                h2 = hwedge
            else:
                i2 = 0
                h2 = hextra
            i3 = _nm_index(n + 1, 0)
            i4 = _nabsm_index(n, 1)
            inverse_b5 = 1.0 / b[i3]
            for i in range(n):
                b6 = b[-i + i3 - 2]
                b7 = b[i + i3]
                a8 = a[i + i4]
                hwedge[i + i1] = inverse_b5 * (
                    0.5
                    * (
                        b6 * (1 - cos_beta) * h2[i + i2 + 2]
                        - b7 * (1 + cos_beta) * h2[i + i2]
                    )
                    - a8 * sin_beta * h2[i + i2 + 1]
                )


@jit
def _step_4(d, n_max, mp_max, hwedge, hv):
    if n_max > 0 and mp_max > 0:
        for n in range(2, n_max + 1):
            for mp in range(1, min(n, mp_max)):
                i1 = _wigner_h_index(n, mp + 1, mp + 1, mp_max) - 1
                i2 = _wigner_h_index(n, mp - 1, mp, mp_max)
                i3 = _wigner_h_index(n, mp, mp, mp_max) - 1
                i4 = _wigner_h_index(n, mp, mp + 1, mp_max)
                i5 = _nm_index(n, mp)
                i6 = _nm_index(n, mp - 1)
                inverse_d5 = 1.0 / d[i5]
                d6 = d[i6]
                hv[_nm_index(n, mp + 1)] = inverse_d5 * (
                    d6 * hwedge[i2] - d[i6] * hv[_nm_index(n, mp)] + d[i5] * hwedge[i4]
                )
                for i in range(1, n - mp):
                    d7 = d[i + i6]
                    d8 = d[i + i5]
                    hwedge[i + i1] = inverse_d5 * (
                        d6 * hwedge[i + i2] - d7 * hwedge[i + i3] + d8 * hwedge[i + i4]
                    )
                i = n - mp
                hwedge[i + i1] = inverse_d5 * (
                    d6 * hwedge[i + i2] - d[i + i6] * hwedge[i + i3]
                )


@jit
def _step_5(d, n_max, mp_max, hwedge, hv):
    if n_max > 0 and mp_max > 0:
        for n in range(0, n_max + 1):
            for mp in range(0, -min(n, mp_max), -1):
                i1 = _wigner_h_index(n, mp - 1, -mp + 1, mp_max) - 1
                i2 = _wigner_h_index(n, mp + 1, -mp + 1, mp_max) - 1
                i3 = _wigner_h_index(n, mp, -mp, mp_max) - 1
                i4 = _wigner_h_index(n, mp, -mp + 1, mp_max)
                i5 = _nm_index(n, mp - 1)
                i6 = _nm_index(n, mp)
                i7 = _nm_index(n, -mp - 1)
                i8 = _nm_index(n, -mp)
                inverse_d5 = 1.0 / d[i5]
                d6 = d[i6]
                d7 = d[i7]
                d8 = d[i8]
                if mp == 0:
                    hv[_nm_index(n, mp - 1)] = inverse_d5 * (
                        d6 * hv[_nm_index(n, mp + 1)]
                        + d7 * hv[_nm_index(n, mp)]
                        - d8 * hwedge[i4]
                    )
                else:
                    hv[_nm_index(n, mp - 1)] = inverse_d5 * (
                        d6 * hwedge[i2] + d7 * hv[_nm_index(n, mp)] - d8 * hwedge[i4]
                    )
                for i in range(1, n + mp):
                    d7 = d[i + i7]
                    d8 = d[i + i8]
                    hwedge[i + i1] = inverse_d5 * (
                        d6 * hwedge[i + i2] + d7 * hwedge[i + i3] - d8 * hwedge[i + i4]
                    )
                i = n + mp
                hwedge[i + i1] = inverse_d5 * (
                    d6 * hwedge[i + i2] + d[i + i7] * hwedge[i + i3]
                )


def _create_wigner_coefficients(ell_max: int):
    n = np.array([n for n in range(ell_max + 2) for _ in range(-n, n + 1)])
    m = np.array([m for n in range(ell_max + 2) for m in range(-n, n + 1)])
    absn = np.array([n for n in range(ell_max + 2) for _ in range(n + 1)])
    absm = np.array([m for n in range(ell_max + 2) for m in range(n + 1)])

    a = np.sqrt(
        (absn + 1 + absm) * (absn + 1 - absm) / ((2 * absn + 1) * (2 * absn + 3))
    )
    b = np.sqrt((n - m - 1) * (n - m) / ((2 * n - 1) * (2 * n + 1)))
    b[m < 0] *= -1
    d = 0.5 * np.sqrt((n - m) * (n + m + 1))
    d[m < 0] *= -1
    with np.errstate(divide="ignore", invalid="ignore"):
        g = 2 * (m + 1) / np.sqrt((n - m) * (n + m + 1))
        h = np.sqrt((n + m + 2) * (n - m - 1) / ((n - m) * (n + m + 1)))
    return a, b, d, g, h


def _complex_powers(z: complex, ell_max: int) -> np.ndarray:
    powers = np.empty(ell_max + 1, dtype=np.complex128)
    powers[0] = 1.0 + 0.0j
    for idx in range(1, ell_max + 1):
        powers[idx] = powers[idx - 1] * z
    return powers


def _to_euler_phases(
    alpha: float, beta: float, gamma: float
) -> Tuple[complex, complex, complex]:
    # Match spherical.Wigner's convention after converting scipy's ZYZ Euler angles
    # into the phases used by the recurrence.
    z_alpha = np.exp(-1j * alpha)
    expi_beta = np.exp(1j * beta)
    z_gamma = np.exp(-1j * gamma)
    return z_alpha, expi_beta, z_gamma


def _compute_wigner_d_complex(
    ell_max: int, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray
) -> np.ndarray:
    if not (alpha.shape == beta.shape == gamma.shape):
        raise ValueError("alpha, beta, and gamma must have identical shapes")

    mp_max = ell_max
    a, b, d, g, h = _create_wigner_coefficients(ell_max)
    hsize = _wigner_h_size(mp_max, ell_max)
    dsize = _wigner_d_size(0, mp_max, ell_max)
    result = np.zeros(alpha.shape + (dsize,), dtype=np.complex128)

    for index in np.ndindex(alpha.shape):
        z_alpha, expi_beta, z_gamma = _to_euler_phases(
            float(alpha[index]), float(beta[index]), float(gamma[index])
        )
        hwedge = np.zeros(hsize, dtype=np.float64)
        hv = np.zeros((ell_max + 1) ** 2, dtype=np.float64)
        hextra = np.zeros(ell_max + 2, dtype=np.float64)

        _step_1(hwedge)
        _step_2(g, h, ell_max, mp_max, hwedge, hextra, hv, expi_beta)
        _step_3(a, b, ell_max, mp_max, hwedge, hextra, expi_beta)
        _step_4(d, ell_max, mp_max, hwedge, hv)
        _step_5(d, ell_max, mp_max, hwedge, hv)

        z_alpha_powers = _complex_powers(z_alpha, ell_max)
        z_gamma_powers = _complex_powers(z_gamma, ell_max)
        out = result[index]
        for ell in range(0, ell_max + 1):
            for mp in range(-ell, 0):
                i_d = _wigner_d_index(ell, mp, -ell, 0, mp_max)
                for m in range(-ell, 0):
                    i_h = _wigner_h_index(ell, mp, m, mp_max)
                    out[i_d] = (
                        _epsilon(mp)
                        * _epsilon(-m)
                        * hwedge[i_h]
                        * z_gamma_powers[-m].conjugate()
                        * z_alpha_powers[-mp].conjugate()
                    )
                    i_d += 1
                for m in range(0, ell + 1):
                    i_h = _wigner_h_index(ell, mp, m, mp_max)
                    out[i_d] = (
                        _epsilon(mp)
                        * _epsilon(-m)
                        * hwedge[i_h]
                        * z_gamma_powers[m]
                        * z_alpha_powers[-mp].conjugate()
                    )
                    i_d += 1
            for mp in range(0, ell + 1):
                i_d = _wigner_d_index(ell, mp, -ell, 0, mp_max)
                for m in range(-ell, 0):
                    i_h = _wigner_h_index(ell, mp, m, mp_max)
                    out[i_d] = (
                        _epsilon(mp)
                        * _epsilon(-m)
                        * hwedge[i_h]
                        * z_gamma_powers[-m].conjugate()
                        * z_alpha_powers[mp]
                    )
                    i_d += 1
                for m in range(0, ell + 1):
                    i_h = _wigner_h_index(ell, mp, m, mp_max)
                    out[i_d] = (
                        _epsilon(mp)
                        * _epsilon(-m)
                        * hwedge[i_h]
                        * z_gamma_powers[m]
                        * z_alpha_powers[mp]
                    )
                    i_d += 1

    return result


def compute_complex_wigner_d_matrices(
    ell_max: int,
    angles: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict[int, np.ndarray]:
    alpha, beta, gamma = angles
    raw = _compute_wigner_d_complex(ell_max, alpha, beta, gamma)
    matrices: Dict[int, np.ndarray] = {}
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
    angles: Tuple[np.ndarray, np.ndarray, np.ndarray],
    complex_to_real: Dict[int, np.ndarray],
) -> Dict[int, torch.Tensor]:
    complex_matrices = compute_complex_wigner_d_matrices(ell_max, angles)
    real_matrices: Dict[int, torch.Tensor] = {}
    for ell, matrix in complex_matrices.items():
        transform = complex_to_real[ell]
        matrix = np.einsum("ij,...jk,kl->...il", transform.conj(), matrix, transform.T)
        if not np.allclose(matrix.imag, 0.0):
            raise ValueError("real Wigner matrix conversion produced complex values")
        real_matrices[ell] = torch.from_numpy(matrix.real)
    return real_matrices
