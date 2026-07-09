from typing import Tuple

import numpy as np


def _import_scipy():
    # deferred: scipy is only needed for quadrature/rotation construction at
    # __init__ time; loading the symmetrized_model package for other helpers
    # should not require it.
    try:
        from scipy.integrate import lebedev_rule
        from scipy.spatial.transform import Rotation
    except ImportError as e:
        raise ImportError(
            "scipy >= 1.15 is required for SymmetrizedModel quadrature construction "
            "(scipy.integrate.lebedev_rule); install it with `pip install scipy`."
        ) from e
    return lebedev_rule, Rotation


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

    lebedev_rule, _ = _import_scipy()
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


def get_rotation_quadrature(
    lebedev_order: int, n_rotations: int, include_inversion: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get rotation matrices and weights for a quadrature on SO(3), optionally
    extended to O(3).

    :param lebedev_order: order of the Lebedev quadrature on the unit sphere
    :param n_rotations: number of in-plane rotations per Lebedev node
    :param include_inversion: if ``True``, extend the quadrature to O(3) by
        appending, for every rotation ``R``, the improper operation ``-R``,
        with halved weights
    :return: rotations of shape (N, 3, 3) and weights of shape (N,), summing
        to 1
    """
    alpha, beta, gamma, weights = get_euler_angles_quadrature(
        lebedev_order, n_rotations
    )
    rotations = _rotations_from_angles(alpha, beta, gamma).as_matrix()
    if include_inversion:
        rotations = np.concatenate([rotations, -rotations], axis=0)
        weights = np.concatenate([0.5 * weights, 0.5 * weights], axis=0)
    return rotations, weights


def _rotations_from_angles(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray
) -> "Rotation":  # noqa: F821 (scipy is imported lazily)
    """
    Compose rotations from ZYZ Euler angles.

    :param alpha: array of alpha angles (M,)
    :param beta: array of beta angles (M,)
    :param gamma: array of gamma angles (K,)
    :return: Rotation object containing all (M*K,) rotations
    """

    _, Rotation = _import_scipy()
    # Compose ZYZ rotations in SO(3)
    Rot = (
        Rotation.from_euler("z", alpha.reshape(-1, 1))
        * Rotation.from_euler("y", beta.reshape(-1, 1))
        * Rotation.from_euler("z", gamma.reshape(-1, 1))
    )

    return Rot
