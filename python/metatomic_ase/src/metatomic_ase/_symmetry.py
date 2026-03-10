from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ._calculator import MetatomicCalculator


import ase  # isort: skip
import ase.calculators.calculator  # isort: skip


class SymmetrizedCalculator(ase.calculators.calculator.Calculator):
    r"""
    Take a MetatomicCalculator and average its predictions to make it (approximately)
    equivariant. Only predictions for energy, forces and stress are supported.

    The default is to average over a quadrature of the orthogonal group O(3) composed
    this way:

    - Lebedev quadrature of the unit sphere (S^2)
    - Equispaced sampling of the unit circle (S^1)
    - Both proper and improper rotations are taken into account by including the
        inversion operation (if ``include_inversion=True``)

    :param base_calculator: the MetatomicCalculator to be symmetrized
    :param l_max: the maximum spherical harmonic degree that the model is expected to
        be able to represent. This is used to choose the quadrature order. If ``0``,
        no rotational averaging will be performed (it can be useful to average only over
        the space group, see ``apply_group_symmetry``).
    :param batch_size: number of rotated systems to evaluate at once. If ``None``, all
        systems will be evaluated at once (this can lead to high memory usage).
    :param include_inversion: if ``True``, the inversion operation will be included in
        the averaging. This is required to average over the full orthogonal group O(3).
    :param apply_space_group_symmetry: if ``True``, the results will be averaged over
        discrete space group of rotations for the input system. The group operations are
        computed with `spglib <https://github.com/spglib/spglib>`_, and the average is
        performed after the O(3) averaging (if any). This has no effect for non-periodic
        systems.
    :param store_rotational_std: if ``True``, the results will contain the standard
        deviation over the different rotations for each property (e.g., ``energy_std``).
    """

    implemented_properties = ["energy", "energies", "forces", "stress", "stresses"]

    def __init__(
        self,
        base_calculator: MetatomicCalculator,
        *,
        l_max: int = 3,
        batch_size: Optional[int] = None,
        include_inversion: bool = True,
        apply_space_group_symmetry: bool = False,
        store_rotational_std: bool = False,
    ) -> None:
        try:
            from scipy.integrate import lebedev_rule  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "scipy is required to use the `SymmetrizedCalculator`, please install "
                "it with `pip install scipy` or `conda install scipy`"
            ) from e

        super().__init__()

        self.base_calculator = base_calculator
        if l_max > 131:
            raise ValueError(
                f"l_max={l_max} is too large, the maximum supported value is 131"
            )
        self.l_max = l_max
        self.include_inversion = include_inversion

        if l_max > 0:
            lebedev_order, n_inplane_rotations = _choose_quadrature(l_max)
            self.quadrature_rotations, self.quadrature_weights = _get_quadrature(
                lebedev_order, n_inplane_rotations, include_inversion
            )
        else:  # no quadrature
            if include_inversion:  # identity and inversion
                self.quadrature_rotations = np.array([np.eye(3), -np.eye(3)])
                self.quadrature_weights = np.array([0.5, 0.5])
            else:  # only the identity
                self.quadrature_rotations = np.array([np.eye(3)])
                self.quadrature_weights = np.array([1.0])

        self.batch_size = (
            batch_size if batch_size is not None else len(self.quadrature_rotations)
        )

        self.store_rotational_std = store_rotational_std
        self.apply_space_group_symmetry = apply_space_group_symmetry

    def calculate(
        self, atoms: ase.Atoms, properties: List[str], system_changes: List[str]
    ) -> None:
        """
        Perform the calculation for the given atoms and properties.

        :param atoms: the :py:class:`ase.Atoms` on which to perform the calculation
        :param properties: list of properties to compute, among ``energy``, ``forces``,
            and ``stress``
        :param system_changes: list of changes to the system since the last call to
            ``calculate``
        """
        super().calculate(atoms, properties, system_changes)
        self.base_calculator.calculate(atoms, properties, system_changes)

        compute_forces_and_stresses = "forces" in properties or "stress" in properties
        per_atom = "energies" in properties

        if len(self.quadrature_rotations) > 0:
            rotated_atoms_list = _rotate_atoms(atoms, self.quadrature_rotations)
            batches = [
                rotated_atoms_list[i : i + self.batch_size]
                for i in range(0, len(rotated_atoms_list), self.batch_size)
            ]
            results: Dict[str, np.ndarray] = {}
            for batch in batches:
                try:
                    batch_results = self.base_calculator.compute_energy(
                        batch,
                        compute_forces_and_stresses,
                        per_atom=per_atom,
                    )
                    for key, value in batch_results.items():
                        results.setdefault(key, [])
                        results[key].extend(
                            [value] if isinstance(value, float) else value
                        )
                except torch.cuda.OutOfMemoryError as e:
                    raise RuntimeError(
                        "Out of memory error encountered during rotational averaging. "
                        "Please reduce the batch size or use lower rotational "
                        "averaging parameters. This can be done by setting the "
                        "`batch_size` and `l_max` parameters while initializing the "
                        "calculator."
                    ) from e

            self.results.update(
                _compute_rotational_average(
                    results,
                    self.quadrature_rotations,
                    self.quadrature_weights,
                    self.store_rotational_std,
                )
            )

        if self.apply_space_group_symmetry:
            # Apply the discrete space group of the system a posteriori
            Q_list, P_list = _get_group_operations(atoms)
            self.results.update(_average_over_group(self.results, Q_list, P_list))


def _choose_quadrature(L_max: int) -> Tuple[int, int]:
    """
    Choose a Lebedev quadrature order and number of in-plane rotations to integrate
    spherical harmonics up to degree ``L_max``.

    :param L_max: maximum spherical harmonic degree
    :return: (lebedev_order, n_inplane_rotations)
    """
    # fmt: off
    available = [
        3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41,
        47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131,
    ]
    # fmt: on

    # pick smallest order >= L_max
    n = min(o for o in available if o >= L_max)
    # minimal gamma count
    K = 2 * L_max + 1
    return n, K


def _rotate_atoms(atoms: ase.Atoms, rotations: List[np.ndarray]) -> List[ase.Atoms]:
    """
    Create a list of copies of ``atoms``, rotated by each of the given ``rotations``.

    :param atoms: the :py:class:`ase.Atoms` to be rotated
    :param rotations: (N, 3, 3) array of orthogonal matrices
    :return: list of N :py:class:`ase.Atoms`, each rotated by the corresponding matrix
    """
    rotated_atoms_list = []
    has_cell = atoms.cell is not None and atoms.cell.rank > 0
    for rot in rotations:
        new_atoms = atoms.copy()
        new_atoms.positions = new_atoms.positions @ rot.T
        if has_cell:
            new_atoms.set_cell(
                new_atoms.cell.array @ rot.T, scale_atoms=False, apply_constraint=False
            )
            new_atoms.wrap()
        rotated_atoms_list.append(new_atoms)
    return rotated_atoms_list


def _get_quadrature(lebedev_order: int, n_rotations: int, include_inversion: bool):
    """
    Lebedev(S^2) x uniform angle quadrature on SO(3).
    If include_inversion=True, extend to O(3) by adding inversion * R.

    :param lebedev_order: order of the Lebedev quadrature on the unit sphere
    :param n_rotations: number of in-plane rotations per Lebedev node
    :param include_inversion: if ``True``, include the inversion operation in the
        quadrature
    :return: (N, 3, 3) array of orthogonal matrices, and (N,) array of weights
        associated to each matrix
    """
    from scipy.integrate import lebedev_rule

    # Lebedev nodes (X: (3, M))
    X, w = lebedev_rule(lebedev_order)  # w sums to 4*pi
    x, y, z = X
    alpha = np.arctan2(y, x)  # (M,)
    beta = np.arccos(z)  # (M,)
    # beta = np.arccos(np.clip(z, -1.0, 1.0))  # (M,)

    K = int(n_rotations)
    gamma = np.linspace(0.0, 2 * np.pi, K, endpoint=False)  # (K,)

    Rot = _rotations_from_angles(alpha, beta, gamma)
    R_so3 = Rot.as_matrix()  # (N, 3, 3)

    # SO(3) Haar–probability weights: w_i/(4*pi*K), repeated over gamma
    w_so3 = np.repeat(w / (4 * np.pi * K), repeats=gamma.size)  # (N,)

    if not include_inversion:
        return R_so3, w_so3

    # Extend to O(3) by appending inversion * R
    P = -np.eye(3)
    R_o3 = np.concatenate([R_so3, P @ R_so3], axis=0)  # (2N, 3, 3)
    w_o3 = np.concatenate([0.5 * w_so3, 0.5 * w_so3], axis=0)

    return R_o3, w_o3


def _rotations_from_angles(alpha, beta, gamma):
    from scipy.spatial.transform import Rotation

    # Build all combinations (alpha_i, beta_i, gamma_j)
    A = np.repeat(alpha, gamma.size).reshape(-1, 1)  # (N, 1)
    B = np.repeat(beta, gamma.size).reshape(-1, 1)  # (N, 1)
    G = np.tile(gamma, alpha.size).reshape(-1, 1)  # (N, 1)

    # Compose ZYZ rotations in SO(3)
    Rot = (
        Rotation.from_euler("z", A)
        * Rotation.from_euler("y", B)
        * Rotation.from_euler("z", G)
    )

    return Rot


def _compute_rotational_average(results, rotations, weights, store_std):
    R = rotations
    B = R.shape[0]
    w = weights
    w = w / w.sum()

    def _wreshape(x):
        return w.reshape((B,) + (1,) * (x.ndim - 1))

    def _wmean(x):
        return np.sum(_wreshape(x) * x, axis=0)

    def _wstd(x):
        mu = _wmean(x)
        return np.sqrt(np.sum(_wreshape(x) * (x - mu) ** 2, axis=0))

    out = {}

    # Energy (B,)
    if "energy" in results:
        E = np.asarray(results["energy"], dtype=float)  # (B,)
        out["energy"] = _wmean(E)  # ()
        if store_std:
            out["energy_rot_std"] = _wstd(E)  # ()

    if "energies" in results:
        E = np.asarray(results["energies"], dtype=float)  # (B,N)
        out["energies"] = _wmean(E)  # (N,)
        if store_std:
            out["energies_rot_std"] = _wstd(E)  # (N,)

    # Forces (B,N,3) from rotated structures: back-rotate with F' R
    if "forces" in results:
        F = np.asarray(results["forces"], dtype=float)  # (B,N,3)
        F_back = F @ R  # F' R
        out["forces"] = _wmean(F_back)  # (N,3)
        if store_std:
            out["forces_rot_std"] = _wstd(F_back)  # (N,3)

    # Stress (B,3,3) from rotated structures: back-rotate with R^T S' R
    if "stress" in results:
        S = np.asarray(results["stress"], dtype=float)  # (B,3,3)
        RT = np.swapaxes(R, 1, 2)
        S_back = RT @ S @ R  # R^T S' R
        out["stress"] = _wmean(S_back)  # (3,3)
        if store_std:
            out["stress_rot_std"] = _wstd(S_back)  # (3,3)

    if "stresses" in results:
        S = np.asarray(results["stresses"], dtype=float)  # (B,N,3,3)
        RT = np.swapaxes(R, 1, 2)
        S_back = RT[:, None, :, :] @ S @ R[:, None, :, :]  # R^T S' R
        out["stresses"] = _wmean(S_back)  # (N,3,3)
        if store_std:
            out["stresses_rot_std"] = _wstd(S_back)  # (N,3,3)

    return out


def _get_group_operations(
    atoms: ase.Atoms, symprec: float = 1e-6, angle_tolerance: float = -1.0
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract point-group rotations Q_g (Cartesian, 3x3) and the corresponding
    atom-index permutations P_g (N x N) induced by the space-group operations.
    Returns Q_list, Cartesian rotation matrices of the point group,
    and P_list, permutation matrices mapping original indexing -> indexing after (R,t),

    :param atoms: input structure
    :param symprec: tolerance for symmetry finding
    :param angle_tolerance: tolerance for symmetry finding (in degrees). If less than 0,
        a value depending on ``symprec`` will be chosen automatically by spglib.
    :return: List of rotation matrices and permutation matrices.

    """
    try:
        import spglib
    except ImportError as e:
        raise ImportError(
            "spglib is required to use the SymmetrizedCalculator with "
            "`apply_group_symmetry=True`. Please install it with "
            "`pip install spglib` or `conda install -c conda-forge spglib`"
        ) from e

    if not (atoms.pbc.all()):
        # No periodic boundary conditions: no symmetry
        return [], []

    # Lattice with column vectors a1,a2,a3 (spglib expects (cell, frac, Z))
    A = atoms.cell.array.T  # (3,3)
    frac = atoms.get_scaled_positions()  # (N,3) in [0,1)
    numbers = atoms.numbers
    N = len(atoms)

    data = spglib.get_symmetry_dataset(
        (atoms.cell.array, frac, numbers),
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        _throw=True,
    )

    if data is None:
        # No symmetry found
        return [], []
    R_frac = data.rotations  # (n_ops, 3,3), integer
    t_frac = data.translations  # (n_ops, 3)
    Z = numbers

    # Match fractional coords modulo 1 within a tolerance, respecting chemical species
    def _match_index(x_new, frac_ref, Z_ref, Z_i, tol=1e-6):
        d = np.abs(frac_ref - x_new)  # (N,3)
        d = np.minimum(d, 1.0 - d)  # periodic distance
        # Mask by identical species
        mask = Z_ref == Z_i
        if not np.any(mask):
            raise RuntimeError("No matching species found while building permutation.")
        # Choose argmin over max-norm within species
        idx = np.where(mask)[0]
        j = idx[np.argmin(np.max(d[idx], axis=1))]

        # Sanity check
        if np.max(d[j]) > tol:
            raise RuntimeError(
                (
                    f"Sanity check failed in _match_index: max distance {np.max(d[j])} "
                    f"exceeds tolerance {tol}."
                )
            )
        return j

    Q_list, P_list = [], []
    seen = set()
    Ainv = np.linalg.inv(A)

    for Rf, tf in zip(R_frac, t_frac, strict=False):
        # Cartesian rotation: Q = A Rf A^{-1}
        Q = A @ Rf @ Ainv
        # Deduplicate rotations (point group) by rounding
        key = tuple(np.round(Q.flatten(), 12))
        if key in seen:
            continue
        seen.add(key)

        # Build the permutation P from i to j
        P = np.zeros((N, N), dtype=int)
        new_frac = (frac @ Rf.T + tf) % 1.0  # images after (Rf,tf)
        for i in range(N):
            j = _match_index(new_frac[i], frac, Z, Z[i])
            P[j, i] = 1  # column i maps to row j

        Q_list.append(Q.astype(float))
        P_list.append(P)

    return Q_list, P_list


def _average_over_group(
    results: dict, Q_list: List[np.ndarray], P_list: List[np.ndarray]
) -> dict:
    """
    Apply the point-group projector in output space.

    :param results: Must contain 'energy' (scalar), and/or 'forces' (N,3), and/or
        'stress' (3,3). These are predictions for the current structure in the reference
        frame.
    :param Q_list: Rotation matrices of the point group, from
        :py:func:`_get_group_operations`
    :param P_list: Permutation matrices of the point group, from
        :py:func:`_get_group_operations`
    :return out: Projected quantities.
    """
    m = len(Q_list)
    if m == 0:
        return results  # nothing to do

    out = {}
    # Energy: unchanged by the projector (scalar)
    if "energy" in results:
        out["energy"] = float(results["energy"])

    # Forces: (N,3) row-vectors; projector: (1/|G|) \sum_g P_g^T F Q_g
    if "forces" in results:
        F = np.asarray(results["forces"], float)
        if F.ndim != 2 or F.shape[1] != 3:
            raise ValueError(f"'forces' must be (N,3), got {F.shape}")
        acc = np.zeros_like(F)
        for Q, P in zip(Q_list, P_list, strict=False):
            acc += P.T @ (F @ Q)
        out["forces"] = acc / m

    # Stress: (3,3); projector: (1/|G|) \sum_g Q_g^T S Q_g
    if "stress" in results:
        S = np.asarray(results["stress"], float)
        if S.shape != (3, 3):
            raise ValueError(f"'stress' must be (3,3), got {S.shape}")
        # S = 0.5 * (S + S.T)  # symmetrize just in case
        acc = np.zeros_like(S)
        for Q in Q_list:
            acc += Q.T @ S @ Q
        S_pg = acc / m
        out["stress"] = S_pg

    return out
