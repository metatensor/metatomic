import operator
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ._calculator import (
    PER_ATOM_QUANTITIES,
    MetatomicCalculator,
    _append_requested_state_changes,
    _snapshot_requested_info_values,
)


import ase  # isort: skip
import ase.calculators.calculator  # isort: skip
from ase.geometry import find_mic  # isort: skip


_PerAtomFeature = Tuple[str, np.ndarray, str]


@dataclass
class _RotationalMomentState:
    """Reference-centered sufficient statistics for one predicted property."""

    reference: np.ndarray
    centered_sum: np.ndarray
    centered_second: Optional[np.ndarray]
    absolute_centered_sum: Optional[np.ndarray]
    absolute_centered_second: Optional[np.ndarray]


class _RotationalAverageAccumulator:
    """Stream prediction batches into reference-centered weighted moments.

    For a fixed reference ``c = x[0]``, this accumulates
    ``m = sum_i w[i] * (x[i] - c)`` and
    ``q = sum_i w[i] * (x[i] - c)**2``. The final mean and variance are
    ``c + m`` and ``q - m**2``. Absolute-weight moments provide the scale used
    to distinguish round-off from a materially negative variance when the
    quadrature contains signed weights.

    Calls to :meth:`update` must contain consecutive rotations in quadrature
    order because weights are selected from the number of predictions already
    consumed. Retained prediction state is one array per property and is
    independent of the number of quadrature rotations.
    """

    _SUPPORTED_PROPERTIES = ("energy", "energies", "forces", "stress")

    def __init__(
        self,
        normalized_weights: np.ndarray,
        *,
        store_std: bool,
    ) -> None:
        weights = np.asarray(normalized_weights, dtype=float).copy()
        if weights.ndim != 1 or len(weights) == 0:
            raise ValueError("rotational weights must be a non-empty 1D array")
        if not np.all(np.isfinite(weights)):
            raise ValueError("rotational weights must be finite")
        weight_sum = np.sum(weights, dtype=np.float64)
        if not np.isfinite(weight_sum) or weight_sum == 0:
            raise ValueError("rotational weights must have a finite nonzero sum")
        weights /= weight_sum
        # Make normalization exact so the reference does not leak into the sum.
        weights[-1] += 1.0 - np.sum(weights, dtype=np.float64)

        self._weights = weights
        self._store_std = store_std
        self._states: Dict[str, _RotationalMomentState] = {}
        self._property_names: Optional[Tuple[str, ...]] = None
        self._property_shapes: Dict[str, Tuple[int, ...]] = {}
        self._seen = 0
        self._absolute_weight_sum = 0.0

    @staticmethod
    def _as_batch(name: str, value, batch_size: int) -> np.ndarray:
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            if batch_size != 1:
                raise ValueError(
                    f"property '{name}' returned a scalar for a batch of "
                    f"{batch_size} systems"
                )
            values = values.reshape(1)
        if values.shape[0] != batch_size:
            raise ValueError(
                f"property '{name}' returned {values.shape[0]} systems for a "
                f"batch of {batch_size}"
            )
        return values

    @staticmethod
    def _back_rotate(
        name: str, values: np.ndarray, rotations: np.ndarray
    ) -> np.ndarray:
        if name == "forces":
            return values @ rotations
        if name == "stress":
            return np.swapaxes(rotations, 1, 2) @ values @ rotations
        return values

    def update(
        self,
        batch_results: Dict[str, object],
        batch_rotations: np.ndarray,
    ) -> None:
        """Consume one consecutive batch from the configured quadrature."""
        rotations = np.asarray(batch_rotations, dtype=float)
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError("batch rotations must have shape (batch, 3, 3)")
        batch_size = len(rotations)
        batch_stop = self._seen + batch_size
        if batch_stop > len(self._weights):
            raise ValueError("received more predictions than quadrature rotations")

        property_names = tuple(
            name for name in self._SUPPORTED_PROPERTIES if name in batch_results
        )
        if self._property_names is None:
            self._property_names = property_names
        elif property_names != self._property_names:
            raise ValueError(
                "base calculator returned inconsistent properties across "
                "rotation batches"
            )

        weights = self._weights[self._seen : batch_stop]
        absolute_weights = np.abs(weights)
        self._absolute_weight_sum += float(np.sum(absolute_weights))
        for name in property_names:
            values = self._as_batch(name, batch_results[name], batch_size)
            property_shape = tuple(values.shape[1:])
            expected_shape = self._property_shapes.setdefault(name, property_shape)
            if property_shape != expected_shape:
                raise ValueError(
                    f"property '{name}' changed shape across rotation batches: "
                    f"expected {expected_shape}, got {property_shape}"
                )

            values = self._back_rotate(name, values, rotations)

            state = self._states.get(name)
            if state is None:
                reference = np.array(values[0], dtype=float, copy=True)
                zeros = np.zeros_like(reference, dtype=float)
                state = _RotationalMomentState(
                    reference=reference,
                    centered_sum=zeros.copy(),
                    centered_second=zeros.copy() if self._store_std else None,
                    absolute_centered_sum=(zeros.copy() if self._store_std else None),
                    absolute_centered_second=(
                        zeros.copy() if self._store_std else None
                    ),
                )
                self._states[name] = state

            centered = values - state.reference
            weight_shape = (len(weights),) + (1,) * (values.ndim - 1)
            signed_weight = weights.reshape(weight_shape)
            state.centered_sum += np.sum(signed_weight * centered, axis=0)

            if self._store_std:
                assert state.centered_second is not None
                assert state.absolute_centered_sum is not None
                assert state.absolute_centered_second is not None
                absolute_weight = absolute_weights.reshape(weight_shape)
                state.centered_second += np.sum(signed_weight * centered**2, axis=0)
                state.absolute_centered_sum += np.sum(
                    absolute_weight * centered, axis=0
                )
                state.absolute_centered_second += np.sum(
                    absolute_weight * centered**2, axis=0
                )

        self._seen = batch_stop

    def finalize(self) -> Dict[str, np.ndarray]:
        """Return the same property dictionary as a full-grid reduction."""
        if self._seen != len(self._weights):
            raise ValueError(
                f"received {self._seen} predictions for "
                f"{len(self._weights)} quadrature rotations"
            )

        output: Dict[str, np.ndarray] = {}
        for name, state in self._states.items():
            mean = state.reference + state.centered_sum
            output[name] = mean
            if not self._store_std:
                continue

            assert state.centered_second is not None
            assert state.absolute_centered_sum is not None
            assert state.absolute_centered_second is not None
            variance = state.centered_second - state.centered_sum**2

            # Reconstruct sum |w_i| (x_i - mean)^2 around the fixed reference.
            scale = (
                state.absolute_centered_second
                - 2.0 * state.centered_sum * state.absolute_centered_sum
                + state.centered_sum**2 * self._absolute_weight_sum
            )
            scale = np.maximum(scale, 0.0)
            finfo = np.finfo(np.result_type(variance.dtype, np.float64))
            tolerance = (
                64.0 * len(self._weights) * finfo.eps * np.maximum(scale, finfo.tiny)
            )
            if np.any(variance < -tolerance):
                minimum = float(np.min(variance))
                raise ValueError(
                    "rotational variance is materially negative "
                    f"({minimum}); increase l_max and check quadrature convergence"
                )
            output[name + "_rot_std"] = np.sqrt(np.maximum(variance, 0.0))

        return output


def _requested_per_atom_features(
    calculator: MetatomicCalculator,
    atoms: ase.Atoms,
    allow_axial: bool = False,
) -> Tuple[List[_PerAtomFeature], List[str]]:
    """Classify requested per-atom inputs for rotations and group filtering.

    Real numerical or Boolean ``(N, 3)`` arrays are interpreted as polar
    vectors, except for ``initial_magmoms``, which are axial. All other
    per-atom inputs are scalars. The returned array names identify caller-owned
    ASE arrays that must be rotated together with the geometry.
    """
    if not hasattr(calculator, "_model"):
        return [], []

    requested = calculator._model.requested_inputs(use_new_names=True)
    features: List[_PerAtomFeature] = []
    vector_array_names: List[str] = []
    for name, options in requested.items():
        if options.sample_kind != "atom":
            continue

        array_name: Optional[str]
        if name.startswith("ase::arrays::"):
            array_name = name[len("ase::arrays::") :]
            if array_name not in atoms.arrays:
                continue
            values = atoms.arrays[array_name]
        elif name in PER_ATOM_QUANTITIES:
            values = PER_ATOM_QUANTITIES[name]["getter"](atoms)
            array_name = {
                "momentum": "momenta",
                "velocity": "momenta",
                "mass": "masses",
                "charge": "initial_charges",
                "ase::initial_magmoms": "initial_magmoms",
                "ase::initial_charges": "initial_charges",
                "ase::tags": "tags",
            }.get(name)
        else:
            continue

        values = np.asarray(values)
        if values.ndim == 0 or values.shape[0] != len(atoms):
            continue
        if name == "ase::arrays::positions":
            raise ValueError(
                "SymmetrizedCalculator does not support the reserved custom input "
                "'ase::arrays::positions'; models must use System.positions so "
                "forces and stress retain the correct derivatives"
            )
        if np.iscomplexobj(values):
            raise ValueError(f"requested per-atom input '{name}' must be real-valued")
        is_numeric = np.issubdtype(values.dtype, np.number) or np.issubdtype(
            values.dtype, np.bool_
        )
        if is_numeric and not np.all(np.isfinite(values)):
            raise ValueError(
                f"requested per-atom input '{name}' contains non-finite values"
            )
        if values.shape == (len(atoms), 3) and is_numeric:
            is_axial = array_name == "initial_magmoms"
            if is_axial and not allow_axial:
                raise ValueError(
                    "SymmetrizedCalculator can not transform vector "
                    "initial_magmoms consistently: the ASE input converter does "
                    "not expose their axial O(3) parity"
                )
            kind = "axial" if is_axial else "polar"
            if array_name is not None and array_name in atoms.arrays:
                vector_array_names.append(array_name)
        else:
            kind = "scalar"
        features.append((name, values, kind))

    return features, sorted(set(vector_array_names))


class SymmetrizedCalculator(ase.calculators.calculator.Calculator):
    r"""
    Take a MetatomicCalculator and average its predictions to make it (approximately)
    equivariant. Only predictions for total/per-atom energy, forces, and stress are
    supported.

    The finite O(3) grid combines a Lebedev sphere rule, equispaced in-plane
    rotations, and (when requested) the improper coset.

    :param base_calculator: the MetatomicCalculator to be symmetrized
    :param l_max: effective angular bandwidth used to choose the finite product
        quadrature. It must cover both the model response and the target
        representation (at least 1 for forces and 2 for stress); check convergence by
        increasing it. Componentwise ``*_rot_std`` must cover the complete
        back-rotated response. The largest supported value is 65. At ``0``, only
        identity and, when requested, inversion are used.
    :param batch_size: number of rotated systems to evaluate at once. If ``None``, all
        rotations are evaluated at once. Predictions are reduced immediately, so a
        finite value bounds the live model-prediction payload; ``None`` evaluates the
        full grid in one batch.
    :param include_inversion: if ``True``, every proper grid rotation is paired with
        its inverted partner, adding the full improper coset required to average over
        O(3). Vector ``initial_magmoms`` are accepted only when this is ``False``,
        because their axial parity is not represented by the ASE input TensorMap.
    :param apply_space_group_symmetry: if ``True``, fully periodic structures are
        averaged over the discrete space group after O(3) averaging, using
        `spglib <https://github.com/spglib/spglib>`_. This step is skipped for other
        structures. Operations that do not preserve requested per-atom inputs are
        discarded.
    :param store_rotational_std: if ``True``, the results will contain the standard
        deviation over the different rotations for each property (e.g.,
        ``energy_rot_std``, ``forces_rot_std``, and ``stress_rot_std``).
    """

    implemented_properties = ["energy", "energies", "forces", "stress"]

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
        super().__init__()

        l_max = _validated_integer("l_max", l_max, 0)
        if batch_size is not None:
            batch_size = _validated_integer("batch_size", batch_size, 1)

        self.base_calculator = base_calculator
        self.l_max = l_max
        self.include_inversion = include_inversion

        if l_max > 0:
            try:
                from scipy.integrate import lebedev_rule  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "scipy is required to use the `SymmetrizedCalculator`, please "
                    "install it with `pip install scipy` or `conda install scipy`"
                ) from e
            lebedev_order, n_inplane_rotations = _choose_quadrature(l_max)
            self.quadrature_rotations, self.quadrature_weights = _get_quadrature(
                lebedev_order, n_inplane_rotations, include_inversion
            )
        else:
            if include_inversion:
                self.quadrature_rotations = np.array([np.eye(3), -np.eye(3)])
                self.quadrature_weights = np.array([0.5, 0.5])
            else:
                self.quadrature_rotations = np.array([np.eye(3)])
                self.quadrature_weights = np.array([1.0])

        self.batch_size = (
            batch_size if batch_size is not None else len(self.quadrature_rotations)
        )

        self.store_rotational_std = store_rotational_std
        self.apply_space_group_symmetry = apply_space_group_symmetry

        self._per_atom_array_watch = getattr(
            base_calculator, "_per_atom_array_watch", []
        )
        self._system_info_watch = getattr(base_calculator, "_system_info_watch", [])

    def check_state(self, atoms: ase.Atoms, tol: float = 1e-15) -> List[str]:
        """Detect changes in every ASE value requested by the wrapped model."""
        changes = super().check_state(atoms, tol=tol)
        return _append_requested_state_changes(
            changes,
            self.atoms,
            atoms,
            self._per_atom_array_watch,
            self._system_info_watch,
            tol,
        )

    def calculate(
        self, atoms: ase.Atoms, properties: List[str], system_changes: List[str]
    ) -> None:
        """Average the requested properties over the configured operations."""
        super().calculate(atoms, properties, system_changes)
        # A later failure must not leave old results attached to the new atoms.
        self.results = {}

        compute_forces_and_stresses = "forces" in properties or "stress" in properties
        if "stress" in properties:
            if not atoms.pbc.all():
                raise ase.calculators.calculator.PropertyNotImplementedError(
                    "SymmetrizedCalculator only defines 3D stress for systems "
                    "periodic in all three directions"
                )
        # The base computes stress with forces for fully periodic systems.
        if compute_forces_and_stresses and atoms.pbc.all():
            cell = atoms.cell.array
            invalid_cell = not np.all(np.isfinite(cell))
            volume = 0.0 if invalid_cell else abs(np.linalg.det(cell))
            if invalid_cell or not np.isfinite(volume) or volume == 0:
                raise ValueError(
                    "SymmetrizedCalculator can not compute 3D stress for a "
                    "singular or non-finite periodic cell"
                )

        per_atom = "energies" in properties
        per_atom_features, vector_arrays = _requested_per_atom_features(
            self.base_calculator,
            atoms,
            allow_axial=not self.include_inversion,
        )

        accumulator = _RotationalAverageAccumulator(
            self.quadrature_weights,
            store_std=self.store_rotational_std,
        )
        for batch_start in range(0, len(self.quadrature_rotations), self.batch_size):
            batch_rotations = self.quadrature_rotations[
                batch_start : batch_start + self.batch_size
            ]
            batch = _rotate_atoms(atoms, batch_rotations, vector_arrays)
            try:
                batch_results = self.base_calculator.compute_energy(
                    batch,
                    compute_forces_and_stresses,
                    per_atom=per_atom,
                )
                del batch
                accumulator.update(batch_results, batch_rotations)
                del batch_results
            except torch.cuda.OutOfMemoryError as e:
                raise RuntimeError(
                    "Out of memory error encountered during rotational averaging. "
                    "Please reduce the batch size or use lower rotational "
                    "averaging parameters. This can be done by setting the "
                    "`batch_size` and `l_max` parameters while initializing the "
                    "calculator."
                ) from e

        final_results = accumulator.finalize()
        # Release moment state before allocating projection temporaries.
        del accumulator

        if self.apply_space_group_symmetry and any(
            name in final_results for name in ("energies", "forces", "stress")
        ):
            Q_list, permutations = _get_group_operations(
                atoms, per_atom_features=per_atom_features
            )
            projected_results = _average_over_group(final_results, Q_list, permutations)
            # Rotational deviations describe the preceding O(3) orbit.
            final_results.update(projected_results)

        # Break shallow-copy aliases in ASE's calculator-owned cache snapshot.
        assert self.atoms is not None
        _snapshot_requested_info_values(self.atoms, self._system_info_watch)
        self.results = final_results


def _validated_integer(name: str, value, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a boolean")
    try:
        normalized = int(operator.index(value))
    except TypeError as error:
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__}"
        ) from error
    if normalized < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {qualifier}, got {normalized}")
    return normalized


def _choose_quadrature(L_max: int) -> Tuple[int, int]:
    """Choose the product grid for Wigner-D products through ``L_max``."""
    L_max = _validated_integer("L_max", L_max, 0)

    # Keep in sync with metatomic.torch.symmetrized_model._quadrature.
    # fmt: off
    available = [
        3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41,
        47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131,
    ]
    # fmt: on

    required_lebedev_degree = 2 * L_max
    if required_lebedev_degree > available[-1]:
        raise ValueError(
            f"l_max={L_max} requires Lebedev degree {required_lebedev_degree}, "
            f"which exceeds the largest available order ({available[-1]}); "
            f"the largest supported l_max is {available[-1] // 2}"
        )
    n = min(o for o in available if o >= required_lebedev_degree)
    return n, 2 * L_max + 1


def _rotate_atoms(
    atoms: ase.Atoms,
    rotations: List[np.ndarray],
    vector_arrays: Optional[List[str]] = None,
) -> List[ase.Atoms]:
    """Apply active rotations to geometry and requested polar/axial arrays.

    ASE stores Cartesian vectors and cell vectors by row, so an operation
    ``R`` maps each of them to ``x @ R.T``. Coordinates are not wrapped; in
    particular, the identity preserves the caller's selected lattice images.
    """
    if vector_arrays is None:
        vector_arrays = []
    rotated_atoms_list = []
    has_cell = atoms.cell is not None and atoms.cell.rank > 0
    for rot in rotations:
        new_atoms = atoms.copy()
        new_atoms.positions = new_atoms.positions @ rot.T
        if has_cell:
            new_atoms.set_cell(
                new_atoms.cell.array @ rot.T, scale_atoms=False, apply_constraint=False
            )
        for name in vector_arrays:
            values = new_atoms.arrays[name]
            new_atoms.arrays[name] = values @ rot.T
        rotated_atoms_list.append(new_atoms)
    return rotated_atoms_list


def _arrays_respect_operation(
    features: List[_PerAtomFeature],
    rotation: np.ndarray,
    permutation: np.ndarray,
    determinant: int,
) -> bool:
    """Whether all requested per-atom inputs respect a space-group action."""
    for _, values, kind in features:
        transformed = values
        if kind in ("polar", "axial"):
            transformed = transformed @ rotation.T
            if kind == "axial":
                parity = determinant
                transformed = parity * transformed

        mapped = values[permutation]
        if kind in ("polar", "axial"):
            # Bound matrix-arithmetic error component by component. Values are
            # compared as represented; adding the source dtype's epsilon would
            # turn nearby but distinct float32 features into false symmetries.
            # A global/unit-sized absolute floor would likewise erase a
            # physically meaningful tiny vector or a small component next to a
            # much larger one.
            operation_scale = np.abs(values) @ np.abs(rotation).T
            if kind == "axial":
                operation_scale *= abs(parity)
            arithmetic_dtype = np.result_type(values.dtype, rotation.dtype)
            arithmetic_eps = np.finfo(arithmetic_dtype).eps
            tolerance = 64.0 * arithmetic_eps * operation_scale
            if not np.all(np.abs(mapped - transformed) <= tolerance):
                return False
        elif not np.array_equal(mapped, transformed):
            return False
    return True


def _group_actions_are_closed(
    fractional_rotations: List[np.ndarray],
    permutations: List[np.ndarray],
) -> bool:
    """Check closure by finding generators rather than all action pairs."""
    if len(fractional_rotations) != len(permutations) or len(permutations) == 0:
        return False

    def action_key(rotation, permutation):
        return (tuple(rotation.flatten().tolist()), tuple(permutation.tolist()))

    actions = {
        action_key(rotation, permutation): (rotation, permutation)
        for rotation, permutation in zip(
            fractional_rotations, permutations, strict=True
        )
    }

    def resolve(rotation, permutation):
        key = action_key(rotation, permutation)
        return key if key in actions else None

    identity_key = resolve(np.eye(3, dtype=np.int64), np.arange(len(permutations[0])))

    if identity_key is None:
        return False

    generators = []
    generated = {identity_key}
    for candidate_key, candidate in actions.items():
        if candidate_key in generated:
            continue
        generators.append(candidate)

        # Re-enumerate canonically after adding each generator.
        generated = {identity_key}
        frontier = [identity_key]
        while frontier:
            current_key = frontier.pop()
            current_rotation, current_permutation = actions[current_key]
            for generator_rotation, generator_permutation in generators:
                composed_key = resolve(
                    generator_rotation @ current_rotation,
                    generator_permutation[current_permutation],
                )
                if composed_key is None:
                    return False
                if composed_key not in generated:
                    generated.add(composed_key)
                    frontier.append(composed_key)

    return len(generated) == len(actions)


def _get_quadrature(lebedev_order: int, n_rotations: int, include_inversion: bool):
    """Return the shared Lebedev-by-angle SO(3)/O(3) quadrature."""
    from metatomic.torch.symmetrized_model import get_rotation_quadrature

    return get_rotation_quadrature(lebedev_order, n_rotations, include_inversion)


def _build_periodic_site_indices(
    frac: np.ndarray,
    species_indices: List[np.ndarray],
    cell: np.ndarray,
    match_tolerance: float,
):
    """Build a conservative periodic candidate index, or request fallback.

    ``||d_frac|| <= ||d_cart|| / sigma_min(cell)`` makes the query complete;
    :func:`find_mic` remains authoritative in Cartesian space.
    """
    if len(species_indices) == 0 or not any(
        len(indices) >= 8 for indices in species_indices
    ):
        return None, 0.0

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None, 0.0

    try:
        singular_values = np.linalg.svd(cell, compute_uv=False)
    except np.linalg.LinAlgError:
        return None, 0.0
    smallest_singular_value = float(singular_values[-1])
    if not np.isfinite(smallest_singular_value) or smallest_singular_value <= 0.0:
        return None, 0.0

    # Cover boundary round-off between tree and Cartesian calculations.
    eps = np.finfo(float).eps
    fractional_query_radius = (
        match_tolerance / smallest_singular_value * (1.0 + 64.0 * eps) + 64.0 * eps
    )
    fractional_query_radius = float(np.nextafter(fractional_query_radius, np.inf))
    # Dense candidate sets are handled by the bounded vectorized fallback.
    if (
        not np.isfinite(fractional_query_radius)
        or fractional_query_radius <= 0.0
        or fractional_query_radius >= 0.25
    ):
        return None, 0.0

    wrapped_frac = frac % 1.0
    try:
        site_indices = [
            cKDTree(wrapped_frac[indices], boxsize=1.0) if len(indices) >= 8 else None
            for indices in species_indices
        ]
    except ValueError:
        return None, 0.0
    return site_indices, fractional_query_radius


def _match_species_pairwise(
    new_frac: np.ndarray,
    frac: np.ndarray,
    indices: np.ndarray,
    cell,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return nearest same-species sites using bounded dense chunks."""
    nearest = np.empty(len(indices), dtype=np.int64)
    nearest_distances = np.empty(len(indices), dtype=float)
    max_pairs_per_chunk = 65_536
    chunk_size = max(1, max_pairs_per_chunk // len(indices))
    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        source = indices[start:stop]
        fractional_delta = new_frac[source, None, :] - frac[None, indices, :]
        cartesian_delta = fractional_delta @ cell.array
        _, distances = find_mic(cartesian_delta.reshape(-1, 3), cell, pbc=True)
        distances = distances.reshape(len(source), len(indices))
        local_nearest = np.argmin(distances, axis=1)
        nearest[start:stop] = local_nearest
        nearest_distances[start:stop] = distances[np.arange(len(source)), local_nearest]
    return nearest, nearest_distances


def _match_species_with_periodic_index(
    new_frac: np.ndarray,
    frac: np.ndarray,
    indices: np.ndarray,
    cell,
    match_tolerance: float,
    site_index,
    fractional_query_radius: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Match indexed candidates, returning ``None`` for exact fallback."""
    query_points = new_frac[indices] % 1.0
    counts = np.asarray(
        site_index.query_ball_point(
            query_points,
            r=fractional_query_radius,
            return_length=True,
        ),
        dtype=np.int64,
    )
    candidate_count = int(np.sum(counts))
    # Keep verification linear for large species and bounded for small ones.
    candidate_limit = max(65_536, 4 * len(indices))
    if (
        np.any(counts == 0)
        or candidate_count * 4 >= len(indices) * len(indices)
        or candidate_count > candidate_limit
    ):
        return None

    # Materialize ragged lists only after the count-only sparse preflight.
    candidates = site_index.query_ball_point(
        query_points,
        r=fractional_query_radius,
        return_sorted=True,
    )

    candidate_indices = np.concatenate(candidates).astype(np.int64, copy=False)
    source_indices = np.repeat(np.arange(len(indices)), counts)
    fractional_delta = (
        new_frac[indices[source_indices]] - frac[indices[candidate_indices]]
    )
    cartesian_delta = fractional_delta @ cell.array
    _, candidate_distances = find_mic(cartesian_delta, cell, pbc=True)

    offsets = np.cumsum(counts) - counts
    accepted_counts = np.add.reduceat(
        (candidate_distances <= match_tolerance).astype(np.int64), offsets
    )
    # Exact fallback owns ambiguity and tie-breaking.
    if np.any(accepted_counts != 1):
        return None

    nearest_distances = np.minimum.reduceat(candidate_distances, offsets)

    # Match ``np.argmin``: lowest local index wins an exact tie.
    is_nearest = candidate_distances == np.repeat(nearest_distances, counts)
    sentinel = len(indices)
    nearest = np.minimum.reduceat(
        np.where(is_nearest, candidate_indices, sentinel), offsets
    )
    return nearest, nearest_distances


def _cartesian_site_match_tolerance(cell: np.ndarray, symprec: float) -> float:
    """Return the Cartesian site threshold including round-off allowance."""
    cell_scale = max(1.0, float(np.linalg.norm(cell, ord=2)))
    return symprec + 64.0 * np.finfo(float).eps * cell_scale


def _get_group_operations(
    atoms: ase.Atoms,
    symprec: float = 1e-6,
    angle_tolerance: float = -1.0,
    per_atom_features: Optional[List[_PerAtomFeature]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Return Cartesian rotations and site permutations for retained actions.

    For each fractional action ``(Rf, tf)``, ``permutation[i]`` is the original
    site matched by the image of site ``i``. Matching is restricted to the same
    species and uses minimum-image Cartesian distance with an effective
    ``symprec`` threshold including round-off. A periodic index can reduce the
    candidate set, but the bounded pairwise path remains the authoritative
    fallback. Every accepted permutation is within tolerance and bijective.

    Input-incompatible actions are discarded. Actions with the same rotation
    but translation-distinct permutations remain separate, and a filtered set
    is accepted only if it is closed under action composition.
    """
    if not atoms.pbc.all():
        return [], []
    if per_atom_features is None:
        per_atom_features = []

    try:
        import spglib
    except ImportError as e:
        raise ImportError(
            "spglib is required to use the SymmetrizedCalculator with "
            "`apply_space_group_symmetry=True`. Please install it with "
            "`pip install spglib` or `conda install -c conda-forge spglib`"
        ) from e

    A = atoms.cell.array.T
    frac = atoms.get_scaled_positions()
    numbers = atoms.numbers
    N = len(atoms)

    dataset_arguments = (atoms.cell.array, frac, numbers)
    with warnings.catch_warnings():
        # spglib 2.7 warns while its legacy global error mode is enabled. The
        # supported non-throwing API is intentional here: ``None`` is handled below.
        warnings.filterwarnings(
            "ignore",
            message="Set OLD_ERROR_HANDLING to false.*",
            category=DeprecationWarning,
        )
        data = spglib.get_symmetry_dataset(
            dataset_arguments,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )

    if data is None:
        return [], []
    if hasattr(data, "rotations"):
        R_frac = data.rotations
        t_frac = data.translations
    else:
        # spglib releases before the dataclass result used mapping access.
        R_frac = data["rotations"]
        t_frac = data["translations"]
    species_indices = [np.flatnonzero(numbers == z) for z in np.unique(numbers)]
    match_tolerance = _cartesian_site_match_tolerance(atoms.cell.array, symprec)
    site_indices, fractional_query_radius = _build_periodic_site_indices(
        frac,
        species_indices,
        atoms.cell.array,
        match_tolerance,
    )

    Q_list, permutations, fractional_rotations = [], [], []
    seen = set()
    Ainv = np.linalg.inv(A)

    for Rf, tf in zip(R_frac, t_frac, strict=True):
        if np.array_equal(Rf, np.eye(3, dtype=Rf.dtype)):
            Q = np.eye(3)
        else:
            # Solve Q A = A Rf rather than explicitly multiplying by A^{-1}.
            Q = np.linalg.solve(A.T, (A @ Rf).T).T
            # Recover entries that are indistinguishable from exact zero at the
            # construction-error scale. This prevents a huge unrelated vector
            # component from hiding a small physical mismatch.
            construction_scale = np.abs(A) @ np.abs(Rf) @ np.abs(Ainv)
            zero_tolerance = 64.0 * np.finfo(Q.dtype).eps * construction_scale
            if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(zero_tolerance)):
                raise ValueError(
                    "space-group Cartesian rotation is numerically undefined for "
                    "this cell"
                )
            Q[np.abs(Q) <= zero_tolerance] = 0.0
        # ``Rf`` is an integer unimodular matrix. Compute its determinant with
        # Python integers so axial-vector parity does not depend on a rounded
        # floating-point determinant.
        (a, b, c), (d, e, f), (g, h, i) = Rf.tolist()
        fractional_determinant = (
            a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        )
        permutation = np.empty(N, dtype=np.int64)
        new_frac = (frac @ Rf.T + tf) % 1.0
        for species, indices in enumerate(species_indices):
            matched = None
            if site_indices is not None and site_indices[species] is not None:
                matched = _match_species_with_periodic_index(
                    new_frac,
                    frac,
                    indices,
                    atoms.cell,
                    match_tolerance,
                    site_indices[species],
                    fractional_query_radius,
                )
            if matched is None:
                matched = _match_species_pairwise(
                    new_frac,
                    frac,
                    indices,
                    atoms.cell,
                )

            nearest, nearest_distances = matched
            if np.any(nearest_distances > match_tolerance):
                maximum = float(np.max(nearest_distances))
                raise RuntimeError(
                    "space-group operation can not be matched to atom sites: "
                    f"maximum Cartesian distance {maximum} exceeds effective "
                    f"matching threshold {match_tolerance} "
                    f"(symprec={symprec})"
                )
            permutation[indices] = indices[nearest]
        if len(np.unique(permutation)) != N:
            raise RuntimeError(
                "space-group operation did not produce a bijective atom permutation"
            )

        if not _arrays_respect_operation(
            per_atom_features,
            Q,
            permutation,
            determinant=fractional_determinant,
        ):
            continue

        # Translation-distinct permutations remain distinct actions.
        key = (tuple(Rf.flatten().tolist()), tuple(permutation.tolist()))
        if key in seen:
            continue
        seen.add(key)

        Q_list.append(Q.astype(float))
        permutations.append(permutation)
        fractional_rotations.append(Rf.astype(np.int64))

    if len(per_atom_features) != 0 and not _group_actions_are_closed(
        fractional_rotations, permutations
    ):
        raise ValueError(
            "model-input-compatible space-group operations are not closed; "
            "the per-atom inputs are numerically ambiguous under the requested "
            "symmetry tolerance"
        )

    return Q_list, permutations


def _average_over_group(
    results: dict, Q_list: List[np.ndarray], permutations: List[np.ndarray]
) -> dict:
    """Apply the space-group projector to supported output properties.

    With ``p = permutation`` and Cartesian row-vector convention, one action
    contributes ``energies[p]``, ``forces[p] @ Q``, and ``Q.T @ stress @ Q``.
    Total energy is already invariant under the discrete site action.
    """
    m = len(Q_list)
    if len(permutations) != m:
        raise ValueError(
            "Q_list and permutations must contain the same number of operations"
        )
    if m == 0:
        return results

    out = {}
    if "energy" in results:
        out["energy"] = float(results["energy"])

    if "energies" in results:
        energies = np.asarray(results["energies"], float)
        if energies.ndim != 1:
            raise ValueError(f"'energies' must be (N,), got {energies.shape}")
        acc = np.zeros_like(energies)
        for permutation in permutations:
            acc += energies[permutation]
        out["energies"] = acc / m

    if "forces" in results:
        F = np.asarray(results["forces"], float)
        if F.ndim != 2 or F.shape[1] != 3:
            raise ValueError(f"'forces' must be (N,3), got {F.shape}")
        acc = np.zeros_like(F)
        for Q, permutation in zip(Q_list, permutations, strict=True):
            acc += F[permutation] @ Q
        out["forces"] = acc / m

    if "stress" in results:
        S = np.asarray(results["stress"], float)
        if S.shape != (3, 3):
            raise ValueError(f"'stress' must be (3,3), got {S.shape}")
        acc = np.zeros_like(S)
        for Q in Q_list:
            acc += Q.T @ S @ Q
        out["stress"] = acc / m

    return out
