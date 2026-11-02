import ase
import numpy as np


def _get_per_atom_charges(atoms: ase.Atoms) -> np.ndarray:
    try:
        return atoms.get_charges()
    except Exception:
        return atoms.get_initial_charges()


PER_ATOM_QUANTITIES = {
    # standard metatomic quantities
    "momentum": {
        "properties_name": "momentum",
        "getter": ase.Atoms.get_momenta,
        "unit": "(eV*u)^(1/2)",
    },
    "mass": {
        "properties_name": "mass",
        "getter": ase.Atoms.get_masses,
        "unit": "u",
    },
    "velocity": {
        "properties_name": "velocity",
        "getter": ase.Atoms.get_velocities,
        "unit": "(eV/u)^(1/2)",
    },
    "charge": {
        "properties_name": "charge",
        "getter": _get_per_atom_charges,
        "unit": "e",
    },
    # ASE-specific quantities
    "ase::initial_magmoms": {
        "properties_name": "magnetic_moment",
        "getter": ase.Atoms.get_initial_magnetic_moments,
        "unit": "e * hbar / (2 * m_e)",
    },
    "ase::initial_charges": {
        "properties_name": "charge",
        "getter": ase.Atoms.get_initial_charges,
        "unit": "e",
    },
    "ase::tags": {
        "properties_name": "tag",
        "getter": ase.Atoms.get_tags,
        "unit": "",
    },
}


def _get_total_charge(atoms: ase.Atoms) -> np.ndarray:
    return np.array([float(atoms.info.get("charge", 0.0))])


def _get_spin_multiplicity(atoms: ase.Atoms) -> np.ndarray:
    # For spin multiplicity, we first check if the user provided it explicitly in
    # `atoms.info["spin_multiplicity"]`, and if not, we fall back to
    # `atoms.info["spin"]` (which is used by the OMOL dataset)
    if "spin_multiplicity" in atoms.info:
        return np.array([float(atoms.info["spin_multiplicity"])])

    return np.array([float(atoms.info.get("spin", 1))])


PER_SYSTEM_QUANTITIES = {
    # standard quantities
    "charge": {
        "properties_name": "charge",
        "getter": _get_total_charge,
        "unit": "e",
    },
    "spin_multiplicity": {
        "properties_name": "spin_multiplicity",
        "getter": _get_spin_multiplicity,
        "unit": "",
    },
}


_MISSING_INFO = object()

# For standard per-system quantities the model can request, the `atoms.info` keys that
# can be read by the getter and the value returned when the key is absent. Used by
# `check_state` to invalidate the cached results when the key changes.
_INFO_KEY_FOR_QUANTITY = {
    "charge": (["charge"], 0.0),
    "spin_multiplicity": (["spin_multiplicity", "spin"], 1),
}


def _full_3x3_to_voigt_6_stress(stress):
    """
    Re-implementation of ``ase.stress.full_3x3_to_voigt_6_stress`` which does not do the
    stress symmetrization correctly (they do ``(stress[1, 2] + stress[1, 2]) / 2.0``)
    """
    return np.array(
        [
            stress[0, 0],
            stress[1, 1],
            stress[2, 2],
            (stress[1, 2] + stress[2, 1]) / 2.0,
            (stress[0, 2] + stress[2, 0]) / 2.0,
            (stress[0, 1] + stress[1, 0]) / 2.0,
        ]
    )
