"""Cartesian layout and spherical character for standard quantities."""

from typing import Dict


def standard_quantity_categories() -> Dict[str, str]:
    """Return the Cartesian layout and spherical character for standard quantities.

    This is the single source of truth for which outputs and inputs are
    decomposed; it mirrors ``KNOWN_QUANTITIES`` in
    ``metatomic-torch/src/quantities.cpp``, minus ``feature``. Only the current
    (singular) spellings appear here: deprecated names are not recognized, and
    the code using this table treats them as custom quantities.

    TorchScript cannot read a module-level dictionary from a compiled function,
    so the table is built by this function and bound to
    :py:data:`STANDARD_QUANTITY_CATEGORIES` for Python callers.
    """
    return {
        # scalars: l = 0
        "charge": "scalar",
        "energy": "scalar",
        "energy_ensemble": "scalar",
        "energy_uncertainty": "scalar",
        "mass": "scalar",
        "spin_multiplicity": "scalar",
        # Cartesian vectors: l = 1
        "heat_flux": "cartesian_vector",
        "momentum": "cartesian_vector",
        "non_conservative_force": "cartesian_vector",
        "position": "cartesian_vector",
        "velocity": "cartesian_vector",
        # symmetric 3x3 matrices: l = 0 and l = 2
        "non_conservative_stress": "symmetric_matrix",
    }


STANDARD_QUANTITY_CATEGORIES: Dict[str, str] = standard_quantity_categories()

#: maximum angular momentum carried by each category above
MAX_ANGULAR_MOMENTUM_PER_CATEGORY: Dict[str, int] = {
    "scalar": 0,
    "cartesian_vector": 1,
    "symmetric_matrix": 2,
}


#: mapping from deprecated quantity names to their current name
NEW_QUANTITY_NAMES: Dict[str, str] = {
    "features": "feature",
    "non_conservative_forces": "non_conservative_force",
    "positions": "position",
    "momenta": "momentum",
    "masses": "mass",
    "velocities": "velocity",
    "charges": "charge",
}

#: mapping from current quantity names to the corresponding deprecated name
DEPRECATED_QUANTITY_NAMES: Dict[str, str] = {
    new: deprecated for deprecated, new in NEW_QUANTITY_NAMES.items()
}
