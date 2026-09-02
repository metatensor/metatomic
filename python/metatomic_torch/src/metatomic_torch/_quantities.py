"""Cartesian layout and spherical character for standard quantities."""

from typing import Dict


def standard_quantity_categories() -> Dict[str, str]:
    """Return the Cartesian layout and spherical character for standard quantities.

    This is the single source of truth for which outputs and inputs are
    decomposed; it mirrors ``KNOWN_QUANTITIES`` in
    ``metatomic-torch/src/quantities.cpp``, minus ``feature``. Only the current
    (singular) spellings appear here: normalize deprecated aliases with
    :py:func:`current_quantity_name` before looking them up.

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


def new_quantity_names() -> Dict[str, str]:
    """Return the mapping from deprecated quantity names to their current name.

    TorchScript cannot read a module-level dictionary from a compiled function,
    so the table is built by this function.
    """
    return {
        "features": "feature",
        "non_conservative_forces": "non_conservative_force",
        "positions": "position",
        "momenta": "momentum",
        "masses": "mass",
        "velocities": "velocity",
        "charges": "charge",
    }


def deprecated_quantity_names() -> Dict[str, str]:
    """Return the mapping from current quantity names to their deprecated name."""
    result: Dict[str, str] = {}
    for deprecated, new in new_quantity_names().items():
        result[new] = deprecated
    return result


def current_quantity_name(name: str) -> str:
    """Replace a deprecated base quantity in ``name`` with its current name."""
    base = name.split("/")[0]
    names = new_quantity_names()
    if base in names:
        return name.replace(base, names[base], 1)
    return name


def deprecated_quantity_name(name: str) -> str:
    """Replace a current base quantity in ``name`` with its deprecated name."""
    base = name.split("/")[0]
    names = deprecated_quantity_names()
    if base in names:
        return name.replace(base, names[base], 1)
    return name
