"""
Python-side mirror of ``metatomic-torch/src/quantities.cpp`` (``KNOWN_QUANTITIES``
and the per-quantity checks), holding the metadata for standard quantities: their
category (Cartesian layout and spherical character) and their deprecated-name
aliases.

This module must not import anything from ``metatomic``, so that any other module
can import it without creating an import cycle.
"""

from typing import Dict


def standard_quantity_categories() -> Dict[str, str]:
    """Return the Cartesian layout of every decomposable standard quantity.

    This is the single source of truth for which outputs and inputs are
    decomposed; it mirrors ``KNOWN_QUANTITIES`` in
    ``metatomic-torch/src/quantities.cpp``, minus ``feature``. Only the current
    (singular) spellings appear here: deprecated names are normalized before
    they reach the code using this table.

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
MAX_O3_LAMBDA_PER_CATEGORY: Dict[str, int] = {
    "scalar": 0,
    "cartesian_vector": 1,
    "symmetric_matrix": 2,
}


def _new_quantity_names() -> Dict[str, str]:
    """Return the map from deprecated quantity names to their current name.

    TorchScript cannot read a module-level dictionary from a compiled function,
    so the table is built by this function and bound to
    :py:data:`NEW_QUANTITY_NAMES` for Python callers.
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


NEW_QUANTITY_NAMES: Dict[str, str] = _new_quantity_names()

#: mapping from current quantity names to the corresponding deprecated name
DEPRECATED_QUANTITY_NAMES: Dict[str, str] = {
    new: deprecated for deprecated, new in NEW_QUANTITY_NAMES.items()
}
