"""Public calculator dispatch, keeping legacy Torch interfaces intact."""

from ase.calculators.calculator import Calculator

from metatomic import BaseModel


class MetatomicCalculator(Calculator):
    """Run Core or legacy Torch models through ASE.

    Core :class:`metatomic.BaseModel` instances use :func:`metatomic.execute_model`.
    Pass ``plugin_name`` or ``model_options`` to load a Core model from a source
    string. Other paths and legacy ``AtomisticModel`` instances retain their
    existing Torch behavior, including Torch TensorMap return values.

    For Core models, ``run_model`` accepts a sequence of :class:`metatomic.Quantity`
    and returns a corresponding list of Core TensorMaps. ``additional_outputs``
    maps user-chosen aliases to quantities. ``input_provider(atoms, quantity)`` can
    supply custom Core TensorMaps; returning ``None`` uses the built-in ASE inputs.
    Core systems and results use ASE units (angstrom and eV).

    Both interfaces accept ``device``, ``variants``, ``non_conservative``,
    ``do_gradients_with_energy``, ``uncertainty_threshold``, ``neighbor_skin``, and
    ``check_consistency``. Legacy paths also accept ``extensions_directory``.
    """

    _implementations = {}

    def __new__(cls, model, **kwargs):
        if not getattr(cls, "_is_backend", False):
            if (
                isinstance(model, BaseModel)
                or kwargs.get("plugin_name") is not None
                or kwargs.get("model_options") is not None
            ):
                from ._core_calculator import _CoreCalculator

                backend = _CoreCalculator
            else:
                from ._calculator import MetatomicCalculator as _LegacyCalculator

                backend = _LegacyCalculator
            if cls is MetatomicCalculator:
                cls = backend
            else:
                # Preserve user subclasses and their super() calls through the
                # selected implementation, without importing Torch for Core.
                key = (cls, backend)
                if key not in cls._implementations:
                    cls._implementations[key] = type(cls.__name__, (cls, backend), {})
                cls = cls._implementations[key]
        return object.__new__(cls)

    @classmethod
    def fromdict(cls, data):
        return MetatomicCalculator(**data)

    def _get_name(self):
        return "metatomiccalculator"

    def model(self):
        """Get the underlying Core or legacy Torch model."""
        raise NotImplementedError

    def metadata(self):
        """Get metadata from the underlying model."""
        raise NotImplementedError

    def run_model(self, atoms, outputs, selected_atoms=None):
        """Compute arbitrary outputs for one ASE Atoms object or a batch.

        With Core models, ``outputs`` is a sequence of Quantity objects (returning
        a list of Core TensorMaps), or an alias-to-Quantity mapping (returning a
        mapping). ``selected_atoms`` is Core Labels with system and atom columns.

        With legacy models, retain the dictionary of ModelOutput objects, Torch
        Labels selection, and dictionary of Torch TensorMaps.
        """
        raise NotImplementedError

    def compute_energy(
        self, atoms, compute_forces_and_stresses=False, *, per_atom=False
    ):
        """Return energy in eV and optionally forces and stresses as NumPy arrays.

        Forces use eV/angstrom; stresses are 3x3 arrays in eV/angstrom^3 and are
        included for batches only when all systems are fully periodic. For a
        batch, each dictionary entry contains one result per system. ``per_atom``
        additionally includes atomic energies under ``energies``.
        """
        raise NotImplementedError
