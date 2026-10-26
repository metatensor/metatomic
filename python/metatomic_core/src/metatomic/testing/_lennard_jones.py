from typing import NamedTuple, Optional

import metatensor as mts
import numpy as np
from metatensor import Labels, TensorBlock, TensorMap

from .._capabilities import ModelCapabilities
from .._metadata import ModelMetadata
from .._model import NumpyModel
from .._quantity import Quantity
from .._system import PairListOptions, System


class _EnergyGradients(NamedTuple):
    """Lennard-Jones data for all the pairs in a system"""

    first: np.ndarray
    second: np.ndarray
    vectors: np.ndarray
    # energy of each pair
    energies: np.ndarray
    # gradient of the energy of each pair with respect to the pair vector
    gradients: np.ndarray


class LennardJonesModel(NumpyModel):
    """
    Shifted Lennard-Jones potential implemented with NumPy, to be used when testing the
    integration of metatomic models with simulation engines.

    The energy of this model is a sum over all pairs within the cutoff of

    .. math::

        4 \\epsilon \\left(
            \\left( \\frac{\\sigma}{r} \\right)^{12}
            - \\left( \\frac{\\sigma}{r} \\right)^6
        \\right) - \\text{shift},

    where :math:`r` is the distance between the atoms in the pair, and
    :math:`\\text{shift}` is a constant ensuring that the energy goes to 0 at the
    cutoff. The energy of each pair is split equally between its two atoms, and when
    using ``selected_atoms``, only the share of the selected atoms is included in the
    per-system energy.

    This model can compute the following outputs:

    - ``energy``, either per-system (with explicit gradients with respect to positions
      and strain) or per-atom;
    - ``energy_ensemble``, with 16 members all equal to the energy;
    - ``energy_uncertainty``, set to :math:`0.001 \\times N_\\text{atoms}^2`. This
      grows artificially fast with the number of atoms in the system, which is useful
      to test the handling of high uncertainties;
    - ``non_conservative_force``, set to the Lennard-Jones forces plus some random
      noise;
    - ``non_conservative_stress``, filled with random numbers.

    Each output also has a ``doubled`` variant (e.g. ``energy/doubled``) which is two
    times the corresponding default output.

    When ``selected_atoms`` is ``None``, the samples of per-atom outputs are randomly
    shuffled, to check that engines handle samples in any order.

    Engines can load this model through the Python plugin (with ``"python"`` as the
    plugin name), using a script like this one:

    .. code-block:: python

        from metatomic.testing import LennardJonesModel

        model = LennardJonesModel(
            atomic_type=18,
            cutoff=7.0,
            epsilon=0.0104,
            sigma=3.4,
            length_unit="angstrom",
            energy_unit="eV",
        )
    """

    def __init__(
        self,
        *,
        atomic_type: int,
        cutoff: float,
        epsilon: float,
        sigma: float,
        length_unit: str,
        energy_unit: str,
        seed: Optional[int] = None,
    ):
        """
        :param atomic_type: atomic type to which ``sigma`` and ``epsilon`` correspond
        :param cutoff: spherical cutoff of the potential, in ``length_unit``
        :param epsilon: depth of the potential well, in ``energy_unit``
        :param sigma: distance at which the (unshifted) potential is zero, in
            ``length_unit``
        :param length_unit: unit of length used by the model
        :param energy_unit: unit of energy used by the model
        :param seed: seed for the random numbers used in the outputs. If this is
            ``None``, a different seed is used for each call to the model; otherwise the
            same seed is used for all calls, making the outputs deterministic.
        """
        super().__init__()

        self._atomic_type = atomic_type
        self._cutoff = cutoff
        self._epsilon = epsilon
        self._sigma = sigma
        self._length_unit = length_unit
        self._energy_unit = energy_unit
        self._seed = seed
        # number of members for the energy ensemble
        self._n_ensemble_members = 16

        self._pairs_options = PairListOptions(
            cutoff=self._cutoff, full_list=False, strict=True
        )

        # shift the energy to 0 at the cutoff
        self._shift = 4 * epsilon * ((sigma / cutoff) ** 12 - (sigma / cutoff) ** 6)

    def capabilities(self) -> ModelCapabilities:
        energy_unit = self._energy_unit
        force_unit = f"{self._energy_unit}/{self._length_unit}"
        stress_unit = f"{self._energy_unit}/{self._length_unit}^3"

        # name, unit, sample kinds and description of all the outputs
        definitions = [
            ("energy", energy_unit, ["system", "atom"], "Lennard-Jones energy"),
            (
                "energy_ensemble",
                energy_unit,
                ["system", "atom"],
                "ensemble of energies, all equal to the Lennard-Jones energy",
            ),
            (
                "energy_uncertainty",
                energy_unit,
                ["system", "atom"],
                "pseudo-uncertainty for tests, scaling with the system size",
            ),
            (
                "non_conservative_force",
                force_unit,
                ["atom"],
                "Lennard-Jones forces, with added random noise",
            ),
            ("non_conservative_stress", stress_unit, ["system"], "random stress"),
        ]

        outputs = []
        for variant, suffix in [("", ""), ("/doubled", ", multiplied by 2")]:
            for name, unit, sample_kinds, description in definitions:
                for sample_kind in sample_kinds:
                    # explicit gradients are available for all per-system energies
                    if sample_kind == "system" and name.startswith("energy"):
                        gradients = ["positions", "strain"]
                    else:
                        gradients = []

                    outputs.append(
                        Quantity(
                            name=name + variant,
                            unit=unit,
                            sample_kind=sample_kind,
                            gradients=gradients,
                            description=description + suffix,
                        )
                    )

        return ModelCapabilities(
            atomic_types=[self._atomic_type],
            interaction_range=self._cutoff,
            length_unit=self._length_unit,
            supported_devices=["cpu"],
            dtype="float64",
            outputs=outputs,
        )

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="Test Lennard-Jones (NumPy)",
            description=(
                "Minimal shifted Lennard-Jones potential, to be used when testing the "
                "integration of metatomic models with various simulation engines."
            ),
            authors=["metatomic authors"],
            references={"implementation": ["https://github.com/metatensor/metatomic"]},
        )

    def requested_pair_lists(
        self,
        requested_outputs: list[Quantity],
    ) -> list[PairListOptions]:
        outputs_using_pairs = ("energy", "energy_ensemble", "non_conservative_force")

        for output in requested_outputs:
            if output.name.split("/")[0] in outputs_using_pairs:
                return [self._pairs_options]

        return []

    def execute_inner(
        self,
        systems: list[System],
        selected_atoms: Optional[Labels],
        requested_outputs: list[Quantity],
    ) -> list[TensorMap]:
        capabilities = self.capabilities()

        # Each output uses a new random number generator created from the same seed,
        # making the `doubled` variants exactly two times the default outputs, even for
        # the random parts.
        seed = np.random.SeedSequence(self._seed)

        results = []
        for request in requested_outputs:
            declared = capabilities.find_output(request)
            if declared is None:
                raise ValueError(
                    f"this model can not compute '{request.name}' with sample kind "
                    f"'{request.sample_kind}'"
                )

            for gradient in request.gradients:
                if gradient not in declared.gradients:
                    raise ValueError(
                        f"this model can not compute explicit gradients of "
                        f"'{request.name}' with respect to '{gradient}'"
                    )

            name, _, variant = request.name.partition("/")
            rng = np.random.default_rng(seed)

            if request.sample_kind == "atom":
                block = self._per_atom_output(name, systems, selected_atoms, rng)
            elif name == "non_conservative_stress":
                system_indexes = _system_samples(systems, selected_atoms)
                block = TensorBlock(
                    values=rng.standard_normal((len(system_indexes), 3, 3, 1)),
                    samples=Labels(["system"], system_indexes.reshape(-1, 1)),
                    components=[Labels.range("xyz_1", 3), Labels.range("xyz_2", 3)],
                    properties=Labels.range("non_conservative_stress", 1),
                )
            else:
                block = self._per_system_energy(
                    name, systems, selected_atoms, request.gradients
                )

            result = TensorMap(Labels.range("_", 1), [block])
            if variant == "doubled":
                result = mts.multiply(result, 2.0)

            results.append(result)

        return results

    def _energy_and_gradients(self, system: System) -> _EnergyGradients:
        """Compute the Lennard-Jones energy and gradients of all pairs in ``system``"""
        pairs = system.pairs(self._pairs_options)
        vectors = pairs.values.reshape(-1, 3)

        distances_2 = np.sum(vectors**2, axis=1)
        sigma_r_6 = (self._sigma**2 / distances_2) ** 3
        sigma_r_12 = sigma_r_6 * sigma_r_6
        energies = 4.0 * self._epsilon * (sigma_r_12 - sigma_r_6) - self._shift

        derivative = 24.0 * self._epsilon * (sigma_r_6 - 2.0 * sigma_r_12) / distances_2
        gradients = derivative[:, None] * vectors

        return _EnergyGradients(
            first=pairs.samples.column("first_atom"),
            second=pairs.samples.column("second_atom"),
            vectors=vectors,
            energies=energies,
            gradients=gradients,
        )

    def _per_atom_output(
        self,
        name: str,
        systems: list[System],
        selected_atoms: Optional[Labels],
        rng: np.random.Generator,
    ) -> TensorBlock:
        """
        Compute the per-atom output ``name`` for the atoms in ``selected_atoms``, or for
        all atoms if ``selected_atoms`` is ``None``.
        """
        samples = _atom_samples(systems, selected_atoms, rng)

        if name == "non_conservative_force":
            values = np.zeros((len(samples), 3, 1))
            components = [Labels.range("xyz", 3)]
            properties = Labels.range("non_conservative_force", 1)
        else:
            n_properties = self._n_ensemble_members if name == "energy_ensemble" else 1
            values = np.zeros((len(samples), n_properties))
            components = []
            properties = Labels.range("energy", n_properties)

        for system_i, system in enumerate(systems):
            n_atoms = len(system)

            if name == "energy_uncertainty":
                system_values = np.full((n_atoms, 1), 0.001 * n_atoms**2)
            elif name == "non_conservative_force":
                pairs = self._energy_and_gradients(system)
                forces = np.zeros((n_atoms, 3))
                np.add.at(forces, pairs.first, pairs.gradients)
                np.add.at(forces, pairs.second, -pairs.gradients)

                if n_atoms != 0:
                    # add some noise to make these forces non-conservative
                    noise = rng.standard_normal((n_atoms, 3))
                    forces += 0.3 * noise * np.mean(np.abs(forces))

                system_values = forces.reshape(n_atoms, 3, 1)
            else:
                pairs = self._energy_and_gradients(system)
                # each pair is only included once (half pair list), and we assign half
                # of the energy to each atom
                energies = np.zeros(n_atoms)
                np.add.at(energies, pairs.first, 0.5 * pairs.energies)
                np.add.at(energies, pairs.second, 0.5 * pairs.energies)

                # all the ensemble members are equal to the energy
                system_values = energies.reshape(n_atoms, 1)

            mask = samples[:, 0] == system_i
            values[mask] = system_values[samples[mask, 1]]

        return TensorBlock(
            values=values,
            samples=Labels(["system", "atom"], samples),
            components=components,
            properties=properties,
        )

    def _per_system_energy(
        self,
        name: str,
        systems: list[System],
        selected_atoms: Optional[Labels],
        gradients: list[str],
    ) -> TensorBlock:
        """
        Compute the per-system output ``name`` (one of ``energy``, ``energy_ensemble``
        or ``energy_uncertainty``) and its ``gradients``.
        """
        system_indexes = _system_samples(systems, selected_atoms)
        n_samples = len(system_indexes)
        n_gradient_samples = sum(len(systems[i]) for i in system_indexes)

        energies = np.zeros(n_samples)
        positions_samples = np.zeros((n_gradient_samples, 3), dtype=np.int32)
        positions_gradients = np.zeros((n_gradient_samples, 3))
        strain_gradients = np.zeros((n_samples, 3, 3))

        start = 0
        for sample_i, system_i in enumerate(system_indexes):
            system = systems[system_i]
            n_atoms = len(system)

            stop = start + n_atoms
            positions_samples[start:stop, 0] = sample_i
            positions_samples[start:stop, 1] = system_i
            positions_samples[start:stop, 2] = np.arange(n_atoms)

            if name == "energy_uncertainty":
                # this does not depend on positions or strain, the gradients are zero
                energies[sample_i] = 0.001 * n_atoms**2
            else:
                pairs = self._energy_and_gradients(system)

                if selected_atoms is None:
                    weights = np.ones(len(pairs.energies))
                else:
                    selected = np.zeros(n_atoms, dtype=bool)
                    current_system = selected_atoms.column("system") == system_i
                    selected[selected_atoms.column("atom")[current_system]] = True

                    # only include the part of the pair energy assigned to the
                    # selected atoms
                    weights = 0.5 * selected[pairs.first] + 0.5 * selected[pairs.second]

                energies[sample_i] = np.sum(weights * pairs.energies)

                pairs_gradients = weights[:, None] * pairs.gradients
                system_gradients = positions_gradients[start:stop]
                np.add.at(system_gradients, pairs.first, -pairs_gradients)
                np.add.at(system_gradients, pairs.second, pairs_gradients)

                strain_gradients[sample_i] = pairs.vectors.T @ pairs_gradients

            start = stop

        n_properties = self._n_ensemble_members if name == "energy_ensemble" else 1
        properties = Labels.range("energy", n_properties)

        block = TensorBlock(
            values=np.repeat(energies[..., None], n_properties, axis=-1),
            samples=Labels(["system"], system_indexes.reshape(-1, 1)),
            components=[],
            properties=properties,
        )

        if "positions" in gradients:
            block.add_gradient(
                "positions",
                TensorBlock(
                    values=np.repeat(
                        positions_gradients[..., None], n_properties, axis=-1
                    ),
                    samples=Labels(["sample", "system", "atom"], positions_samples),
                    components=[Labels.range("xyz", 3)],
                    properties=properties,
                ),
            )

        if "strain" in gradients:
            block.add_gradient(
                "strain",
                TensorBlock(
                    values=np.repeat(
                        strain_gradients[..., None], n_properties, axis=-1
                    ),
                    samples=Labels.range("sample", n_samples),
                    components=[Labels.range("xyz_1", 3), Labels.range("xyz_2", 3)],
                    properties=properties,
                ),
            )

        return block


def _system_samples(
    systems: list[System],
    selected_atoms: Optional[Labels],
) -> np.ndarray:
    """Get the indexes of the systems to include in per-system outputs"""
    if selected_atoms is None:
        return np.arange(len(systems), dtype=np.int32)
    else:
        return np.unique(np.asarray(selected_atoms.column("system"), dtype=np.int32))


def _atom_samples(
    systems: list[System],
    selected_atoms: Optional[Labels],
    rng: np.random.Generator,
) -> np.ndarray:
    """Get the ``(system, atom)`` samples to include in per-atom outputs"""
    if selected_atoms is not None:
        return np.asarray(selected_atoms.values, dtype=np.int32)

    samples = np.array(
        [
            [system_i, atom_i]
            for system_i, system in enumerate(systems)
            for atom_i in range(len(system))
        ],
        dtype=np.int32,
    ).reshape(-1, 2)

    # randomly shuffle the samples to make sure engines handle out of order samples
    return samples[rng.permutation(len(samples))]
