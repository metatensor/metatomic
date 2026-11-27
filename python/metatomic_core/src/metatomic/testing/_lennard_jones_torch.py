from typing import Optional

import torch
from metatensor import Labels, TensorBlock, TensorMap

from .._capabilities import ModelCapabilities
from .._metadata import ModelMetadata
from .._model._torch import TorchModel
from .._quantity import Quantity
from .._system import PairListOptions, System


_TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}


class TorchLennardJonesModel(TorchModel):
    """
    Shifted Lennard-Jones potential implemented with PyTorch, to be used when testing
    the integration of metatomic models with simulation engines.

    This is the :py:class:`metatomic.TorchModel` counterpart of
    :py:class:`metatomic.testing.LennardJonesModel`. It computes the same outputs and
    ``/doubled`` variants, runs on CPU and GPU devices, and relies on PyTorch autograd
    (through :py:class:`metatomic.TorchModel`) for position and strain gradients.

    Engines can load this model through the Python plugin (with ``"python"`` as the
    plugin name), using a script like this one:

    .. code-block:: python

        from metatomic.testing import TorchLennardJonesModel

        model = TorchLennardJonesModel(
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
        dtype: str = "float64",
        supported_devices: Optional[list[str]] = None,
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
        :param dtype: data type declared by the model (``"float32"`` or ``"float64"``)
        :param supported_devices: devices declared by the model. Defaults to
            ``["cpu", "cuda", "rocm", "metal"]`` for ``"float32"`` and
            ``["cpu", "cuda", "rocm"]`` for ``"float64"``.
        :param seed: seed for the random numbers used in the outputs. If this is
            ``None``, a different seed is used for each call to the model; otherwise the
            same seed is used for all calls, making the outputs deterministic.
        """
        super().__init__()
        if dtype not in _TORCH_DTYPES:
            raise ValueError(
                f"dtype must be one of {list(_TORCH_DTYPES)}, got '{dtype}'"
            )

        self._atomic_type = atomic_type
        self._cutoff = cutoff
        self._epsilon = epsilon
        self._sigma = sigma
        self._length_unit = length_unit
        self._energy_unit = energy_unit
        self._dtype = dtype
        if supported_devices is None:
            self._supported_devices = (
                ["cpu", "cuda", "rocm", "metal"]
                if dtype == "float32"
                else ["cpu", "cuda", "rocm"]
            )
        else:
            self._supported_devices = list(supported_devices)
        self._seed = seed
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
                    # TorchModel adds autograd positions and strain gradients for
                    # per-system energy, energy_ensemble and energy_uncertainty
                    outputs.append(
                        Quantity(
                            name=name + variant,
                            unit=unit,
                            sample_kind=sample_kind,
                            gradients=[],
                            description=description + suffix,
                        )
                    )

        return ModelCapabilities(
            atomic_types=[self._atomic_type],
            interaction_range=self._cutoff,
            length_unit=self._length_unit,
            supported_devices=self._supported_devices,
            dtype=self._dtype,
            outputs=outputs,
        )

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="Test Lennard-Jones (PyTorch)",
            description=(
                "Minimal shifted Lennard-Jones potential implemented with PyTorch, "
                "to be used when testing the integration of metatomic models with "
                "various simulation engines."
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

    def forward(
        self,
        systems: list[System],
        selected_atoms: Optional[Labels],
        outputs: list[Quantity],
    ) -> list[TensorMap]:
        if systems:
            device = systems[0].positions.device
            dtype = systems[0].positions.dtype
        else:
            device = torch.device("cpu")
            dtype = _TORCH_DTYPES[self._dtype]

        # Each output uses a fresh generator seeded from the same base seed so that
        # `/doubled` variants are exactly twice the default outputs, including random
        # noise and sample shuffling.
        if self._seed is None:
            base_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        else:
            base_seed = int(self._seed)

        results = []
        for request in outputs:
            name, _, variant = request.name.partition("/")
            scale = 2.0 if variant == "doubled" else 1.0
            gen = torch.Generator().manual_seed(base_seed)

            if request.sample_kind == "atom":
                block = self._per_atom_output(
                    name, systems, selected_atoms, gen, device, dtype, scale
                )
            elif name == "non_conservative_stress":
                system_indexes = _system_samples(systems, selected_atoms, device)
                values = (
                    torch.randn((len(system_indexes), 3, 3, 1), generator=gen).to(
                        device=device, dtype=dtype
                    )
                    * scale
                )
                block = TensorBlock(
                    values=values,
                    samples=Labels(["system"], system_indexes.reshape(-1, 1)),
                    components=[
                        _range_labels("xyz_1", 3, device),
                        _range_labels("xyz_2", 3, device),
                    ],
                    properties=_range_labels("non_conservative_stress", 1, device),
                )
            else:
                block = self._per_system_energy(
                    name, systems, selected_atoms, device, dtype, scale
                )

            results.append(TensorMap(_range_labels("_", 1, device), [block]))

        return results

    def _pair_energies(
        self, system: System, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pair indices, vectors and shifted Lennard-Jones energies."""
        pairs = system.pairs(self._pairs_options)
        vectors = pairs.values.reshape(-1, 3)
        first = _as_tensor(pairs.samples.column("first_atom"), torch.int64, device)
        second = _as_tensor(pairs.samples.column("second_atom"), torch.int64, device)

        distances_2 = (vectors**2).sum(dim=1)
        sigma_r_6 = (self._sigma**2 / distances_2) ** 3
        sigma_r_12 = sigma_r_6 * sigma_r_6
        energies = 4.0 * self._epsilon * (sigma_r_12 - sigma_r_6) - self._shift
        return first, second, vectors, energies.to(dtype=dtype)

    def _per_atom_output(
        self,
        name: str,
        systems: list[System],
        selected_atoms: Optional[Labels],
        gen: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
        scale: float,
    ) -> TensorBlock:
        samples = _atom_samples(systems, selected_atoms, gen, device)

        if name == "non_conservative_force":
            values = torch.zeros((len(samples), 3, 1), dtype=dtype, device=device)
            components = [_range_labels("xyz", 3, device)]
            properties = _range_labels("non_conservative_force", 1, device)
        else:
            n_properties = self._n_ensemble_members if name == "energy_ensemble" else 1
            values = torch.zeros(
                (len(samples), n_properties), dtype=dtype, device=device
            )
            components = []
            properties = _range_labels("energy", n_properties, device)

        for system_i, system in enumerate(systems):
            n_atoms = len(system)

            if name == "energy_uncertainty":
                system_values = torch.full(
                    (n_atoms, 1), 0.001 * n_atoms**2, dtype=dtype, device=device
                )
            elif name == "non_conservative_force":
                with torch.no_grad():
                    pairs = system.pairs(self._pairs_options)
                    vectors = pairs.values.reshape(-1, 3)
                    first = _as_tensor(
                        pairs.samples.column("first_atom"), torch.int64, device
                    )
                    second = _as_tensor(
                        pairs.samples.column("second_atom"), torch.int64, device
                    )
                    distances_2 = (vectors**2).sum(dim=1)
                    sigma_r_6 = (self._sigma**2 / distances_2) ** 3
                    sigma_r_12 = sigma_r_6 * sigma_r_6
                    derivative = (
                        24.0
                        * self._epsilon
                        * (sigma_r_6 - 2.0 * sigma_r_12)
                        / distances_2
                    )
                    pair_grad = derivative[:, None] * vectors
                    forces = torch.zeros((n_atoms, 3), dtype=dtype, device=device)
                    forces = forces.index_add(0, first, pair_grad)
                    forces = forces.index_add(0, second, -pair_grad)
                    if n_atoms != 0:
                        noise = torch.randn((n_atoms, 3), generator=gen).to(
                            device=device, dtype=dtype
                        )
                        forces = forces + 0.3 * noise * forces.abs().mean()
                    system_values = forces.reshape(n_atoms, 3, 1)
            else:
                first, second, _, pair_energies = self._pair_energies(
                    system, device, dtype
                )
                energies = torch.zeros(n_atoms, dtype=dtype, device=device)
                energies = energies.index_add(0, first, 0.5 * pair_energies)
                energies = energies.index_add(0, second, 0.5 * pair_energies)
                system_values = energies.reshape(n_atoms, 1)

            mask = samples[:, 0] == system_i
            atom_indices = samples[mask, 1].to(dtype=torch.int64)
            values[mask] = system_values[atom_indices]

        return TensorBlock(
            values=values * scale,
            samples=Labels(["system", "atom"], samples),
            components=components,
            properties=properties,
        )

    def _per_system_energy(
        self,
        name: str,
        systems: list[System],
        selected_atoms: Optional[Labels],
        device: torch.device,
        dtype: torch.dtype,
        scale: float,
    ) -> TensorBlock:
        system_indexes = _system_samples(systems, selected_atoms, device)
        selected_values = (
            _as_tensor(selected_atoms.values, torch.int64, device)
            if selected_atoms is not None
            else None
        )

        energies = []
        for system_i in system_indexes.tolist():
            system = systems[system_i]
            n_atoms = len(system)

            if name == "energy_uncertainty":
                energies.append(
                    torch.tensor(0.001 * n_atoms**2, dtype=dtype, device=device)
                )
            else:
                first, second, _, pair_energies = self._pair_energies(
                    system, device, dtype
                )
                if selected_values is None:
                    energies.append(pair_energies.sum())
                else:
                    selected = torch.zeros(n_atoms, dtype=dtype, device=device)
                    current = selected_values[:, 0] == system_i
                    selected[selected_values[current, 1]] = 1.0
                    weights = 0.5 * selected[first] + 0.5 * selected[second]
                    energies.append((weights * pair_energies).sum())

        if energies:
            stacked = torch.stack(energies) * scale
        else:
            stacked = torch.zeros((0,), dtype=dtype, device=device)

        n_properties = self._n_ensemble_members if name == "energy_ensemble" else 1
        values = stacked.reshape(-1, 1).repeat(1, n_properties)
        return TensorBlock(
            values=values,
            samples=Labels(["system"], system_indexes.reshape(-1, 1)),
            components=[],
            properties=_range_labels("energy", n_properties, device),
        )


def _as_tensor(array, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(dtype=dtype, device=device)
    return torch.tensor(array, dtype=dtype, device=device)


def _range_labels(name: str, end: int, device: torch.device) -> Labels:
    return Labels(
        [name], torch.arange(end, dtype=torch.int32, device=device).reshape(-1, 1)
    )


def _system_samples(
    systems: list[System],
    selected_atoms: Optional[Labels],
    device: torch.device,
) -> torch.Tensor:
    """Get the indexes of the systems to include in per-system outputs."""
    if selected_atoms is None:
        return torch.arange(len(systems), dtype=torch.int32, device=device)
    column = _as_tensor(selected_atoms.column("system"), torch.int32, device)
    return torch.unique(column, sorted=True)


def _atom_samples(
    systems: list[System],
    selected_atoms: Optional[Labels],
    gen: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Get the ``(system, atom)`` samples to include in per-atom outputs."""
    if selected_atoms is not None:
        return _as_tensor(selected_atoms.values, torch.int32, device).reshape(-1, 2)

    rows = [
        [system_i, atom_i]
        for system_i, system in enumerate(systems)
        for atom_i in range(len(system))
    ]
    samples = torch.tensor(rows, dtype=torch.int32).reshape(-1, 2)
    perm = torch.randperm(len(samples), generator=gen)
    return samples[perm].to(device=device)
