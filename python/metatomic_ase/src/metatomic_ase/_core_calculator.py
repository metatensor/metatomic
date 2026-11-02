"""ASE boundary for backend-independent Core models."""

import ctypes
import os
import warnings

import ase
import numpy as np
import vesin
from ase.calculators.calculator import PropertyNotImplementedError
from metatensor import Labels, TensorBlock, TensorMap

from metatomic import BaseModel, Quantity, System, execute_model, load_model

from ._inputs import (
    PER_ATOM_QUANTITIES,
    PER_SYSTEM_QUANTITIES,
    _full_3x3_to_voigt_6_stress,
)
from ._interface import MetatomicCalculator


def _length_factor(from_unit):
    # Use the existing C API until the public Python helper (#370) is available
    # in the minimum supported Core version.
    from metatomic._c_lib import _get_library

    factor = ctypes.c_double()
    _get_library().mta_unit_conversion_factor(
        from_unit.encode(), b"angstrom", ctypes.byref(factor)
    )
    return factor.value


def _numpy(tensor):
    # Convert blocks separately: older metatensor releases return None for an
    # empty info value, which TensorMap.to() cannot copy.
    result = TensorMap(
        tensor.keys.to(device="cpu").to(arrays="numpy"),
        [block.to(device="cpu").to(arrays="numpy") for block in tensor.blocks()],
    )
    for key, value in tensor.info().items():
        result.set_info(key, value or "")
    return result


class _CoreCalculator(MetatomicCalculator):
    _is_backend = True

    def __init__(
        self,
        model,
        *,
        additional_outputs=None,
        plugin_name=None,
        model_options=None,
        input_provider=None,
        check_consistency=False,
        device=None,
        variants=None,
        non_conservative=False,
        do_gradients_with_energy=True,
        uncertainty_threshold=0.1,
        neighbor_skin=2.0,
        extensions_directory=None,
    ):
        super().__init__()
        if extensions_directory is not None:
            raise ValueError("extensions_directory is only supported for legacy models")
        if non_conservative not in (True, False, "forces", "stress"):
            raise ValueError(
                "non_conservative must be False, True, 'forces', or 'stress'"
            )
        if neighbor_skin < 0:
            raise ValueError("neighbor_skin must be non-negative")
        if uncertainty_threshold is not None and uncertainty_threshold <= 0:
            raise ValueError("uncertainty_threshold must be positive or None")
        self.parameters = dict(
            plugin_name=plugin_name,
            model_options=model_options,
            check_consistency=check_consistency,
            device=str(device) if device is not None else None,
            variants=variants,
            non_conservative=non_conservative,
            do_gradients_with_energy=do_gradients_with_energy,
            uncertainty_threshold=uncertainty_threshold,
            neighbor_skin=neighbor_skin,
        )
        if not isinstance(model, BaseModel):
            source = os.fspath(model)
            model = load_model(source, options=model_options, plugin_name=plugin_name)
            self.parameters["model"] = source
        self._model = model
        self._capabilities = model.capabilities()
        self._dtype = np.dtype(self._capabilities.dtype)
        self._device = self._select_device(device)
        self._input_provider = input_provider
        self._nc_forces = non_conservative in (True, "forces")
        self._nc_stress = non_conservative in (True, "stress")
        self._variants = variants or {}
        self._additional_output_requests = {
            alias: Quantity.from_dict(q) if isinstance(q, dict) else q
            for alias, q in (additional_outputs or {}).items()
        }
        for alias, q in self._additional_output_requests.items():
            if not isinstance(alias, str) or not isinstance(q, Quantity):
                raise TypeError(
                    "additional_outputs must map aliases to Quantity objects"
                )
        self.parameters["additional_outputs"] = {
            alias: q.to_dict() for alias, q in self._additional_output_requests.items()
        }
        self._neighbors = {}
        self.additional_outputs = {}
        # Extra inputs can depend on the request. Track their actual values for
        # the cached calculation, including callback-provided and pair inputs.
        self._cached_inputs = []
        self.implemented_properties = ["energy", "energies", "forces", "stress"]

    def _select_device(self, requested):
        import sys

        torch = sys.modules.get("torch")
        supported = self._capabilities.supported_devices
        candidates = [str(requested)] if requested is not None else supported
        for candidate in candidates:
            kind = candidate.split(":")[0]
            kind = {"mps": "metal"}.get(kind, kind)
            if kind not in supported:
                continue
            if kind != "cpu":
                try:
                    import torch
                except ImportError:
                    continue
                if kind in ("cuda", "rocm") and not torch.cuda.is_available():
                    continue
                if kind == "metal" and not torch.backends.mps.is_available():
                    continue
            device = candidate.replace("metal", "mps").replace("rocm", "cuda")
            if torch is not None and isinstance(self._model, torch.nn.Module):
                self._model.to(device=device)
            return device
        raise ValueError(
            f"no available device among {candidates}, model supports {supported}"
        )

    def model(self):
        return self._model

    def metadata(self):
        return self._model.metadata()

    def todict(self):
        if "model" not in self.parameters or self._input_provider is not None:
            raise RuntimeError("todict requires a model source and no input_provider")
        return dict(self.parameters)

    def _pick(self, name, sample_kind, required=True):
        variant = self._variants.get(name, self._variants.get("energy"))
        candidates = [
            q
            for q in self._capabilities.outputs
            if q.sample_kind == sample_kind and q.name.split("/")[0] == name
        ]
        target = name if variant is None else f"{name}/{variant}"
        matches = [q for q in candidates if q.name == target]
        if matches:
            return matches[0]
        if variant is None and len(candidates) == 1:
            return candidates[0]
        if required:
            raise PropertyNotImplementedError(
                f"model does not unambiguously support {target!r} "
                f"with {sample_kind!r} samples"
            )
        return None

    def _arrays(self, array):
        array = np.ascontiguousarray(array)
        if self._device == "cpu":
            return array
        import torch

        return torch.as_tensor(array, device=self._device)

    def _input(self, atoms, quantity):
        if self._input_provider is not None:
            value = self._input_provider(atoms, quantity)
            if value is not None:
                if not isinstance(value, TensorMap):
                    raise TypeError(
                        "input_provider must return a Core TensorMap or None"
                    )
                return value
        name = quantity.name
        if quantity.sample_kind == "atom":
            standard = PER_ATOM_QUANTITIES
            prefix, source = "ase::arrays::", atoms.arrays
            samples = Labels(
                ["system", "atom"],
                np.column_stack(
                    [np.zeros(len(atoms), dtype=np.int32), np.arange(len(atoms))]
                ),
            )
        elif quantity.sample_kind == "system":
            standard = PER_SYSTEM_QUANTITIES
            prefix, source = "ase::info::", atoms.info
            samples = Labels("system", np.zeros((1, 1), dtype=np.int32))
        else:
            raise ValueError(
                f"input_provider is required for {name!r} ({quantity.sample_kind})"
            )
        if name.startswith(prefix):
            key = name[len(prefix) :]
            if key not in source:
                raise ValueError(f"model requested missing ASE input {name!r}")
            values = source[key]
            if quantity.sample_kind == "system":
                values = [values]
            unit, property_name = "", "_"
        elif name in standard:
            info = standard[name]
            values = info["getter"](atoms)
            unit, property_name = info["unit"], info["properties_name"]
        else:
            raise ValueError(
                f"model requested input {name!r}, which is not available in ASE"
            )
        values = np.asarray(values, dtype=self._dtype)
        if values.shape not in ((len(samples),), (len(samples), 3)):
            raise ValueError(f"unexpected shape {values.shape} for input {name!r}")
        components = [] if values.ndim == 1 else [Labels.range("xyz", 3)]
        block = TensorBlock(
            values[..., None],
            samples,
            components,
            Labels.range(property_name, 1),
        )
        tensor = TensorMap(Labels.range("_", 1), [block])
        if self._device != "cpu":
            tensor = tensor.to(device=self._device, arrays="torch")
        tensor.set_info("unit", unit)
        return tensor

    def _systems(self, atoms_list, requests):
        inputs = self._model.requested_inputs(requests)
        pair_lists = self._model.requested_pair_lists(requests)
        factor = _length_factor(self._capabilities.length_unit)
        systems = []
        for atoms in atoms_list:
            cell = np.array(atoms.cell.array, dtype=self._dtype, copy=True)
            cell[~atoms.pbc] = 0
            system = System(
                "angstrom",
                self._arrays(atoms.numbers.astype(np.int32)),
                self._arrays(atoms.positions.astype(self._dtype)),
                self._arrays(cell),
                self._arrays(atoms.pbc),
            )
            for options in pair_lists:
                cutoff = options.cutoff * factor
                key = (cutoff, options.full_list, options.strict)
                if key not in self._neighbors:
                    self._neighbors[key] = vesin.NeighborList(
                        cutoff=cutoff,
                        full_list=options.full_list,
                        skin=self.parameters["neighbor_skin"],
                    )
                i, j, shifts, vectors = self._neighbors[key].compute(
                    points=atoms.positions,
                    box=cell.astype(np.float64),
                    periodic=atoms.pbc,
                    quantities="ijSD",
                    copy=True,
                )
                pairs = TensorBlock(
                    vectors.astype(self._dtype).reshape(-1, 3, 1),
                    Labels(
                        [
                            "first_atom",
                            "second_atom",
                            "cell_shift_a",
                            "cell_shift_b",
                            "cell_shift_c",
                        ],
                        np.column_stack([i, j, shifts]).astype(np.int32),
                    ),
                    [Labels.range("xyz", 3)],
                    Labels.range("distance", 1),
                )
                if self._device != "cpu":
                    pairs = pairs.to(device=self._device, arrays="torch")
                system.add_pairs(options, pairs)
            for q in inputs:
                system.add_custom_data(q.name, self._input(atoms, q))
            systems.append(system)
        return systems, inputs

    def run_model(self, atoms, outputs, selected_atoms=None):
        """Return Core TensorMaps in request order (or by alias for a mapping)."""
        aliases = list(outputs) if isinstance(outputs, dict) else None
        requests = list(outputs.values()) if aliases is not None else list(outputs)
        atoms_list = [atoms] if isinstance(atoms, ase.Atoms) else list(atoms)
        systems, _ = self._systems(atoms_list, requests)
        results = execute_model(
            self._model,
            systems,
            requests,
            selected_atoms=selected_atoms,
            check_consistency=self.parameters["check_consistency"],
        )
        return (
            dict(zip(aliases, results, strict=True)) if aliases is not None else results
        )

    def _requests(self, properties, opportunistic=False):
        unknown = set(properties) - set(self.implemented_properties)
        if unknown:
            raise PropertyNotImplementedError(
                f"unsupported ASE properties: {sorted(unknown)}"
            )
        requests = {}
        need_forces = "forces" in properties
        need_stress = "stress" in properties
        need_energy = bool(set(properties) & {"energy", "energies"})
        need_energy |= need_forces and not self._nc_forces
        need_energy |= need_stress and not self._nc_stress
        if need_energy:
            declared = self._pick("energy", "system", required=False)
            if declared is None:
                declared = self._pick("energy", "atom")
            gradients = []
            for gradient, needed, nc in [
                ("positions", need_forces, self._nc_forces),
                ("strain", need_stress, self._nc_stress),
            ]:
                if not nc and (needed or opportunistic):
                    if gradient in declared.gradients:
                        gradients.append(gradient)
                    elif needed:
                        raise PropertyNotImplementedError(
                            f"model does not support {gradient} gradients"
                        )
            requests["energy"] = Quantity(
                name=declared.name,
                unit="eV",
                sample_kind=declared.sample_kind,
                gradients=gradients,
            )
        if "energies" in properties:
            declared = self._pick("energy", "atom")
            requests["energies"] = Quantity(
                name=declared.name, unit="eV", sample_kind="atom"
            )
        for name, kind, unit, needed, alias in [
            (
                "non_conservative_force",
                "atom",
                "eV/angstrom",
                self._nc_forces and (need_forces or opportunistic),
                "forces",
            ),
            (
                "non_conservative_stress",
                "system",
                "eV/angstrom^3",
                self._nc_stress and (need_stress or opportunistic),
                "stress",
            ),
        ]:
            if needed:
                declared = self._pick(name, kind)
                requests[alias] = Quantity(
                    name=declared.name, unit=unit, sample_kind=kind
                )
        return requests

    @staticmethod
    def _per_atom(block, atoms_list, sign=1):
        values = np.asarray(block.values)[..., 0] * sign
        result = [np.zeros((len(atoms), *values.shape[1:])) for atoms in atoms_list]
        for i, array in enumerate(result):
            mask = block.samples.column("system") == i
            np.add.at(array, block.samples.column("atom")[mask], values[mask])
        return result

    def _results(self, predictions, atoms_list):
        results = {}
        if "energy" in predictions:
            block = _numpy(predictions["energy"]).block()
            energy = np.zeros(len(atoms_list))
            np.add.at(energy, block.samples.column("system"), block.values[:, 0])
            results["energy"] = energy.tolist()
            if "positions" in block.gradients_list():
                results["forces"] = self._per_atom(
                    block.gradient("positions"), atoms_list, -1
                )
            if "strain" in block.gradients_list():
                gradient = block.gradient("strain")
                stresses = np.zeros((len(atoms_list), 3, 3))
                # Strain gradients refer to the parent energy samples.
                systems = block.samples.column("system")[
                    gradient.samples.column("sample")
                ]
                np.add.at(stresses, systems, gradient.values[..., 0])
                if all(atoms.cell.volume > 0 for atoms in atoms_list):
                    results["stress"] = [
                        s / a.cell.volume
                        for s, a in zip(stresses, atoms_list, strict=True)
                    ]
        if "energies" in predictions:
            results["energies"] = self._per_atom(
                _numpy(predictions["energies"]).block(), atoms_list
            )
        if "forces" in predictions:
            forces = self._per_atom(_numpy(predictions["forces"]).block(), atoms_list)
            results["forces"] = [f - f.mean(axis=0, keepdims=True) for f in forces]
        if "stress" in predictions:
            block = _numpy(predictions["stress"]).block()
            stresses = np.zeros((len(atoms_list), 3, 3))
            np.add.at(stresses, block.samples.column("system"), block.values[..., 0])
            results["stress"] = list(stresses)
        return results

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        if "stress" in properties:
            if atoms.cell.volume == 0:
                raise PropertyNotImplementedError(
                    "stress requires a non-zero cell volume"
                )
            if not atoms.pbc.all():
                warnings.warn(
                    "stress requested for a system not periodic in all directions",
                    stacklevel=2,
                )
        opportunistic = self.parameters["do_gradients_with_energy"] and bool(
            set(properties) & {"energy", "energies"}
        )
        requests = self._requests(properties, opportunistic)
        uq = self._pick("energy_uncertainty", "atom", required=False)
        if (
            "energy" in requests
            and uq is not None
            and self.parameters["uncertainty_threshold"] is not None
        ):
            requests["uncertainty"] = Quantity(
                name=uq.name, unit="eV", sample_kind="atom"
            )
        all_requests = list(requests.values()) + list(
            self._additional_output_requests.values()
        )
        systems, inputs = self._systems([atoms], all_requests)
        tensors = execute_model(
            self._model,
            systems,
            all_requests,
            check_consistency=self.parameters["check_consistency"],
        )
        predictions = dict(zip(requests, tensors[: len(requests)], strict=True))
        results = self._results(predictions, [atoms])
        self.results = {name: values[0] for name, values in results.items()}
        if "stress" in self.results:
            self.results["stress"] = _full_3x3_to_voigt_6_stress(self.results["stress"])
        self.additional_outputs = dict(
            zip(self._additional_output_requests, tensors[len(requests) :], strict=True)
        )
        self._cached_inputs = [
            (q, _numpy(systems[0].custom_data(q.name)).copy()) for q in inputs
        ]
        if "uncertainty" in predictions:
            values = self._per_atom(
                _numpy(predictions["uncertainty"]).block(), [atoms]
            )[0]
            threshold = self.parameters["uncertainty_threshold"]
            if np.any(values > threshold):
                warnings.warn(
                    f"Atomic energy uncertainties exceed {threshold} eV "
                    f"for atoms {np.where(values > threshold)[0]}",
                    stacklevel=2,
                )

    def check_state(self, atoms, tol=1e-15):
        import metatensor

        changes = super().check_state(atoms, tol)
        for quantity, old in self._cached_inputs:
            new = _numpy(self._input(atoms, quantity))
            if not metatensor.equal(old, new):
                changes.append(quantity.name)
        return changes

    def compute_energy(
        self, atoms, compute_forces_and_stresses=False, *, per_atom=False
    ):
        single = isinstance(atoms, ase.Atoms)
        atoms_list = [atoms] if single else list(atoms)
        if not atoms_list:
            return {"energy": [], **({"energies": []} if per_atom else {})}
        properties = ["energy"] + (["energies"] if per_atom else [])
        if compute_forces_and_stresses:
            properties.append("forces")
            if all(a.pbc.all() for a in atoms_list):
                properties.append("stress")
        requests = self._requests(properties)
        predictions = self.run_model(atoms_list, requests)
        results = self._results(predictions, atoms_list)
        return {k: v[0] for k, v in results.items()} if single else results
