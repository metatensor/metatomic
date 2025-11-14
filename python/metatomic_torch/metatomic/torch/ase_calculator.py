import logging
import os
import pathlib
import warnings
from typing import Dict, List, Optional, Tuple, Union

import metatensor.torch as mts
import numpy as np
import torch
import vesin
from metatensor.torch import Labels, TensorBlock, TensorMap
from torch.profiler import record_function

from . import (
    AtomisticModel,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
    load_atomistic_model,
    pick_device,
    pick_output,
    register_autograd_neighbors,
)


import ase  # isort: skip
import ase.neighborlist  # isort: skip
import ase.calculators.calculator  # isort: skip
from ase.calculators.calculator import (  # isort: skip
    InputError,
    PropertyNotImplementedError,
    all_properties as ALL_ASE_PROPERTIES,
)

FilePath = Union[str, bytes, pathlib.PurePath]

LOGGER = logging.getLogger(__name__)


STR_TO_DTYPE = {
    "float32": torch.float32,
    "float64": torch.float64,
}


class MetatomicCalculator(ase.calculators.calculator.Calculator):
    """
    The :py:class:`MetatomicCalculator` class implements ASE's
    :py:class:`ase.calculators.calculator.Calculator` API using metatomic
    models to compute energy, forces and any other supported property.

    This class can be initialized with any :py:class:`AtomisticModel`, and
    used to run simulations using ASE's MD facilities.

    Neighbor lists are computed using the fast
    `vesin <https://luthaf.fr/vesin/latest/index.html>`_ neighbor list library,
    unless the system has mixed periodic and non-periodic boundary conditions (which
    are not yet supported by ``vesin``), in which case the slower ASE neighbor list
    is used.
    """

    def __init__(
        self,
        model: Union[FilePath, AtomisticModel],
        *,
        additional_outputs: Optional[Dict[str, ModelOutput]] = None,
        extensions_directory=None,
        check_consistency=False,
        device=None,
        variants: Optional[Dict[str, Optional[str]]] = None,
        non_conservative=False,
        do_gradients_with_energy=True,
        uncertainty_threshold=0.1,
    ):
        """
        :param model: model to use for the calculation. This can be a file path, a
            Python instance of :py:class:`AtomisticModel`, or the output of
            :py:func:`torch.jit.script` on :py:class:`AtomisticModel`.
        :param additional_outputs: Dictionary of additional outputs to be computed by
            the model. These outputs will always be computed whenever the
            :py:meth:`calculate` function is called (e.g. by
            :py:meth:`ase.Atoms.get_potential_energy`,
            :py:meth:`ase.optimize.optimize.Dynamics.run`, *etc.*) and stored in the
            :py:attr:`additional_outputs` attribute. If you want more control over when
            and how to compute specific outputs, you should use :py:meth:`run_model`
            instead.
        :param extensions_directory: if the model uses extensions, we will try to load
            them from this directory
        :param check_consistency: should we check the model for consistency when
            running, defaults to False.
        :param device: torch device to use for the calculation. If ``None``, we will try
            the options in the model's ``supported_device`` in order.
        :param variants: dictionary mapping output names to a variant that should be
            used for the calculations (e.g. ``{"energy": "PBE"}``). If ``"energy"`` is
            set to a variant also the uncertainty and non conservative outputs will be
            taken from this variant. This behaviour can be overriden by setting the
            corresponding keys explicitly to ``None`` or to another value (e.g.
            ``{"energy_uncertainty": "r2scan"}``).
        :param non_conservative: if ``True``, the model will be asked to compute
            non-conservative forces and stresses. This can afford a speed-up,
            potentially at the expense of physical correctness (especially in molecular
            dynamics simulations).
        :param do_gradients_with_energy: if ``True``, this calculator will always
            compute the energy gradients (forces and stress) when the energy is
            requested (e.g. through ``atoms.get_potential_energy()``). Because the
            results of a calculation are cached by ASE, this means future calls to
            ``atom.get_forces()`` will return immediately, without needing to execute
            the model again. If you are mainly interested in the energy, you can set
            this to ``False`` and enjoy a faster model. Forces will still be calculated
            if requested with ``atoms.get_forces()``.
        :param uncertainty_threshold: threshold for the atomic energy uncertainty in eV.
            This will only be used if the model supports atomic uncertainty estimation
            (https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c00704). Set this to
            ``None`` to disable uncertainty quantification even if the model supports
            it.
        """
        super().__init__()

        self.parameters = {
            "extensions_directory": extensions_directory,
            "check_consistency": bool(check_consistency),
            "variants": variants,
            "non_conservative": bool(non_conservative),
            "do_gradients_with_energy": bool(do_gradients_with_energy),
            "additional_outputs": additional_outputs,
            "uncertainty_threshold": uncertainty_threshold,
        }

        # Load the model
        if isinstance(model, (str, bytes, pathlib.PurePath)):
            if not os.path.exists(model):
                raise InputError(f"given model path '{model}' does not exist")

            # only store the model in self.parameters if is it the path to a file
            self.parameters["model"] = str(model)

            model = load_atomistic_model(
                model, extensions_directory=extensions_directory
            )

        elif isinstance(model, torch.jit.RecursiveScriptModule):
            if model.original_name != "AtomisticModel":
                raise InputError(
                    "torch model must be 'AtomisticModel', "
                    f"got '{model.original_name}' instead"
                )
        elif isinstance(model, AtomisticModel):
            # nothing to do
            pass
        else:
            raise TypeError(f"unknown type for model: {type(model)}")

        self.parameters["device"] = str(device) if device is not None else None
        # get the best device according what the model supports and what's available on
        # the current machine
        capabilities = model.capabilities()
        self._device = torch.device(
            pick_device(capabilities.supported_devices, self.parameters["device"])
        )

        if capabilities.dtype in STR_TO_DTYPE:
            self._dtype = STR_TO_DTYPE[capabilities.dtype]
        else:
            raise ValueError(
                f"found unexpected dtype in model capabilities: {capabilities.dtype}"
            )

        # resolve the output keys to use based on the requested variants
        variants = variants or {}
        default_variant = variants.get("energy")

        resolved_variants = {
            key: variants.get(key, default_variant)
            for key in [
                "energy",
                "energy_uncertainty",
                "non_conservative_forces",
                "non_conservative_stress",
            ]
        }

        outputs = capabilities.outputs
        self._energy_key = pick_output("energy", outputs, resolved_variants["energy"])

        has_energy_uq = any("energy_uncertainty" in key for key in outputs.keys())
        if has_energy_uq and uncertainty_threshold is not None:
            self._energy_uq_key = pick_output(
                "energy_uncertainty", outputs, resolved_variants["energy_uncertainty"]
            )
        else:
            self._energy_uq_key = "energy_uncertainty"

        if non_conservative:
            if (
                "non_conservative_stress" in variants
                and "non_conservative_forces" in variants
                and (
                    (variants["non_conservative_stress"] is None)
                    != (variants["non_conservative_forces"] is None)
                )
            ):
                raise ValueError(
                    "if both 'non_conservative_stress' and "
                    "'non_conservative_forces' are present in `variants`, they "
                    "must either be both `None` or both not `None`."
                )

            self._nc_forces_key = pick_output(
                "non_conservative_forces",
                outputs,
                resolved_variants["non_conservative_forces"],
            )
            self._nc_stress_key = pick_output(
                "non_conservative_stress",
                outputs,
                resolved_variants["non_conservative_stress"],
            )
        else:
            self._nc_forces_key = "non_conservative_forces"
            self._nc_stress_key = "non_conservative_stress"

        if additional_outputs is None:
            self._additional_output_requests = {}
        else:
            assert isinstance(additional_outputs, dict)
            for name, output in additional_outputs.items():
                assert isinstance(name, str)
                assert isinstance(output, torch.ScriptObject)
                assert "explicit_gradients_setter" in output._method_names(), (
                    "outputs must be ModelOutput instances"
                )

            self._additional_output_requests = additional_outputs

        self._model = model.to(device=self._device)

        self._calculate_uncertainty = (
            self._energy_uq_key in self._model.capabilities().outputs
            # we require per-atom uncertainties to capture local effects
            and self._model.capabilities().outputs[self._energy_uq_key].per_atom
            and uncertainty_threshold is not None
        )

        if self._calculate_uncertainty:
            assert uncertainty_threshold is not None
            if uncertainty_threshold <= 0.0:
                raise ValueError(
                    f"`uncertainty_threshold` is {uncertainty_threshold} but must "
                    "be positive"
                )

        # We do our own check to verify if a property is implemented in `calculate()`,
        # so we pretend to be able to compute all properties ASE knows about.
        self.implemented_properties = ALL_ASE_PROPERTIES

        self.additional_outputs: Dict[str, TensorMap] = {}
        """
        Additional outputs computed by :py:meth:`calculate` are stored in this
        dictionary.

        The keys will match the keys of the ``additional_outputs`` parameters to the
        constructor; and the values will be the corresponding raw
        :py:class:`metatensor.torch.TensorMap` produced by the model.
        """

    def todict(self):
        if "model" not in self.parameters:
            raise RuntimeError(
                "can not save metatensor model in ASE `todict`, please initialize "
                "`MetatomicCalculator` with a path to a saved model file if you need "
                "to use `todict`"
            )

        return self.parameters

    @classmethod
    def fromdict(cls, data):
        return MetatomicCalculator(**data)

    def metadata(self) -> ModelMetadata:
        """Get the metadata of the underlying model"""
        return self._model.metadata()

    def run_model(
        self,
        atoms: Union[ase.Atoms, List[ase.Atoms]],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """
        Run the model on the given ``atoms``, computing the requested ``outputs`` and
        only these.

        The output of the model is returned directly, and as such the blocks' ``values``
        will be :py:class:`torch.Tensor`.

        This is intended as an easy way to run metatensor models on
        :py:class:`ase.Atoms` when the model can compute outputs not supported by the
        standard ASE's calculator interface.

        All the parameters have the same meaning as the corresponding ones in
        :py:meth:`metatomic.torch.ModelInterface.forward`.

        :param atoms: :py:class:`ase.Atoms`, or list of :py:class:`ase.Atoms`, on which
            to run the model
        :param outputs: outputs of the model that should be predicted
        :param selected_atoms: subset of atoms on which to run the calculation
        """
        if isinstance(atoms, ase.Atoms):
            atoms_list = [atoms]
        else:
            atoms_list = atoms

        systems = []
        for atoms in atoms_list:
            types, positions, cell, pbc = _ase_to_torch_data(
                atoms=atoms, dtype=self._dtype, device=self._device
            )
            system = System(types, positions, cell, pbc)
            # Compute the neighbors lists requested by the model
            for options in self._model.requested_neighbor_lists():
                neighbors = _compute_ase_neighbors(
                    atoms, options, dtype=self._dtype, device=self._device
                )
                register_autograd_neighbors(
                    system,
                    neighbors,
                    check_consistency=self.parameters["check_consistency"],
                )
                system.add_neighbor_list(options, neighbors)
            systems.append(system)

        available_outputs = self._model.capabilities().outputs
        for key in outputs:
            if key not in available_outputs:
                raise ValueError(f"this model does not support '{key}' output")

        options = ModelEvaluationOptions(
            length_unit="angstrom",
            outputs=outputs,
            selected_atoms=selected_atoms,
        )
        return self._model(
            systems=systems,
            options=options,
            check_consistency=self.parameters["check_consistency"],
        )

    def calculate(
        self,
        atoms: ase.Atoms,
        properties: List[str],
        system_changes: List[str],
    ) -> None:
        """
        Compute some ``properties`` with this calculator, and return them in the format
        expected by ASE.

        This is not intended to be called directly by users, but to be an implementation
        detail of ``atoms.get_energy()`` and related functions. See
        :py:meth:`ase.calculators.calculator.Calculator.calculate` for more information.
        """
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )

        # In the next few lines, we decide which properties to calculate among energy,
        # forces and stress. In addition to the requested properties, we calculate the
        # energy if any of the three is requested, as it is an intermediate step in the
        # calculation of the other two. We also calculate the forces if the stress is
        # requested, and vice-versa. The overhead for the latter operation is also
        # small, assuming that the majority of the model computes forces and stresses
        # by backward propagation as opposed to forward-mode differentiation.
        calculate_energy = (
            "energy" in properties
            or "energies" in properties
            or "forces" in properties
            or "stress" in properties
        )
        calculate_energies = "energies" in properties
        calculate_forces = "forces" in properties or "stress" in properties
        calculate_stress = "stress" in properties
        if calculate_stress and not atoms.pbc.all():
            warnings.warn(
                "stress requested but likely to be wrong, since the system is not "
                "periodic in all directions",
                stacklevel=2,
            )
        if "forces" in properties and atoms.pbc.all():
            # we have PBCs, and, since the user/integrator requested forces, we will run
            # backward anyway, so let's do the stress as well for free (this saves
            # another forward-backward call later if the stress is requested)
            calculate_stress = True
        if "stresses" in properties:
            raise NotImplementedError("'stresses' are not implemented yet")

        if self.parameters["do_gradients_with_energy"]:
            if calculate_energies or calculate_energy:
                calculate_forces = True
                calculate_stress = True

        with record_function("MetatomicCalculator::prepare_inputs"):
            outputs = self._ase_properties_to_metatensor_outputs(
                properties,
                calculate_forces=calculate_forces,
                calculate_stress=calculate_stress,
                calculate_stresses=False,
            )
            outputs.update(self._additional_output_requests)
            if calculate_energy and self._calculate_uncertainty:
                outputs[self._energy_uq_key] = ModelOutput(
                    quantity="energy",
                    unit="eV",
                    per_atom=True,
                    explicit_gradients=[],
                )

            capabilities = self._model.capabilities()
            for name in outputs.keys():
                if name not in capabilities.outputs:
                    raise ValueError(
                        f"you asked for the calculation of {name}, but this model "
                        "does not support it"
                    )

            types, positions, cell, pbc = _ase_to_torch_data(
                atoms=atoms, dtype=self._dtype, device=self._device
            )

            do_backward = False
            if calculate_forces and not self.parameters["non_conservative"]:
                do_backward = True
                positions.requires_grad_(True)

            if calculate_stress and not self.parameters["non_conservative"]:
                do_backward = True

                strain = torch.eye(
                    3, requires_grad=True, device=self._device, dtype=self._dtype
                )

                positions = positions @ strain
                positions.retain_grad()

                cell = cell @ strain

            run_options = ModelEvaluationOptions(
                length_unit="angstrom",
                outputs=outputs,
                selected_atoms=None,
            )

        with record_function("MetatomicCalculator::compute_neighbors"):
            # convert from ase.Atoms to metatomic.torch.System
            system = System(types, positions, cell, pbc)

            for options in self._model.requested_neighbor_lists():
                neighbors = _compute_ase_neighbors(
                    atoms, options, dtype=self._dtype, device=self._device
                )
                register_autograd_neighbors(
                    system,
                    neighbors,
                    check_consistency=self.parameters["check_consistency"],
                )
                system.add_neighbor_list(options, neighbors)

        # no `record_function` here, this will be handled by AtomisticModel
        outputs = self._model(
            [system],
            run_options,
            check_consistency=self.parameters["check_consistency"],
        )
        energy = outputs[self._energy_key]

        with record_function("MetatomicCalculator::sum_energies"):
            if run_options.outputs[self._energy_key].per_atom:
                assert len(energy) == 1
                assert energy.sample_names == ["system", "atom"]
                assert torch.all(energy.block().samples["system"] == 0)
                energies = energy
                assert energies.block().values.shape == (len(atoms), 1)

                energy = mts.sum_over_samples(energy, sample_names=["atom"])

            assert len(energy.block().gradients_list()) == 0
            assert energy.block().values.shape == (1, 1)

        with record_function("ASECalculator::uncertainty_warning"):
            if calculate_energy and self._calculate_uncertainty:
                uncertainty = outputs[self._energy_uq_key].block().values
                assert uncertainty.shape == (len(atoms), 1)
                uncertainty = uncertainty.detach().cpu().numpy()

                threshold = self.parameters["uncertainty_threshold"]
                if np.any(uncertainty > threshold):
                    warnings.warn(
                        "Some of the atomic energy uncertainties are larger than the "
                        f"threshold of {threshold} eV. The prediction is above the "
                        f"threshold for atoms {np.where(uncertainty > threshold)[0]}.",
                        stacklevel=2,
                    )

        if do_backward:
            if energy.block().values.grad_fn is None:
                # did the user actually request a gradient, or are we trying to
                # compute one just for efficiency?
                if "forces" in properties or "stress" in properties:
                    # the user asked for it, let it fail below
                    pass
                else:
                    # we added the calculation, let's remove it
                    do_backward = False
                    calculate_forces = False
                    calculate_stress = False

        with record_function("MetatomicCalculator::run_backward"):
            if do_backward:
                energy.block().values.backward()

        with record_function("MetatomicCalculator::convert_outputs"):
            self.results = {}

            if calculate_energies:
                energies_values = energies.block().values.detach().reshape(-1)
                energies_values = energies_values.to(device="cpu").to(
                    dtype=torch.float64
                )
                atom_indexes = energies.block().samples.column("atom")

                result = torch.zeros_like(energies_values)
                result.index_add_(0, atom_indexes, energies_values)
                self.results["energies"] = result.numpy()

            if calculate_energy:
                energy_values = energy.block().values.detach()
                energy_values = energy_values.to(device="cpu").to(dtype=torch.float64)
                self.results["energy"] = energy_values.numpy()[0, 0]

            if calculate_forces:
                if self.parameters["non_conservative"]:
                    forces_values = outputs[self._nc_forces_key].block().values.detach()
                    # remove any spurious net force
                    forces_values = forces_values - forces_values.mean(
                        dim=0, keepdim=True
                    )
                else:
                    forces_values = -system.positions.grad
                forces_values = forces_values.reshape(-1, 3)
                forces_values = forces_values.to(device="cpu").to(dtype=torch.float64)
                self.results["forces"] = forces_values.numpy()

            if calculate_stress:
                if self.parameters["non_conservative"]:
                    stress_values = outputs[self._nc_stress_key].block().values.detach()
                else:
                    stress_values = strain.grad / atoms.cell.volume
                stress_values = stress_values.reshape(3, 3)
                stress_values = stress_values.to(device="cpu").to(dtype=torch.float64)
                self.results["stress"] = _full_3x3_to_voigt_6_stress(
                    stress_values.numpy()
                )

            self.additional_outputs = {}
            for name in self._additional_output_requests:
                self.additional_outputs[name] = outputs[name]

    def compute_energy(
        self,
        atoms: Union[ase.Atoms, List[ase.Atoms]],
        compute_forces_and_stresses: bool = False,
        *,
        per_atom: bool = False,
    ) -> Dict[str, Union[Union[float, np.ndarray], List[Union[float, np.ndarray]]]]:
        """
        Compute the energy of the given ``atoms``.

        Energies are computed in eV, forces in eV/Å, and stresses in 3x3 tensor format
        and in units of eV/Å^3.

        :param atoms: :py:class:`ase.Atoms`, or list of :py:class:`ase.Atoms`, on which
            to run the model
        :param compute_forces_and_stresses: if ``True``, the model will also compute
            forces and stresses. IMPORTANT: stresses will only be computed if all
            provided systems have periodic boundary conditions in all directions.
        :param per_atom: if ``True``, the per-atom energies will also be
            computed.
        :return: A dictionary with the computed properties. The dictionary will contain
            the ``energy`` as a float. If ``compute_forces_and_stresses`` is True,
            the ``forces`` and ``stress`` will also be included as numpy arrays.
            If ``per_atom`` is True, the ``energies`` key will also be present,
            containing the per-atom energies as a numpy array.
            In case of a list of :py:class:`ase.Atoms`, the dictionary values will
            instead be lists of the corresponding properties, in the same format.
        """
        if isinstance(atoms, ase.Atoms):
            atoms_list = [atoms]
            was_single = True
        else:
            atoms_list = atoms
            was_single = False

        properties = ["energy"]
        energy_per_atom = False
        if per_atom:
            energy_per_atom = True
            properties.append("energies")

        outputs = self._ase_properties_to_metatensor_outputs(
            properties=properties,
            calculate_forces=compute_forces_and_stresses,
            calculate_stress=compute_forces_and_stresses,
            calculate_stresses=False,
        )

        systems = []
        if compute_forces_and_stresses:
            strains = []
        for atoms in atoms_list:
            types, positions, cell, pbc = _ase_to_torch_data(
                atoms=atoms, dtype=self._dtype, device=self._device
            )
            if compute_forces_and_stresses and not self.parameters["non_conservative"]:
                positions.requires_grad_(True)
                strain = torch.eye(
                    3, requires_grad=True, device=self._device, dtype=self._dtype
                )
                positions = positions @ strain
                positions.retain_grad()
                cell = cell @ strain
                strains.append(strain)
            system = System(types, positions, cell, pbc)
            # Compute the neighbors lists requested by the model
            for options in self._model.requested_neighbor_lists():
                neighbors = _compute_ase_neighbors(
                    atoms, options, dtype=self._dtype, device=self._device
                )
                register_autograd_neighbors(
                    system,
                    neighbors,
                    check_consistency=self.parameters["check_consistency"],
                )
                system.add_neighbor_list(options, neighbors)
            systems.append(system)

        options = ModelEvaluationOptions(
            length_unit="angstrom",
            outputs=outputs,
        )
        predictions = self._model(
            systems=systems,
            options=options,
            check_consistency=self.parameters["check_consistency"],
        )
        energies = predictions[self._energy_key]

        if energy_per_atom:
            # Get per-atom energies
            sorted_block = mts.sort_block(energies.block(), axes="samples")
            energies_values = (
                sorted_block.values.detach()
                .reshape(-1)
                .to(device="cpu")
                .to(dtype=torch.float64)
            )

            split_sizes = [len(system) for system in systems]
            atom_indices = sorted_block.samples.column("atom")
            energies_values = torch.split(energies_values, split_sizes, dim=0)
            split_atom_indices = torch.split(atom_indices, split_sizes, dim=0)
            split_energies = []
            for atom_indices, values in zip(
                split_atom_indices, energies_values, strict=True
            ):
                split_energy = torch.zeros(len(atom_indices), dtype=values.dtype)
                split_energy.index_add_(0, atom_indices, values)
                split_energies.append(split_energy)

            total_energy = (
                mts.sum_over_samples(energies, ["atom"])
                .block()
                .values.detach()
                .cpu()
                .numpy()
                .flatten()
                .tolist()
            )
            results_as_numpy_arrays = {
                "energy": total_energy,
                "energies": [e.numpy() for e in split_energies],
            }
        else:
            results_as_numpy_arrays = {
                "energy": energies.block().values.squeeze(-1).detach().cpu().numpy(),
            }

        if compute_forces_and_stresses:
            if self.parameters["non_conservative"]:
                results_as_numpy_arrays["forces"] = (
                    predictions[self._nc_forces_key]
                    .block()
                    .values.squeeze(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                # all the forces are concatenated in a single array, so we need to
                # split them into the original systems
                split_sizes = [len(system) for system in systems]
                split_indices = np.cumsum(split_sizes[:-1])
                results_as_numpy_arrays["forces"] = np.split(
                    results_as_numpy_arrays["forces"], split_indices, axis=0
                )

                # remove net forces
                results_as_numpy_arrays["forces"] = [
                    f - f.mean(axis=0, keepdims=True)
                    for f in results_as_numpy_arrays["forces"]
                ]

                if all(atoms.pbc.all() for atoms in atoms_list):
                    results_as_numpy_arrays["stress"] = [
                        s
                        for s in predictions[self._nc_stress_key]
                        .block()
                        .values.squeeze(-1)
                        .detach()
                        .cpu()
                        .numpy()
                    ]
            else:
                energy_tensor = energies.block().values
                energy_tensor.backward(torch.ones_like(energy_tensor))
                results_as_numpy_arrays["forces"] = [
                    -system.positions.grad.cpu().numpy() for system in systems
                ]
                if all(atoms.pbc.all() for atoms in atoms_list):
                    results_as_numpy_arrays["stress"] = [
                        strain.grad.cpu().numpy() / atoms.cell.volume
                        for strain, atoms in zip(strains, atoms_list, strict=False)
                    ]
        if was_single:
            for key, value in results_as_numpy_arrays.items():
                results_as_numpy_arrays[key] = value[0]
        return results_as_numpy_arrays

    def _ase_properties_to_metatensor_outputs(
        self,
        properties,
        *,
        calculate_forces,
        calculate_stress,
        calculate_stresses,
    ):
        energy_properties = []
        for p in properties:
            if p in ["energy", "energies", "forces", "stress", "stresses"]:
                energy_properties.append(p)
            else:
                raise PropertyNotImplementedError(
                    f"property '{p}' it not yet supported by this calculator, "
                    "even if it might be supported by the model"
                )

        output = ModelOutput(
            quantity="energy",
            unit="ev",
            explicit_gradients=[],
        )

        if "energies" in properties or "stresses" in properties:
            output.per_atom = True
        else:
            output.per_atom = False

        metatensor_outputs = {self._energy_key: output}
        if calculate_forces and self.parameters["non_conservative"]:
            metatensor_outputs[self._nc_forces_key] = ModelOutput(
                quantity="force",
                unit="eV/Angstrom",
                per_atom=True,
            )

        if calculate_stress and self.parameters["non_conservative"]:
            metatensor_outputs[self._nc_stress_key] = ModelOutput(
                quantity="pressure",
                unit="eV/Angstrom^3",
                per_atom=False,
            )

        if calculate_stresses and self.parameters["non_conservative"]:
            raise NotImplementedError(
                "non conservative, per-atom stress is not yet implemented"
            )

        available_outputs = self._model.capabilities().outputs
        for key in metatensor_outputs:
            if key not in available_outputs:
                raise ValueError(f"this model does not support '{key}' output")

        return metatensor_outputs


def _compute_ase_neighbors(atoms, options, dtype, device):
    # options.strict is ignored by this function, since `ase.neighborlist.neighbor_list`
    # only computes strict NL, and these are valid even with `strict=False`

    if np.all(atoms.pbc) or np.all(~atoms.pbc):
        nl_i, nl_j, nl_S, nl_D = vesin.ase_neighbor_list(
            "ijSD",
            atoms,
            cutoff=options.engine_cutoff(engine_length_unit="angstrom"),
        )
    else:
        nl_i, nl_j, nl_S, nl_D = ase.neighborlist.neighbor_list(
            "ijSD",
            atoms,
            cutoff=options.engine_cutoff(engine_length_unit="angstrom"),
        )

    if not options.full_list:
        # The pair selection code here below avoids a relatively slow loop over
        # all pairs to improve performance
        reject_condition = (
            # we want a half neighbor list, so drop all duplicated neighbors
            (nl_j < nl_i)
            | (
                (nl_i == nl_j)
                & (
                    # only create pairs with the same atom twice if the pair spans more
                    # than one unit cell
                    ((nl_S[:, 0] == 0) & (nl_S[:, 1] == 0) & (nl_S[:, 2] == 0))
                    # When creating pairs between an atom and one of its periodic
                    # images, the code generates multiple redundant pairs
                    # (e.g. with shifts 0 1 1 and 0 -1 -1); and we want to only keep one
                    # of these. We keep the pair in the positive half plane of shifts.
                    | (
                        (nl_S.sum(axis=1) < 0)
                        | (
                            (nl_S.sum(axis=1) == 0)
                            & (
                                (nl_S[:, 2] < 0)
                                | ((nl_S[:, 2] == 0) & (nl_S[:, 1] < 0))
                            )
                        )
                    )
                )
            )
        )
        selected = np.logical_not(reject_condition)
        nl_i = nl_i[selected]
        nl_j = nl_j[selected]
        nl_S = nl_S[selected]
        nl_D = nl_D[selected]

    samples = np.concatenate([nl_i[:, None], nl_j[:, None], nl_S], axis=1)
    distances = torch.from_numpy(nl_D).to(dtype=dtype, device=device)

    return TensorBlock(
        values=distances.reshape(-1, 3, 1),
        samples=Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=torch.from_numpy(samples).to(dtype=torch.int32, device=device),
            assume_unique=True,
        ),
        components=[Labels.range("xyz", 3).to(device)],
        properties=Labels.range("distance", 1).to(device),
    )


def _ase_to_torch_data(atoms, dtype, device):
    """Get the positions, cell and pbc from ASE atoms as torch tensors"""

    types = torch.from_numpy(atoms.numbers).to(dtype=torch.int32, device=device)
    positions = torch.from_numpy(atoms.positions).to(dtype=dtype, device=device)
    cell = torch.zeros((3, 3), dtype=dtype, device=device)
    pbc = torch.tensor(atoms.pbc, dtype=torch.bool, device=device)

    cell[pbc] = torch.tensor(atoms.cell[atoms.pbc], dtype=dtype, device=device)

    return types, positions, cell, pbc


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
        else:
            # no quadrature
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
    available = [
        3,
        5,
        7,
        9,
        11,
        13,
        15,
        17,
        19,
        21,
        23,
        25,
        27,
        29,
        31,
        35,
        41,
        47,
        53,
        59,
        65,
        71,
        77,
        83,
        89,
        95,
        101,
        107,
        113,
        119,
        125,
        131,
    ]
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
    A = np.repeat(alpha, gamma.size)  # (N,)
    B = np.repeat(beta, gamma.size)  # (N,)
    G = np.tile(gamma, alpha.size)  # (N,)

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

    # Lattice with column vectors a1,a2,a3 (spglib expects (cell, frac, Z))
    A = atoms.cell.array.T  # (3,3)
    frac = atoms.get_scaled_positions()  # (N,3) in [0,1)
    numbers = atoms.numbers
    N = len(atoms)

    data = spglib.get_symmetry_dataset(
        (atoms.cell.array, frac, numbers),
        symprec=symprec,
        angle_tolerance=angle_tolerance,
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
