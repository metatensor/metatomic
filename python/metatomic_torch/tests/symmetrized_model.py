import re
from typing import Dict, List, Optional

import metatensor.torch as mts
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    SymmetrizedModel,
    System,
    load_atomistic_model,
)
from metatomic.torch.o3 import O3Transformation
from metatomic.torch.o3._decompose import (
    _cartesian_vectors_to_spherical,
    _o3_mu_labels,
    _symmetric_matrices_to_spherical,
    decompose_quantity,
)
from metatomic.torch.o3._quadrature import (
    _rotations_from_euler_angles,
    choose_quadrature,
    get_euler_angles_quadrature,
    get_rotation_quadrature,
)
from metatomic.torch.o3._utils import map_selected_atoms_to_rotated_copies


def _make_single_block_tensor_map(
    values: torch.Tensor, sample_name: str = "sample"
) -> TensorMap:
    """Create a one-block TensorMap test input from ``values``."""
    device = values.device
    components = [
        Labels.range(f"component_{axis}", size).to(device=device)
        for axis, size in enumerate(values.shape[1:-1])
    ]
    return TensorMap(
        Labels("_", torch.tensor([[0]], dtype=torch.int64, device=device)),
        [
            TensorBlock(
                values=values,
                samples=Labels.range(sample_name, values.shape[0]).to(device=device),
                components=components,
                properties=Labels.range("property", values.shape[-1]).to(device=device),
            )
        ],
    )


def _tensor_map_with_components(
    values: torch.Tensor,
    component_names,
) -> TensorMap:
    """Create a one-block TensorMap with the requested component-axis names."""
    components = [
        Labels.range(name, values.shape[axis + 1])
        for axis, name in enumerate(component_names)
    ]
    return TensorMap(
        Labels("_", torch.tensor([[0]], dtype=torch.int64)),
        [
            TensorBlock(
                values=values,
                samples=Labels.range("system", values.shape[0]),
                components=components,
                properties=Labels.range("property", values.shape[-1]),
            )
        ],
    )


class _EmptyModel(torch.nn.Module):
    """Provide the model interface without producing any outputs."""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        return {}


def _forward_test_system(
    positions: List[List[float]],
    dtype: torch.dtype = torch.float64,
    requires_grad: bool = False,
) -> System:
    """Create a non-periodic input System with configurable dtype and autograd."""
    position_values = torch.tensor(
        positions,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    return System(
        types=torch.ones(len(position_values), dtype=torch.int64),
        positions=position_values,
        cell=torch.zeros((3, 3), dtype=dtype),
        pbc=torch.tensor([False, False, False]),
    )


def _system_scalar_tensor_map(
    values: torch.Tensor,
    property_name: str = "property",
) -> TensorMap:
    """Package one scalar response for each System in a model call."""
    device = values.device
    return TensorMap(
        Labels("_", torch.tensor([[0]], dtype=torch.int64, device=device)),
        [
            TensorBlock(
                values=values,
                samples=Labels(
                    "system",
                    torch.arange(
                        len(values),
                        dtype=torch.int64,
                        device=device,
                    ).reshape(-1, 1),
                ),
                components=[],
                properties=Labels(
                    property_name,
                    torch.arange(
                        values.shape[-1],
                        dtype=torch.int64,
                        device=device,
                    ).reshape(-1, 1),
                ),
            )
        ],
    )


class _LinearEnergyModel(torch.nn.Module):
    """Return the first atom's x coordinate as every requested scalar output."""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        values = torch.stack([system.positions[0, 0] for system in systems]).reshape(
            -1, 1
        )
        result = torch.jit.annotate(Dict[str, TensorMap], {})
        for output_name in outputs:
            result[output_name] = _system_scalar_tensor_map(values)
        return result


class _CountingLinearEnergyModel(_LinearEnergyModel):
    """Record how often ``forward`` is called and the requests it receives."""

    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.requested_names: List[List[str]] = []
        self.requested_units: List[str] = []
        self.requested_sample_kinds: List[str] = []
        self.requested_explicit_gradients: List[List[str]] = []
        self.requested_descriptions: List[str] = []

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        self.call_count += 1
        self.requested_names.append(list(outputs.keys()))
        for output in outputs.values():
            self.requested_units.append(output.unit)
            self.requested_sample_kinds.append(output.sample_kind)
            self.requested_explicit_gradients.append(list(output.explicit_gradients))
            self.requested_descriptions.append(output.description)
        return super().forward(systems, outputs, selected_atoms)


class _OffsetLinearEnergyModel(_LinearEnergyModel):
    """Add a large invariant offset to the linear response."""

    def __init__(self, offset: float):
        super().__init__()
        self._offset = offset

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        values = self._offset + torch.stack(
            [system.positions[0, 0] for system in systems]
        ).reshape(-1, 1)
        result = torch.jit.annotate(Dict[str, TensorMap], {})
        for output_name in outputs:
            result[output_name] = _system_scalar_tensor_map(values)
        return result


class _InconsistentSampleModel(torch.nn.Module):
    """Label every returned sample as system 0, whatever the input batch."""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        values = torch.stack([system.positions[0, 0] for system in systems]).reshape(
            -1, 1
        )
        sample_values = torch.stack(
            [
                torch.zeros(len(systems), dtype=torch.int64),
                torch.arange(len(systems), dtype=torch.int64),
            ],
            dim=1,
        )
        result = torch.jit.annotate(Dict[str, TensorMap], {})
        for output_name in outputs:
            result[output_name] = TensorMap(
                Labels("_", torch.tensor([[0]])),
                [
                    TensorBlock(
                        values=values,
                        samples=Labels(["system", "atom"], sample_values),
                        components=[],
                        properties=Labels.range("property", 1),
                    )
                ],
            )
        return result


class _LinearModelWithRequirements(torch.nn.Module):
    """Provide a scalar output while requesting custom data and a neighbor list."""

    def __init__(self):
        super().__init__()
        self._neighbor_list = NeighborListOptions(
            2.5,
            False,
            True,
            "linear model",
        )

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self._neighbor_list]

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        return {
            "mtt::field": ModelOutput(
                unit="eV",
                sample_kind="atom",
                description="Cartesian field used by the model.",
            )
        }

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        values: List[torch.Tensor] = []
        for system in systems:
            neighbors = system.get_neighbor_list(self._neighbor_list)
            field = system.get_data("mtt::field")
            invariant_input = (
                neighbors.values.square().sum() + field.block().values.square().sum()
            )
            values.append(system.positions[0, 0] + 0.01 * invariant_input)
        scalar_values = torch.stack(values).reshape(-1, 1)

        result = torch.jit.annotate(Dict[str, TensorMap], {})
        for output_name in outputs:
            property_name = "energy" if output_name == "energy" else "property"
            result[output_name] = _system_scalar_tensor_map(
                scalar_values,
                property_name,
            )
        return result


def _system_with_linear_model_requirements(
    neighbor_options: NeighborListOptions,
    device: torch.device,
) -> System:
    """Create the float32 System required by ``_LinearModelWithRequirements``."""
    system = _forward_test_system(
        [[1.0, 2.0, 3.0], [1.2, 2.1, 3.1]],
        dtype=torch.float32,
    )
    system.add_neighbor_list(
        neighbor_options,
        TensorBlock(
            values=(system.positions[1] - system.positions[0]).reshape(1, 3, 1),
            samples=Labels(
                [
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.int64),
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        ),
    )
    field = TensorMap(
        Labels(
            "_",
            torch.tensor([[0]], dtype=torch.int64),
        ),
        [
            TensorBlock(
                values=system.positions.unsqueeze(-1),
                samples=Labels.range("atom", len(system)),
                components=[Labels.range("xyz", 3)],
                properties=Labels.range("field", 1),
            )
        ],
    )
    field.set_info("unit", "eV")
    system.add_data("mtt::field", field)
    return system.to(device=device)


class _O3PolynomialSectorModel(torch.nn.Module):
    """Return one analytic polynomial response in every O(3) sector to lambda=3."""

    # the polynomials 1, x, x*y, and x*y*z transform purely in lambda=0..3 with
    # sigma=+1; multiplying by det(positions) flips the parity to sigma=-1

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        device = systems[0].positions.device
        sectors = [
            (o3_lambda, o3_sigma) for o3_lambda in range(4) for o3_sigma in (1, -1)
        ]
        values: List[torch.Tensor] = []
        for system in systems:
            x, y, z = system.positions[0]
            sigma_plus_values = [
                x.new_ones(()),
                x,
                x * y,
                x * y * z,
            ]
            pseudoscalar = torch.det(system.positions)
            system_values: List[torch.Tensor] = []
            for o3_lambda, o3_sigma in sectors:
                value = sigma_plus_values[o3_lambda]
                if o3_sigma == -1:
                    value = pseudoscalar * value
                system_values.append(value)
            values.append(torch.stack(system_values))

        tensor = TensorMap(
            Labels("_", torch.tensor([[0]], dtype=torch.int64, device=device)),
            [
                TensorBlock(
                    values=torch.stack(values),
                    samples=Labels(
                        "system",
                        torch.arange(
                            len(systems),
                            dtype=torch.int64,
                            device=device,
                        ).reshape(-1, 1),
                    ),
                    components=[],
                    properties=Labels(
                        ["source_lambda", "source_sigma"],
                        torch.tensor(sectors, dtype=torch.int64, device=device),
                    ),
                )
            ],
        )
        result = torch.jit.annotate(Dict[str, TensorMap], {})
        for output_name in outputs:
            result[output_name] = tensor
        return result


class _AtomFeatureModel(torch.nn.Module):
    """Return one component-less per-atom feature that is not O(3) invariant."""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        device = systems[0].positions.device
        values: List[torch.Tensor] = []
        samples: List[torch.Tensor] = []
        for system_index, system in enumerate(systems):
            for atom_index in range(len(system)):
                values.append(system.positions[atom_index, 0].reshape(1))
                samples.append(
                    torch.tensor(
                        [system_index, atom_index],
                        dtype=torch.int64,
                        device=device,
                    )
                )

        tensor = TensorMap(
            Labels("_", torch.tensor([[0]], dtype=torch.int64, device=device)),
            [
                TensorBlock(
                    torch.stack(values),
                    Labels(["system", "atom"], torch.stack(samples)),
                    [],
                    Labels.range("feature", 1).to(device=device),
                )
            ],
        )
        result = torch.jit.annotate(Dict[str, TensorMap], {})
        for output_name in outputs:
            result[output_name] = tensor
        return result


class _DegreeSevenEnergyModel(torch.nn.Module):
    """Return an odd degree-seven response with a degree-fourteen square."""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        energies: List[torch.Tensor] = []
        for system in systems:
            x, y, z = system.positions[0]
            fourth_order = 0.625 * (x**4 + y**4 + z**4)
            mixed = x**2 * y**2 + x**2 * z**2 + y**2 * z**2
            energies.append(1000.0 * x * y * z * (fourth_order - mixed))
        return {
            "energy": _system_scalar_tensor_map(torch.stack(energies).reshape(-1, 1))
        }


class _EquivariantOutputModel(torch.nn.Module):
    """Provide exactly equivariant scalar, Cartesian, and spherical test outputs."""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        device = systems[0].positions.device
        result: Dict[str, TensorMap] = {}
        system_samples = Labels(
            "system",
            torch.arange(len(systems), dtype=torch.int64, device=device).reshape(-1, 1),
        )
        placeholder = Labels(
            "_",
            torch.tensor([[0]], dtype=torch.int64, device=device),
        )
        properties = Labels.range("property", 1).to(device=device)

        if "energy" in outputs:
            energy = torch.stack(
                [system.positions.square().sum() for system in systems]
            ).reshape(-1, 1)
            result["energy"] = TensorMap(
                placeholder,
                [TensorBlock(energy, system_samples, [], properties)],
            )

        if "non_conservative_force" in outputs:
            force_values: List[torch.Tensor] = []
            force_samples: List[torch.Tensor] = []
            if selected_atoms is None:
                for system_index, system in enumerate(systems):
                    for atom_index in range(len(system)):
                        force_values.append(system.positions[atom_index])
                        force_samples.append(
                            torch.tensor(
                                [system_index, atom_index],
                                dtype=torch.int64,
                                device=device,
                            )
                        )
            else:
                system_indices = selected_atoms.column("system").to(dtype=torch.long)
                atom_indices = selected_atoms.column("atom").to(dtype=torch.long)
                for row in range(len(selected_atoms)):
                    system_index = int(system_indices[row])
                    atom_index = int(atom_indices[row])
                    force_values.append(systems[system_index].positions[atom_index])
                    force_samples.append(selected_atoms.values[row])

            if len(force_values) == 0:
                force = torch.empty(
                    (0, 3, 1),
                    dtype=systems[0].positions.dtype,
                    device=device,
                )
                samples = torch.empty((0, 2), dtype=torch.int64, device=device)
            else:
                force = torch.stack(force_values).unsqueeze(-1)
                samples = torch.stack(force_samples)
            result["non_conservative_force"] = TensorMap(
                placeholder,
                [
                    TensorBlock(
                        force,
                        Labels(["system", "atom"], samples),
                        [Labels.range("xyz", 3).to(device=device)],
                        properties,
                    )
                ],
            )

        if "non_conservative_stress" in outputs:
            stress = torch.stack(
                [system.positions.T @ system.positions for system in systems]
            ).unsqueeze(-1)
            result["non_conservative_stress"] = TensorMap(
                placeholder,
                [
                    TensorBlock(
                        stress,
                        system_samples,
                        [
                            Labels.range("xyz_1", 3).to(device=device),
                            Labels.range("xyz_2", 3).to(device=device),
                        ],
                        properties,
                    )
                ],
            )

        if "mtt::spherical_vector" in outputs:
            spherical = torch.stack(
                [system.positions[0].roll(-1) for system in systems]
            ).unsqueeze(-1)
            result["mtt::spherical_vector"] = TensorMap(
                Labels(
                    ["o3_lambda", "o3_sigma"],
                    torch.tensor([[1, 1]], dtype=torch.int64, device=device),
                ),
                [
                    TensorBlock(
                        spherical,
                        system_samples,
                        [_o3_mu_labels(1, device)],
                        properties,
                    )
                ],
            )

        if "mtt::spherical_quadrupole" in outputs:
            matrices = torch.stack(
                [
                    torch.outer(system.positions[0], system.positions[0])
                    for system in systems
                ]
            ).unsqueeze(-1)
            _, spherical = _symmetric_matrices_to_spherical(matrices)
            result["mtt::spherical_quadrupole"] = TensorMap(
                Labels(
                    ["o3_lambda", "o3_sigma"],
                    torch.tensor([[2, 1]], dtype=torch.int64, device=device),
                ),
                [
                    TensorBlock(
                        spherical,
                        system_samples,
                        [_o3_mu_labels(2, device)],
                        properties,
                    )
                ],
            )

        return result


class TestQuadrature:
    """Test quadrature weights and grid properties."""

    def test_weights_sum(self):
        """Quadrature weights should sum to 1 (normalized Haar measure on SO(3))."""
        for L_max in [3, 5, 7]:
            lebedev_order, n_inplane = choose_quadrature(L_max)
            _, _, _, w = get_euler_angles_quadrature(lebedev_order, n_inplane)
            # The weights are w_i / (4*pi*K) repeated K times, where w_i sum to 4*pi
            # So total sum = sum(w_i)/(4*pi*K) * K = sum(w_i)/(4*pi) = 1
            assert np.allclose(w.sum(), 1.0, atol=1e-12), (
                f"Weights don't sum to 1 for L_max={L_max}: sum={w.sum()}"
            )

    def test_euler_angle_rotations_are_in_so3(self):
        """Euler-angle matrices should be orthogonal with determinant +1."""
        lebedev_order, n_inplane = choose_quadrature(5)
        alpha, beta, gamma, _ = get_euler_angles_quadrature(lebedev_order, n_inplane)
        rotations = _rotations_from_euler_angles(alpha, beta, gamma)
        matrices = rotations.as_matrix()

        identity = np.broadcast_to(np.eye(3), matrices.shape)
        assert np.allclose(
            matrices @ matrices.transpose(0, 2, 1),
            identity,
            rtol=0.0,
            atol=1e-12,
        )
        assert np.allclose(
            np.linalg.det(matrices),
            1.0,
            rtol=0.0,
            atol=1e-12,
        )

    def test_quadrature_validation(self):
        """Quadrature construction rejects invalid degrees, counts, and orders."""
        message = (
            "the requested quadrature degree max_angular_momentum=132 exceeds the "
            "largest available Lebedev order (131)"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            choose_quadrature(132)

        message = "max_angular_momentum must be non-negative, got -1"
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            choose_quadrature(-1)

        message = "max_angular_momentum must be an integer, got float"
        with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
            choose_quadrature(1.5)

        message = "n_rotations must be positive, got 0"
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            get_rotation_quadrature(3, 0)

        message = "n_rotations must be an integer, got float"
        with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
            get_rotation_quadrature(3, 1.5)

        supported_orders = [
            *range(3, 32, 2),
            *range(35, 132, 6),
        ]
        message = (
            f"unsupported Lebedev order 4; supported orders are {supported_orders}"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            get_rotation_quadrature(4, 3)

    def test_degree_two_grid_resolves_l1_products(self):
        order, n_rotations = choose_quadrature(2)
        rotations, weights = get_rotation_quadrature(order, n_rotations)
        function = rotations[:, 2, 0]

        norm = np.sum(weights * function**2)
        projection_matrix = np.einsum("g,gij,g->ij", weights, rotations, function)
        projected_norm = 3.0 * np.sum(projection_matrix**2)
        assert np.isclose(norm, 1.0 / 3.0, atol=1e-12)
        assert np.isclose(projected_norm, 1.0 / 3.0, atol=1e-12)

    def test_rotation_quadrature_matrices(self):
        """Inversion should pair every proper rotation with an improper partner."""
        rotations, _ = get_rotation_quadrature(11, 5)
        o3_rotations, _ = get_rotation_quadrature(11, 5, include_inversion=True)

        assert len(o3_rotations) == 2 * len(rotations)
        dets = np.linalg.det(o3_rotations)
        assert np.allclose(np.sort(dets), np.repeat([-1.0, 1.0], len(rotations)))


class TestSymmetrizedModelConstruction:
    """Test construction of the quadrature and persistent Wigner-D storage."""

    def test_character_limit_controls_default_grid(self):
        """Character sectors should raise the default grid degree when necessary."""
        model = SymmetrizedModel(
            _EmptyModel(),
            max_angular_momentum_target=0,
            max_angular_momentum_character=2,
        )

        assert model.max_angular_momentum_grid == 4

    def test_rejects_grid_too_small_for_character_sectors(self):
        """An explicit grid must resolve products for every requested sector."""
        message = (
            "max_angular_momentum_grid must be at least twice "
            "max_angular_momentum_character"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            SymmetrizedModel(
                _EmptyModel(),
                max_angular_momentum_target=0,
                max_angular_momentum_character=2,
                max_angular_momentum_grid=3,
            )

    def test_rejects_invalid_constructor_arguments(self):
        """Every integer constructor argument should enforce its documented range."""
        message = "max_angular_momentum_target must be non-negative, got -1"
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            SymmetrizedModel(_EmptyModel(), max_angular_momentum_target=-1)

        message = "max_angular_momentum_target must be an integer, got bool"
        with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
            SymmetrizedModel(_EmptyModel(), max_angular_momentum_target=True)

        message = "max_angular_momentum_input must be an integer, got float"
        with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
            SymmetrizedModel(
                _EmptyModel(),
                max_angular_momentum_target=0,
                max_angular_momentum_input=1.5,
            )

        message = "max_angular_momentum_character must be non-negative, got -1"
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            SymmetrizedModel(
                _EmptyModel(),
                max_angular_momentum_target=0,
                max_angular_momentum_character=-1,
            )

        message = "batch_size must be positive, got 0"
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            SymmetrizedModel(
                _EmptyModel(),
                max_angular_momentum_target=0,
                batch_size=0,
            )

        message = "max_angular_momentum_grid must be non-negative, got -1"
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            SymmetrizedModel(
                _EmptyModel(),
                max_angular_momentum_target=0,
                max_angular_momentum_grid=-1,
            )

    def test_rejects_a_model_stored_on_an_unsupported_device(self):
        """Reject direct construction from a model outside CPU or CUDA."""
        base_model = _EmptyModel()
        base_model.register_buffer("_device_marker", torch.empty(0, device="meta"))

        message = "SymmetrizedModel supports CPU and CUDA execution"
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            SymmetrizedModel(base_model, max_angular_momentum_target=0)


class TestSymmetrizedModelForward:
    """Test how requested averages and diagnostics are computed and returned."""

    def test_character_projection_separates_sectors_through_lambda_three(self):
        """Character projection separates the eight analytic sectors to lambda=3."""
        source_name = "mtt::o3_polynomial_sectors"
        requested_name = "o3::character_projection::" + source_name
        sectors = [
            (chi_lambda, chi_sigma) for chi_lambda in range(4) for chi_sigma in (1, -1)
        ]
        model = SymmetrizedModel(
            _O3PolynomialSectorModel(),
            max_angular_momentum_target=0,
            max_angular_momentum_character=3,
            max_angular_momentum_grid=6,
            batch_size=17,
        )
        system = _forward_test_system(torch.eye(3, dtype=torch.float64).tolist())

        result = model(
            [system],
            {requested_name: ModelOutput(sample_kind="system")},
            None,
        )[requested_name]

        assert result.keys.names == ["chi_lambda", "chi_sigma"]
        assert result.keys.values.tolist() == [list(sector) for sector in sectors]
        expected_properties = Labels(
            ["source_lambda", "source_sigma"],
            torch.tensor(sectors, dtype=torch.int64),
        )
        # O(3) averages on the unit sphere: <x^2>=1/3, <(xy)^2>=1/15,
        # <(xyz)^2>=1/105
        expected_norms = [1.0, 1.0 / 3.0, 1.0 / 15.0, 1.0 / 105.0]
        for key, block in result.items():
            assert block.samples == Labels("system", torch.tensor([[0]]))
            assert block.components == []
            assert block.properties == expected_properties

            expected = torch.zeros((1, len(sectors)), dtype=torch.float64)
            source_index = sectors.index(
                (int(key["chi_lambda"]), int(key["chi_sigma"]))
            )
            expected[0, source_index] = expected_norms[int(key["chi_lambda"])]
            assert torch.allclose(
                block.values,
                expected,
                rtol=0.0,
                atol=1.0e-11,
            )

    def test_energy_results_match_analytic_values_and_reuse_predictions(self):
        """Reuse each energy prediction for its average and both diagnostics."""
        base_model = _CountingLinearEnergyModel()
        batch_size = 5
        model = SymmetrizedModel(
            base_model,
            max_angular_momentum_target=0,
            max_angular_momentum_character=1,
            max_angular_momentum_grid=2,
            batch_size=batch_size,
        )
        system = _forward_test_system([[1.0, 2.0, 3.0]])
        outputs = {
            "energy": ModelOutput(sample_kind="system"),
            "o3::variance::energy": ModelOutput(sample_kind="system"),
            "o3::character_projection::energy": ModelOutput(sample_kind="system"),
        }

        with torch.inference_mode():
            result = model([system], outputs, None)

        assert set(result) == set(outputs)
        n_rotations = len(model._rotation_matrices)
        assert base_model.call_count == 2 * (
            (n_rotations + batch_size - 1) // batch_size
        )
        assert all(names == ["energy"] for names in base_model.requested_names)

        assert torch.allclose(
            result["energy"].block().values,
            torch.zeros((1, 1), dtype=torch.float64),
            atol=1.0e-12,
        )
        # <x^2> under O(3) rotations of r=(1, 2, 3): |r|^2 / 3 = 14/3
        expected_variance = torch.tensor([[14.0 / 3.0]], dtype=torch.float64)
        assert torch.allclose(
            result["o3::variance::energy"].block().values,
            expected_variance,
            atol=1.0e-12,
        )

        projection = result["o3::character_projection::energy"]
        assert projection.keys.names == [
            "o3_lambda",
            "o3_sigma",
            "chi_lambda",
            "chi_sigma",
        ]
        vector_projection = projection.block(
            {
                "o3_lambda": 0,
                "o3_sigma": 1,
                "chi_lambda": 1,
                "chi_sigma": 1,
            }
        )
        assert torch.allclose(
            vector_projection.values.squeeze(1),
            expected_variance,
            atol=1.0e-12,
        )
        for key, block in projection.items():
            if int(key["chi_lambda"]) == 1 and int(key["chi_sigma"]) == 1:
                continue
            assert torch.allclose(
                block.values,
                torch.zeros_like(block.values),
                atol=1.0e-12,
            )

    def test_stress_character_projection_combines_target_and_character_sectors(self):
        """Keep the stress irreps separate from its O(3) character sectors."""
        requested_name = "o3::character_projection::non_conservative_stress"
        model = SymmetrizedModel(
            _EquivariantOutputModel(),
            max_angular_momentum_target=2,
            max_angular_momentum_character=2,
            max_angular_momentum_grid=4,
            batch_size=17,
        )
        system = _forward_test_system([[1.0, 2.0, 3.0], [-0.5, 0.25, 1.0]])

        result = model(
            [system],
            {requested_name: ModelOutput(sample_kind="system")},
            None,
        )

        assert set(result) == {requested_name}
        projection = result[requested_name]
        assert projection.keys.names == [
            "o3_lambda",
            "o3_sigma",
            "chi_lambda",
            "chi_sigma",
        ]
        assert {
            tuple(int(value) for value in key.values) for key in projection.keys
        } == {
            (o3_lambda, 1, chi_lambda, chi_sigma)
            for o3_lambda in (0, 2)
            for chi_lambda in range(3)
            for chi_sigma in (1, -1)
        }

        for key, block in projection.items():
            o3_lambda = int(key["o3_lambda"])
            chi_lambda = int(key["chi_lambda"])
            chi_sigma = int(key["chi_sigma"])
            assert block.components == [_o3_mu_labels(o3_lambda, block.values.device)]

            if chi_lambda == o3_lambda and chi_sigma == 1:
                assert bool(torch.any(block.values > 1.0e-12))
            else:
                assert torch.allclose(
                    block.values,
                    torch.zeros_like(block.values),
                    rtol=0.0,
                    atol=1.0e-11,
                )

    @pytest.mark.parametrize(
        ("requested_name", "unit"),
        [
            ("energy", "eV"),
            ("o3::variance::energy", "(eV)^2"),
            ("o3::character_projection::energy", "(eV)^2"),
        ],
    )
    def test_source_request_contains_only_the_shared_sample_kind(
        self,
        requested_name,
        unit,
    ):
        """Do not pass diagnostic metadata to the underlying source output."""
        base_model = _CountingLinearEnergyModel()
        model = SymmetrizedModel(
            base_model,
            max_angular_momentum_target=0,
            max_angular_momentum_character=1,
            max_angular_momentum_grid=2,
        )

        model(
            [_forward_test_system([[1.0, 2.0, 3.0]])],
            {
                requested_name: ModelOutput(
                    unit=unit,
                    sample_kind="system",
                    description="Metadata for the public result.",
                )
            },
            None,
        )

        assert all(names == ["energy"] for names in base_model.requested_names)
        assert set(base_model.requested_sample_kinds) == {"system"}
        assert set(base_model.requested_units) == {""}
        assert base_model.requested_explicit_gradients == [
            [] for _ in base_model.requested_explicit_gradients
        ]
        assert set(base_model.requested_descriptions) == {""}

    def test_rejects_an_output_above_the_declared_target_rank(self):
        """Reject a rank-two spherical output when the declared limit is one."""
        model = SymmetrizedModel(
            _EquivariantOutputModel(),
            max_angular_momentum_target=1,
        )

        message = (
            "output 'mtt::spherical_quadrupole' contains o3_lambda=2, "
            "exceeding max_angular_momentum_target=1"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            model(
                [_forward_test_system([[1.0, 2.0, 3.0]])],
                {
                    "mtt::spherical_quadrupole": ModelOutput(
                        sample_kind="system",
                    )
                },
                None,
            )

    def test_rejects_a_negative_quadrature_error_and_converges(self):
        """A degree-12 grid rejects the degree-14 response; degree 14 is exact."""
        position = torch.tensor(
            [[-1.12984253e-2, 3.64940445e-4, -9.99936104e-1]],
            dtype=torch.float64,
        )
        position = position / torch.linalg.norm(position)
        system = _forward_test_system(position.tolist())
        variance_name = "o3::variance::energy"
        variance_request = {
            variance_name: ModelOutput(sample_kind="system"),
        }

        underresolved = SymmetrizedModel(
            _DegreeSevenEnergyModel(),
            max_angular_momentum_target=0,
            max_angular_momentum_grid=12,
            batch_size=64,
        )
        message = (
            "finite O(3) variance is materially negative; the quadrature does "
            "not resolve this response. Increase max_angular_momentum_grid above 12 "
            "and check convergence"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            underresolved([system], variance_request, None)

        resolved = SymmetrizedModel(
            _DegreeSevenEnergyModel(),
            max_angular_momentum_target=0,
            max_angular_momentum_grid=14,
            batch_size=64,
        )
        outputs = {
            "energy": ModelOutput(sample_kind="system"),
            variance_name: ModelOutput(sample_kind="system"),
        }
        result = resolved([system], outputs, None)
        # closed-form Haar variance of the degree-seven response at |r| = 1
        expected_variance = 1.0e6 * 17.0 / 137280.0
        assert result["energy"].block().values.item() == pytest.approx(
            0.0,
            abs=1.0e-12,
        )
        assert result[variance_name].block().values.item() == pytest.approx(
            expected_variance,
            rel=1.0e-12,
        )

    @pytest.mark.parametrize(
        "source_name",
        [
            "energy/pbe",
            "mtt::feature::node",
            # the reserved prefix is stripped exactly once, keeping "mtt::aux::"
            "mtt::aux::features",
        ],
    )
    def test_preserves_variant_and_custom_output_names(self, source_name):
        """Return variants and custom outputs under their exact requested names."""
        base_model = _CountingLinearEnergyModel()
        model = SymmetrizedModel(
            base_model,
            max_angular_momentum_target=0,
            max_angular_momentum_grid=2,
        )
        variance_name = "o3::variance::" + source_name
        outputs = {
            source_name: ModelOutput(sample_kind="system"),
            variance_name: ModelOutput(sample_kind="system"),
        }

        result = model(
            [_forward_test_system([[1.0, 2.0, 3.0]])],
            outputs,
            None,
        )

        assert set(result) == set(outputs)
        assert all(names == [source_name] for names in base_model.requested_names)
        assert torch.allclose(
            result[variance_name].block().values,
            torch.tensor([[14.0 / 3.0]], dtype=torch.float64),
            atol=1.0e-12,
        )

    def test_component_less_output_averages_and_measures_invariance(self):
        """Features have no spherical character: plain mean, invariance variance."""
        system = _forward_test_system([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
        model = SymmetrizedModel(
            _AtomFeatureModel(),
            max_angular_momentum_target=0,
            max_angular_momentum_grid=2,
            batch_size=5,
        )
        outputs = {
            "feature": ModelOutput(sample_kind="atom"),
            "o3::variance::feature": ModelOutput(sample_kind="atom"),
        }

        result = model([system], outputs, None)

        mean = result["feature"].block()
        assert mean.samples.values.tolist() == [[0, 0], [0, 1]]
        # the mean of x over O(3) is zero, without any back-rotation
        assert torch.allclose(mean.values, torch.zeros_like(mean.values), atol=1.0e-12)

        variance = result["o3::variance::feature"]
        # the tensor is passed through undecomposed, keeping its original keys
        assert variance.keys.names == ["_"]
        assert torch.allclose(
            variance.block().values,
            # <x^2> - <x>^2 = |r|^2 / 3 for each atom
            (system.positions.square().sum(dim=1) / 3.0).reshape(-1, 1),
            atol=1.0e-12,
        )

    def test_selected_atoms_excludes_unselected_input_systems(self):
        """Selecting only from System 1 must not create samples for System 0."""
        systems = [
            _forward_test_system([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            _forward_test_system([[0.0, 0.0, 3.0], [4.0, 5.0, 6.0]]),
        ]
        model = SymmetrizedModel(
            _EquivariantOutputModel(),
            max_angular_momentum_target=1,
            max_angular_momentum_grid=2,
            batch_size=5,
        )
        outputs = {
            "non_conservative_force": ModelOutput(sample_kind="atom"),
            "o3::variance::non_conservative_force": ModelOutput(sample_kind="atom"),
        }
        selected_atoms = Labels(
            ["system", "atom"],
            torch.tensor([[1, 1]], dtype=torch.int64),
        )

        result = model(systems, outputs, selected_atoms)

        mean = result["non_conservative_force"].block()
        assert mean.samples.values.tolist() == [[1, 1]]
        assert torch.allclose(
            mean.values.squeeze(-1),
            systems[1].positions[1].reshape(1, 3),
            atol=1.0e-12,
        )
        variance = result["o3::variance::non_conservative_force"]
        assert variance.keys.values.tolist() == [[1, 1]]
        assert variance.block().samples.values.tolist() == [[1, 1]]
        assert torch.allclose(
            variance.block().values,
            torch.zeros_like(variance.block().values),
            atol=1.0e-12,
        )

    def test_empty_selected_atoms_returns_empty_outputs(self):
        """A fully empty atom selection must not create artificial samples."""
        systems = [
            _forward_test_system([[1.0, 0.0, 0.0]]),
            _forward_test_system([[0.0, 2.0, 0.0]]),
        ]
        model = SymmetrizedModel(
            _EquivariantOutputModel(),
            max_angular_momentum_target=1,
            max_angular_momentum_grid=2,
            batch_size=5,
        )
        outputs = {
            "non_conservative_force": ModelOutput(sample_kind="atom"),
            "o3::variance::non_conservative_force": ModelOutput(sample_kind="atom"),
        }
        selected_atoms = Labels(
            ["system", "atom"],
            torch.empty((0, 2), dtype=torch.int64),
        )

        result = model(systems, outputs, selected_atoms)

        assert set(result) == set(outputs)
        mean = result["non_conservative_force"].block()
        assert mean.samples.names == ["system", "atom"]
        assert len(mean.samples) == 0
        assert mean.values.shape == (0, 3, 1)

        variance = result["o3::variance::non_conservative_force"]
        assert variance.keys.names == ["o3_lambda", "o3_sigma"]
        assert variance.keys.values.tolist() == [[1, 1]]
        variance_block = variance.block()
        assert variance_block.samples.names == ["system", "atom"]
        assert len(variance_block.samples) == 0
        assert variance_block.components == []
        assert variance_block.values.shape == (0, 1)

    def test_multiple_systems_keep_per_system_rows_in_order(self):
        """Joined outputs should keep one correct row per input System, in order."""
        systems = [
            _forward_test_system([[1.0, 2.0, 3.0]]),
            _forward_test_system([[-0.5, 0.25, 1.0], [0.5, -1.0, 2.0]]),
        ]
        model = SymmetrizedModel(
            _EquivariantOutputModel(),
            max_angular_momentum_target=0,
            max_angular_momentum_grid=2,
            batch_size=5,
        )
        outputs = {
            "energy": ModelOutput(sample_kind="system"),
            "o3::variance::energy": ModelOutput(sample_kind="system"),
        }

        result = model(systems, outputs, None)

        energy = result["energy"].block()
        assert energy.samples.values.tolist() == [[0], [1]]
        expected = torch.stack(
            [system.positions.square().sum() for system in systems]
        ).reshape(-1, 1)
        assert torch.allclose(energy.values, expected, atol=1.0e-12)
        variance = result["o3::variance::energy"].block()
        assert variance.samples.values.tolist() == [[0], [1]]
        assert torch.allclose(
            variance.values,
            torch.zeros_like(variance.values),
            atol=1.0e-12,
        )

    def test_equivariant_outputs_preserve_values_metadata_and_zero_variance(self):
        """Return exact equivariant outputs unchanged and report zero variance."""
        system = _forward_test_system([[1.0, 2.0, 3.0], [-0.5, 0.25, 1.0]])
        sources = [
            "energy",
            "non_conservative_force",
            "non_conservative_stress",
            "mtt::spherical_vector",
            "mtt::spherical_quadrupole",
        ]
        outputs = {
            name: ModelOutput(
                sample_kind="atom" if name == "non_conservative_force" else "system"
            )
            for name in sources
        }
        for name in sources:
            outputs["o3::variance::" + name] = outputs[name]
        model = SymmetrizedModel(
            _EquivariantOutputModel(),
            max_angular_momentum_target=2,
            max_angular_momentum_grid=2,
            batch_size=7,
        )

        result = model([system], outputs, None)

        assert set(result) == set(outputs)
        assert torch.allclose(
            result["energy"].block().values,
            system.positions.square().sum().reshape(1, 1),
            atol=1.0e-12,
        )
        assert torch.allclose(
            result["non_conservative_force"].block().values.squeeze(-1),
            system.positions,
            atol=1.0e-12,
        )
        assert torch.allclose(
            result["non_conservative_stress"].block().values.squeeze(-1),
            (system.positions.T @ system.positions).unsqueeze(0),
            atol=1.0e-12,
        )
        assert torch.allclose(
            result["mtt::spherical_vector"].block().values.squeeze(-1),
            system.positions[0].roll(-1).reshape(1, 3),
            atol=1.0e-12,
        )
        quadrupole = result["mtt::spherical_quadrupole"]
        assert quadrupole.keys.values.tolist() == [[2, 1]]
        _, expected_quadrupole = _symmetric_matrices_to_spherical(
            torch.outer(system.positions[0], system.positions[0]).reshape(1, 3, 3, 1)
        )
        assert torch.allclose(
            quadrupole.block().values,
            expected_quadrupole,
            atol=1.0e-12,
        )

        expected_target_keys = {
            "o3::variance::energy": [[0, 1]],
            "o3::variance::non_conservative_force": [[1, 1]],
            "o3::variance::non_conservative_stress": [[0, 1], [2, 1]],
            "o3::variance::mtt::spherical_vector": [[1, 1]],
            "o3::variance::mtt::spherical_quadrupole": [[2, 1]],
        }
        for name, expected_keys in expected_target_keys.items():
            variance = result[name]
            assert variance.keys.names == ["o3_lambda", "o3_sigma"]
            assert variance.keys.values.tolist() == expected_keys
            for block in variance.blocks():
                assert block.components == []
                assert torch.allclose(
                    block.values,
                    torch.zeros_like(block.values),
                    atol=1.0e-12,
                )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_and_implicit_autograd(self, dtype):
        """Variance should preserve the model dtype and its implicit backward path."""
        system = _forward_test_system(
            [[1.0, 2.0, 3.0]],
            dtype=dtype,
            requires_grad=True,
        )
        model = SymmetrizedModel(
            _LinearEnergyModel(),
            max_angular_momentum_target=0,
            max_angular_momentum_grid=2,
        )
        outputs = {
            "o3::variance::energy": ModelOutput(sample_kind="system"),
        }

        result = model([system], outputs, None)
        variance = result["o3::variance::energy"].block().values

        assert variance.dtype == dtype
        gradient = torch.autograd.grad(variance.sum(), system.positions)[0]
        tolerance = 2.0e-5 if dtype == torch.float32 else 1.0e-12
        assert torch.allclose(
            gradient,
            2.0 * system.positions / 3.0,
            rtol=0.0,
            atol=tolerance,
        )

    def test_average_output_preserves_implicit_autograd(self):
        """The averaged output should keep the implicit backward path to positions."""
        system = _forward_test_system(
            [[1.0, 2.0, 3.0], [-0.5, 0.25, 1.0]],
            requires_grad=True,
        )
        model = SymmetrizedModel(
            _EquivariantOutputModel(),
            max_angular_momentum_target=0,
            max_angular_momentum_grid=2,
        )

        result = model([system], {"energy": ModelOutput(sample_kind="system")}, None)

        gradient = torch.autograd.grad(
            result["energy"].block().values.sum(),
            system.positions,
        )[0]
        assert torch.allclose(
            gradient,
            2.0 * system.positions,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_rejects_unknown_o3_requests(self):
        """Every unrecognized 'o3::' request is reserved, not a source output."""
        model = SymmetrizedModel(
            _LinearEnergyModel(),
            max_angular_momentum_target=0,
            max_angular_momentum_grid=2,
        )
        message = (
            "requested output 'o3::variance_extra::energy' uses the 'o3::' "
            "prefix reserved by SymmetrizedModel, but is neither a variance nor "
            "a character-projection request"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            model(
                [_forward_test_system([[1.0, 2.0, 3.0]])],
                {"o3::variance_extra::energy": ModelOutput(sample_kind="system")},
                None,
            )

    def test_input_limit_distinguishes_spherical_from_cartesian(self):
        """A zero angular-momentum input limit still allows Cartesian custom data."""
        model = SymmetrizedModel(
            _LinearEnergyModel(),
            max_angular_momentum_target=0,
            max_angular_momentum_grid=2,
        )
        outputs = {"energy": ModelOutput(sample_kind="system")}

        cartesian_system = _forward_test_system([[1.0, 2.0, 3.0]])
        cartesian_system.add_data(
            "mtt::field",
            TensorMap(
                Labels("_", torch.tensor([[0]])),
                [
                    TensorBlock(
                        values=torch.ones((1, 3, 1), dtype=torch.float64),
                        samples=Labels.range("atom", 1),
                        components=[Labels.range("xyz", 3)],
                        properties=Labels.range("property", 1),
                    )
                ],
            ),
        )
        model([cartesian_system], outputs, None)

        spherical_system = _forward_test_system([[1.0, 2.0, 3.0]])
        spherical_system.add_data(
            "mtt::field",
            TensorMap(
                Labels(["o3_lambda", "o3_sigma"], torch.tensor([[1, 1]])),
                [
                    TensorBlock(
                        values=torch.ones((1, 3, 1), dtype=torch.float64),
                        samples=Labels.range("atom", 1),
                        components=[_o3_mu_labels(1, torch.device("cpu"))],
                        properties=Labels.range("property", 1),
                    )
                ],
            ),
        )
        message = (
            "custom input 'mtt::field' contains o3_lambda=1, exceeding "
            "max_angular_momentum_input=0"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            model([spherical_system], outputs, None)

    def test_variance_is_stable_with_large_mean_offset(self):
        """A huge invariant offset must not destroy the variance numerically."""
        model = SymmetrizedModel(
            _OffsetLinearEnergyModel(1.0e8),
            max_angular_momentum_target=0,
            max_angular_momentum_grid=2,
        )
        system = _forward_test_system([[1.0, 2.0, 3.0]])

        result = model(
            [system],
            {"o3::variance::energy": ModelOutput(sample_kind="system")},
            None,
        )

        # <x^2> - <x>^2 = |r|^2 / 3, unchanged by the constant offset
        assert result["o3::variance::energy"].block().values.item() == pytest.approx(
            14.0 / 3.0,
            rel=1.0e-5,
        )

    def test_inconsistent_sample_labels_from_wrapped_model_are_rejected(self):
        """A wrapped model mislabelling its per-copy samples fails loudly."""
        model = SymmetrizedModel(
            _InconsistentSampleModel(),
            max_angular_momentum_target=0,
            max_angular_momentum_grid=2,
        )
        message = (
            "SymmetrizedModel expects every rotated copy to produce the same "
            "sample labels in the same order."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            model(
                [_forward_test_system([[1.0, 2.0, 3.0]])],
                {"energy": ModelOutput(sample_kind="atom")},
                None,
            )

    def test_rejects_invalid_requests_before_model_evaluation(self):
        """Invalid public requests should fail without running the source model."""
        base_model = _CountingLinearEnergyModel()
        model = SymmetrizedModel(base_model, max_angular_momentum_target=0)
        system = _forward_test_system([[1.0, 2.0, 3.0]])

        assert model([], {}, None) == {}
        message = "SymmetrizedModel requires at least one System"
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            model([], {"energy": ModelOutput(sample_kind="system")}, None)
        message = (
            "max_angular_momentum_character must be set to request "
            "character projections"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            model(
                [system],
                {"o3::character_projection::energy": ModelOutput(sample_kind="system")},
                None,
            )
        message = (
            "SymmetrizedModel does not support explicit gradients for output 'energy'"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            model(
                [system],
                {
                    "energy": ModelOutput(
                        sample_kind="system",
                        explicit_gradients=["positions"],
                    )
                },
                None,
            )
        message = (
            "all requests derived from 'energy' must use the same sample_kind; "
            "got 'system' and 'atom'"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            model(
                [system],
                {
                    "energy": ModelOutput(sample_kind="system"),
                    "o3::variance::energy": ModelOutput(sample_kind="atom"),
                },
                None,
            )
        assert base_model.call_count == 0

    def test_downcast_integration_buffers_warn_and_run(self):
        """Calling .float() on the module warns and loses accuracy, not correctness."""
        model = SymmetrizedModel(
            _LinearEnergyModel(),
            max_angular_momentum_target=0,
            max_angular_momentum_grid=2,
        ).float()
        system = _forward_test_system([[1.0, 2.0, 3.0]], dtype=torch.float32)

        with pytest.warns(UserWarning, match="downcast from float64"):
            result = model(
                [system],
                {"o3::variance::energy": ModelOutput(sample_kind="system")},
                None,
            )

        variance = result["o3::variance::energy"].block().values
        assert variance.dtype == torch.float32
        # <x^2> under O(3) rotations of r=(1, 2, 3): |r|^2 / 3 = 14/3
        assert variance.item() == pytest.approx(14.0 / 3.0, rel=1.0e-3)

    def test_rejects_a_model_that_omits_the_requested_output(self):
        """Fail loudly when the underlying model does not return a source."""
        model = SymmetrizedModel(
            _EmptyModel(),
            max_angular_momentum_target=0,
            max_angular_momentum_grid=2,
        )

        message = "underlying model did not return requested output 'energy'"
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            model(
                [_forward_test_system([[1.0, 2.0, 3.0]])],
                {"energy": ModelOutput(sample_kind="system")},
                None,
            )

    def test_rejects_a_non_finite_variance(self):
        """A NaN model response should fail the variance finiteness check."""
        model = SymmetrizedModel(
            _LinearEnergyModel(),
            max_angular_momentum_target=0,
            max_angular_momentum_grid=2,
        )

        message = "O(3) variance is not finite for block ((o3_lambda=0, o3_sigma=1))"
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            model(
                [_forward_test_system([[float("nan"), 2.0, 3.0]])],
                {"o3::variance::energy": ModelOutput(sample_kind="system")},
                None,
            )

    def test_is_scriptable_and_serializable(self, tmp_path):
        """The complete forward path should execute after scripting and reloading."""
        constructor_arguments = {
            "max_angular_momentum_target": 0,
            "max_angular_momentum_character": 1,
            "max_angular_momentum_grid": 2,
            "batch_size": 5,
        }
        eager = SymmetrizedModel(_LinearEnergyModel(), **constructor_arguments)
        scripted = torch.jit.script(
            SymmetrizedModel(_LinearEnergyModel(), **constructor_arguments)
        )
        path = tmp_path / "symmetrized-model.pt"
        torch.jit.save(scripted, str(path))
        loaded = torch.jit.load(str(path))
        system = _forward_test_system([[1.0, 2.0, 3.0]])
        outputs = {
            "energy": ModelOutput(sample_kind="system"),
            "o3::variance::energy": ModelOutput(sample_kind="system"),
            "o3::character_projection::energy": ModelOutput(sample_kind="system"),
        }

        expected = eager([system], outputs, None)
        actual = loaded([system], outputs, None)

        assert set(actual) == set(expected)
        for name in expected:
            mts.allclose_raise(actual[name], expected[name], rtol=0.0, atol=1.0e-12)

        # a SymmetrizedModel can also wrap a scripted + reloaded inner model
        inner_path = tmp_path / "inner-model.pt"
        torch.jit.save(torch.jit.script(_LinearEnergyModel()), str(inner_path))
        rewrapped = SymmetrizedModel(
            torch.jit.load(str(inner_path)),
            **constructor_arguments,
        )
        rewrapped_result = rewrapped([system], outputs, None)
        for name in expected:
            mts.allclose_raise(
                rewrapped_result[name],
                expected[name],
                rtol=0.0,
                atol=1.0e-12,
            )


class TestSymmetrizedModelWrap:
    """Test exported-model capabilities, dependencies, and execution."""

    @pytest.mark.parametrize("max_angular_momentum_character", [None, 1])
    def test_wrap_declares_capabilities(self, max_angular_momentum_character):
        """Wrapping declares averages and diagnostics with squared units."""
        source_outputs = {
            "energy": ModelOutput(
                unit="eV",
                sample_kind="system",
                explicit_gradients=["positions"],
            ),
            "mass": ModelOutput(unit="u", sample_kind="atom"),
            "mtt::pair": ModelOutput(sample_kind="atom_pair"),
        }
        base = AtomisticModel(
            _EmptyModel().eval(),
            ModelMetadata(name="wrapped source model"),
            ModelCapabilities(
                outputs=source_outputs,
                atomic_types=[1, 6, 8],
                interaction_range=4.5,
                length_unit="A",
                supported_devices=["cuda", "mps", "cpu"],
                dtype="float32",
            ),
        )

        wrapped = SymmetrizedModel.wrap(
            base,
            max_angular_momentum_target=0,
            max_angular_momentum_character=max_angular_momentum_character,
            max_angular_momentum_grid=2,
        )

        capabilities = wrapped.capabilities()
        assert wrapped.metadata().name == "wrapped source model"
        assert capabilities.atomic_types == [1, 6, 8]
        assert capabilities.interaction_range == 4.5
        assert capabilities.length_unit == "A"
        assert capabilities.supported_devices == ["cuda", "cpu"]

        expected_names = set(source_outputs)
        expected_names.update("o3::variance::" + name for name in source_outputs)
        if max_angular_momentum_character is not None:
            expected_names.update(
                "o3::character_projection::" + name for name in source_outputs
            )
        # "masses" is a compatibility alias added by AtomisticModel; it must not
        # become another declared source with its own diagnostics
        assert set(capabilities.outputs) == expected_names | {"masses"}
        assert "o3::variance::masses" not in capabilities.outputs
        assert "o3::character_projection::masses" not in capabilities.outputs

        for name, source_output in source_outputs.items():
            squared_unit = (
                "" if source_output.unit == "" else f"({source_output.unit})^2"
            )
            assert capabilities.outputs["o3::variance::" + name].unit == squared_unit
            character_name = "o3::character_projection::" + name
            if max_angular_momentum_character is None:
                assert character_name not in capabilities.outputs
            else:
                assert capabilities.outputs[character_name].unit == squared_unit

    def test_guesses_limits_from_standard_quantities(self):
        """Both limits default to what the standard quantities require."""

        class _VelocityInputModel(_EmptyModel):
            def requested_inputs(self) -> Dict[str, ModelOutput]:
                # the custom input is skipped by the guess as well
                return {
                    "velocity": ModelOutput(sample_kind="atom"),
                    "mtt::field": ModelOutput(sample_kind="atom"),
                }

        def guessed_target(outputs):
            base = AtomisticModel(
                _VelocityInputModel().eval(),
                ModelMetadata(),
                ModelCapabilities(
                    outputs=outputs,
                    atomic_types=[1],
                    interaction_range=0.0,
                    length_unit="A",
                    supported_devices=["cpu"],
                    dtype="float64",
                ),
            )
            wrapped = SymmetrizedModel.wrap(base, max_angular_momentum_grid=2)
            # velocity is a Cartesian vector
            assert wrapped.module.max_angular_momentum_input == 1
            return wrapped.module.max_angular_momentum_target

        energy = ModelOutput(unit="eV", sample_kind="system")
        force = ModelOutput(unit="eV/A", sample_kind="atom")
        stress = ModelOutput(unit="eV/A^3", sample_kind="system")

        assert guessed_target({"energy": energy}) == 0
        assert guessed_target({"feature": ModelOutput(sample_kind="atom")}) == 0
        assert guessed_target({"energy": energy, "non_conservative_force": force}) == 1
        assert guessed_target({"non_conservative_stress": stress}) == 2
        # a custom output is skipped, the standard ones still set the limit
        custom = ModelOutput(sample_kind="system")
        assert guessed_target({"energy": energy, "mtt::custom": custom}) == 0

    def test_rejects_guessing_a_limit_without_standard_outputs(self):
        """Only non-standard outputs leave nothing to guess the limit from."""
        base = AtomisticModel(
            _EmptyModel().eval(),
            ModelMetadata(),
            ModelCapabilities(
                outputs={
                    "mtt::custom": ModelOutput(sample_kind="system"),
                    "mtt::other": ModelOutput(sample_kind="system"),
                },
                atomic_types=[1],
                interaction_range=0.0,
                length_unit="A",
                supported_devices=["cpu"],
                dtype="float64",
            ),
        )

        message = (
            "no standard quantities were found among the outputs "
            "['mtt::custom', 'mtt::other'], please set max_angular_momentum_target "
            "explicitly"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            SymmetrizedModel.wrap(base)

    @pytest.mark.parametrize(
        "source_name",
        [
            "o3::variance::mtt::source",
            "o3::character_projection::mtt::source",
        ],
    )
    def test_rejects_reserved_source_names(self, source_name):
        """Reject source names that look like wrapper-generated diagnostics."""
        base = AtomisticModel(
            _EmptyModel().eval(),
            ModelMetadata(),
            ModelCapabilities(
                outputs={source_name: ModelOutput(sample_kind="system")},
                atomic_types=[1],
                interaction_range=0.0,
                length_unit="A",
                supported_devices=["cpu"],
                dtype="float64",
            ),
        )

        message = (
            f"the wrapped model output '{source_name}' uses a prefix reserved "
            "by SymmetrizedModel"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            SymmetrizedModel.wrap(base, max_angular_momentum_target=0)

    def test_rejects_models_without_a_supported_device(self):
        """Reject models whose declared devices contain neither CPU nor CUDA."""
        base = AtomisticModel(
            _EmptyModel().eval(),
            ModelMetadata(),
            ModelCapabilities(
                outputs={"mtt::value": ModelOutput(sample_kind="system")},
                atomic_types=[1],
                interaction_range=0.0,
                length_unit="A",
                supported_devices=["mps"],
                dtype="float64",
            ),
        )

        message = (
            "SymmetrizedModel supports CPU and CUDA execution, but the "
            "wrapped model declares ['mps']"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            SymmetrizedModel.wrap(base, max_angular_momentum_target=0)

    def test_preserves_requirements_and_runs_after_save_load(self, tmp_path):
        """Preserve model requirements through wrapping, saving, and reloading."""
        metadata = ModelMetadata(name="model with requirements")
        base = AtomisticModel(
            _LinearModelWithRequirements().eval(),
            metadata,
            ModelCapabilities(
                outputs={
                    "mtt::linear": ModelOutput(
                        unit="eV",
                        sample_kind="system",
                        description="First Cartesian coordinate.",
                    )
                },
                atomic_types=[1],
                interaction_range=2.5,
                length_unit="A",
                supported_devices=["cpu"],
                dtype="float32",
            ),
        )
        base_path = tmp_path / "base-model.pt"
        base.save(base_path)
        loaded_base = load_atomistic_model(base_path)
        base_requestors = set(loaded_base.requested_neighbor_lists()[0].requestors())

        wrapped = SymmetrizedModel.wrap(
            loaded_base,
            max_angular_momentum_target=0,
            # 'mtt::linear' and 'mtt::field' are not standard quantities, so
            # both limits have to be given explicitly
            max_angular_momentum_input=0,
            max_angular_momentum_character=1,
            max_angular_momentum_grid=2,
            batch_size=5,
        )
        assert (
            set(loaded_base.requested_neighbor_lists()[0].requestors())
            == base_requestors
        )

        wrapped_path = tmp_path / "symmetrized-model.pt"
        wrapped.save(wrapped_path)
        loaded = load_atomistic_model(wrapped_path)

        requested_inputs = loaded.requested_inputs(use_new_names=True)
        assert set(requested_inputs) == {"mtt::field"}
        assert requested_inputs["mtt::field"].unit == "eV"
        assert requested_inputs["mtt::field"].sample_kind == "atom"
        assert (
            requested_inputs["mtt::field"].description
            == "Cartesian field used by the model."
        )

        requested_neighbor_lists = loaded.requested_neighbor_lists()
        assert len(requested_neighbor_lists) == 1
        neighbor_options = requested_neighbor_lists[0]
        assert neighbor_options.cutoff == 2.5
        assert neighbor_options.full_list is False
        assert neighbor_options.strict is True
        assert base_requestors.issubset(set(neighbor_options.requestors()))

        system = _system_with_linear_model_requirements(
            neighbor_options,
            torch.device("cpu"),
        )

        requested_outputs = {
            "mtt::linear": ModelOutput(
                unit="meV",
                sample_kind="system",
            ),
            "o3::variance::mtt::linear": ModelOutput(
                unit="(meV)^2",
                sample_kind="system",
            ),
            "o3::character_projection::mtt::linear": ModelOutput(
                unit="(meV)^2",
                sample_kind="system",
            ),
        }
        evaluation_options = ModelEvaluationOptions(
            length_unit="A",
            outputs=requested_outputs,
        )
        eager = wrapped([system], evaluation_options, check_consistency=True)
        reloaded = loaded([system], evaluation_options, check_consistency=True)

        assert set(reloaded) == set(requested_outputs)
        for name in eager:
            mts.allclose_raise(
                reloaded[name],
                eager[name],
                rtol=0.0,
                atol=0.0,
            )
            for block in reloaded[name].blocks():
                assert block.values.dtype == torch.float32

        expected_variance = torch.tensor(
            [[14.0 / 3.0 * 1.0e6]],
            dtype=torch.float32,
        )
        assert torch.allclose(
            reloaded["o3::variance::mtt::linear"].block().values,
            expected_variance,
            rtol=2.0e-5,
            atol=1.0,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_saved_wrapper_runs_on_cuda(self, tmp_path):
        """Match CPU results after moving a saved float32 wrapper to CUDA.

        The wrapper's model dtype is float32, while its integration buffers stay
        float64 throughout; nothing here downcasts the module itself.
        """
        base = AtomisticModel(
            _LinearModelWithRequirements().eval(),
            ModelMetadata(name="CUDA source model"),
            ModelCapabilities(
                outputs={
                    "energy": ModelOutput(
                        unit="eV",
                        sample_kind="system",
                    )
                },
                atomic_types=[1],
                interaction_range=2.5,
                length_unit="A",
                supported_devices=["cpu", "cuda"],
                dtype="float32",
            ),
        )
        wrapped = SymmetrizedModel.wrap(
            base,
            max_angular_momentum_target=0,
            max_angular_momentum_input=0,
            max_angular_momentum_character=1,
            max_angular_momentum_grid=2,
            batch_size=5,
        )
        path = tmp_path / "cuda-symmetrized-model.pt"
        wrapped.save(path)

        cpu_model = load_atomistic_model(path)
        cuda_device = torch.device("cuda", torch.cuda.current_device())
        cuda_model = load_atomistic_model(path).to(device=cuda_device)
        neighbor_options = cpu_model.requested_neighbor_lists()[0]
        cpu_system = _system_with_linear_model_requirements(
            neighbor_options,
            torch.device("cpu"),
        )
        cuda_system = cpu_system.to(device=cuda_device)

        cpu_module = SymmetrizedModel(
            _LinearEnergyModel(), max_angular_momentum_target=0
        )
        message = "SymmetrizedModel and input Systems must use the same device"
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            cpu_module(
                [cuda_system],
                {"energy": ModelOutput(sample_kind="system")},
                None,
            )

        requested_outputs = {
            "energy": ModelOutput(
                unit="meV",
                sample_kind="system",
            ),
            "o3::variance::energy": ModelOutput(
                unit="(meV)^2",
                sample_kind="system",
            ),
            "o3::character_projection::energy": ModelOutput(
                unit="(meV)^2",
                sample_kind="system",
            ),
        }
        evaluation_options = ModelEvaluationOptions(
            length_unit="A",
            outputs=requested_outputs,
        )
        with torch.inference_mode():
            expected = cpu_model(
                [cpu_system],
                evaluation_options,
                check_consistency=True,
            )
            actual = cuda_model(
                [cuda_system],
                evaluation_options,
                check_consistency=True,
            )

        assert set(actual) == set(requested_outputs)
        assert actual["energy"].block().values.device.type == "cuda"
        for name, tensor in actual.items():
            mts.allclose_raise(
                tensor.to(device="cpu"),
                expected[name],
                rtol=2.0e-5,
                atol=2.0e-5,
            )


def test_selected_atoms_system_column_found_by_name():
    """The rotated-copy index goes into the "system" column wherever it sits."""
    selection = Labels(["atom", "system"], torch.tensor([[3, 0], [5, 0]]))
    rotated = map_selected_atoms_to_rotated_copies(selection, 0, 2)
    assert rotated.names == ["atom", "system"]
    assert rotated.values[:, 0].tolist() == [3, 5, 3, 5]
    assert rotated.values[:, 1].tolist() == [0, 0, 1, 1]


def test_cartesian_vectors_to_spherical():
    """Map Cartesian components to the real spherical l=1 ordering."""
    values = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]],
        dtype=torch.float64,
    )

    result = _cartesian_vectors_to_spherical(values, component_axis=1)

    assert torch.equal(
        result,
        torch.tensor(
            [[[2.0, 20.0], [3.0, 30.0], [1.0, 10.0]]],
            dtype=torch.float64,
        ),
    )


@pytest.mark.parametrize("inversion", [1.0, -1.0])
def test_cartesian_vectors_to_spherical_commutes_with_o3(inversion):
    """Converting before or after an O(3) transformation should give the same result."""
    proper_rotation = torch.tensor(
        [
            [-2.0 / 3.0, 2.0 / 15.0, 11.0 / 15.0],
            [2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0],
            [1.0 / 3.0, 14.0 / 15.0, 2.0 / 15.0],
        ],
        dtype=torch.float64,
    )
    transformation = O3Transformation(
        inversion * proper_rotation,
        max_angular_momentum=1,
    )
    cartesian = torch.tensor(
        [[1.2, -0.7, 2.3], [-0.4, 1.1, 0.8]],
        dtype=torch.float64,
    )

    transformed_cartesian = _cartesian_vectors_to_spherical(
        transformation.transform_cartesian(cartesian),
        component_axis=1,
    )
    transformed_spherical = transformation.transform_spherical(
        _cartesian_vectors_to_spherical(cartesian, component_axis=1),
        ell=1,
        sigma=1,
    )

    assert torch.allclose(
        transformed_cartesian,
        transformed_spherical,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_symmetric_matrices_to_spherical_known_components():
    """Known matrices map as expected and the symmetric norm is preserved."""
    matrices = torch.zeros((3, 3, 3, 1), dtype=torch.float64)
    matrices[0, :, :, 0] = torch.eye(3, dtype=torch.float64)
    matrices[1, 0, 0, 0] = 1.0
    matrices[1, 1, 1, 0] = -1.0
    matrices[2, 0, 1, 0] = 2.0
    matrices[2, 1, 0, 0] = -2.0

    with pytest.warns(UserWarning, match="materially antisymmetric"):
        l0, l2 = _symmetric_matrices_to_spherical(matrices)

    expected_l0 = torch.zeros((3, 1, 1), dtype=torch.float64)
    expected_l0[0, 0, 0] = 3.0**0.5
    expected_l2 = torch.zeros((3, 5, 1), dtype=torch.float64)
    expected_l2[1, 4, 0] = 2.0**0.5
    assert torch.allclose(l0, expected_l0, rtol=0.0, atol=1.0e-12)
    assert torch.allclose(l2, expected_l2, rtol=0.0, atol=1.0e-12)

    generator = torch.Generator().manual_seed(1234)
    random_matrices = torch.randn(
        (4, 3, 3, 2),
        dtype=torch.float64,
        generator=generator,
    )
    symmetric = 0.5 * (random_matrices + random_matrices.transpose(1, 2))

    with pytest.warns(UserWarning, match="materially antisymmetric"):
        l0, l2 = _symmetric_matrices_to_spherical(random_matrices)

    spherical_norm_squared = l0.square().sum(dim=1) + l2.square().sum(dim=1)
    cartesian_norm_squared = symmetric.square().sum(dim=(1, 2))
    assert torch.allclose(
        spherical_norm_squared,
        cartesian_norm_squared,
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("inversion", [1.0, -1.0])
def test_symmetric_matrices_to_spherical_commutes_with_o3(inversion):
    """Cartesian and spherical transformations should give the same components."""
    proper_rotation = torch.tensor(
        [
            [-2.0 / 3.0, 2.0 / 15.0, 11.0 / 15.0],
            [2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0],
            [1.0 / 3.0, 14.0 / 15.0, 2.0 / 15.0],
        ],
        dtype=torch.float64,
    )
    transformation = O3Transformation(
        inversion * proper_rotation,
        max_angular_momentum=2,
    )
    matrices = torch.tensor(
        [
            [[1.2, -0.7, 2.3], [-0.7, 1.1, 0.8], [2.3, 0.8, -0.4]],
            [[-0.2, 1.4, 0.5], [1.4, 0.9, -1.1], [0.5, -1.1, 2.0]],
        ],
        dtype=torch.float64,
    ).unsqueeze(-1)

    matrix = transformation.matrix
    transformed_matrices = torch.einsum(
        "ia,sabp,jb->sijp",
        matrix,
        matrices,
        matrix,
    )
    transformed_l0, transformed_l2 = _symmetric_matrices_to_spherical(
        transformed_matrices
    )
    l0, l2 = _symmetric_matrices_to_spherical(matrices)

    expected_l0 = transformation.transform_spherical(
        l0[..., 0], ell=0, sigma=1
    ).unsqueeze(-1)
    expected_l2 = transformation.transform_spherical(
        l2[..., 0], ell=2, sigma=1
    ).unsqueeze(-1)
    assert torch.allclose(transformed_l0, expected_l0, rtol=0.0, atol=1.0e-12)
    assert torch.allclose(transformed_l2, expected_l2, rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize(
    "source_name",
    [
        "energy",
        "energy/pbe",
        "energy_ensemble/member",
        "energy_uncertainty/direct",
        "charge",
    ],
)
def test_decompose_quantity_scalar_quantities(source_name):
    """Scalar quantities and their variants become one l=0 spherical block."""
    values = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    tensor = _tensor_map_with_components(values, [])
    tensor.set_info("unit", "eV")

    result = decompose_quantity(source_name, tensor)

    assert result.keys.names == ["o3_lambda", "o3_sigma"]
    assert result.keys.values.tolist() == [[0, 1]]
    assert torch.equal(result.block().values, values.unsqueeze(1))
    assert result.block().samples == tensor.block().samples
    assert result.block().components == [_o3_mu_labels(0, values.device)]
    assert result.block().properties == tensor.block().properties
    assert result.info() == tensor.info()


@pytest.mark.parametrize(
    "source_name",
    [
        "non_conservative_force/direct",
        "velocity",
    ],
)
def test_decompose_quantity_cartesian_vectors_preserve_autograd(source_name):
    """Cartesian vectors should become l=1 and preserve implicit autograd."""
    values = torch.tensor(
        [[[1.0], [2.0], [3.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    tensor = _tensor_map_with_components(values, ["xyz"])

    result = decompose_quantity(source_name, tensor)

    assert result.keys.names == ["o3_lambda", "o3_sigma"]
    assert result.keys.values.tolist() == [[1, 1]]
    assert result.block().components == [_o3_mu_labels(1, values.device)]
    assert torch.equal(
        result.block().values,
        torch.tensor([[[2.0], [3.0], [1.0]]], dtype=torch.float64),
    )

    result.block().values.sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))


def test_decompose_quantity_non_conservative_stress_combines_irreps():
    """Stress should return l=0 and l=2 blocks, warning about discarded skew."""
    values = torch.zeros((2, 3, 3, 1), dtype=torch.float64)
    values[0, :, :, 0] = torch.eye(3, dtype=torch.float64)
    values[1, 0, 1, 0] = 2.0
    values[1, 1, 0, 0] = -2.0
    tensor = _tensor_map_with_components(values, ["xyz_1", "xyz_2"])

    with pytest.warns(UserWarning, match="materially antisymmetric"):
        result = decompose_quantity("non_conservative_stress/direct", tensor)

    assert result.keys.names == ["o3_lambda", "o3_sigma"]
    assert result.keys.values.tolist() == [[0, 1], [2, 1]]
    block_l0 = result.block({"o3_lambda": 0, "o3_sigma": 1})
    block_l2 = result.block({"o3_lambda": 2, "o3_sigma": 1})
    assert block_l0.components == [_o3_mu_labels(0, values.device)]
    assert block_l2.components == [_o3_mu_labels(2, values.device)]
    assert torch.allclose(
        block_l0.values,
        torch.tensor([[[3.0**0.5]], [[0.0]]], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert torch.equal(block_l2.values, torch.zeros((2, 5, 1), dtype=torch.float64))
    assert block_l0.samples == tensor.block().samples
    assert block_l2.samples == tensor.block().samples
    assert block_l0.properties == tensor.block().properties
    assert block_l2.properties == tensor.block().properties


def test_decompose_quantity_does_not_infer_custom_cartesian_semantics():
    """A generic 3x3 output should pass through unchanged."""
    tensor = _tensor_map_with_components(
        torch.rand((1, 3, 3, 1), dtype=torch.float64),
        ["xyz_1", "xyz_2"],
    )

    result = decompose_quantity("mtt::custom", tensor)

    mts.equal_raise(result, tensor)


def test_forward_rejects_outputs_with_attached_gradients():
    """The wrapper should not silently discard explicit TensorBlock gradients."""

    class _AttachedGradientModel(torch.nn.Module):
        def forward(
            self,
            systems: List[System],
            outputs: Dict[str, ModelOutput],
            selected_atoms: Optional[Labels],
        ) -> Dict[str, TensorMap]:
            properties = Labels.range("property", 1)
            block = TensorBlock(
                values=torch.ones((len(systems), 1), dtype=torch.float64),
                samples=Labels.range("system", len(systems)),
                components=[],
                properties=properties,
            )
            block.add_gradient(
                "positions",
                TensorBlock(
                    values=torch.ones((1, 3, 1), dtype=torch.float64),
                    samples=Labels("sample", torch.tensor([[0]], dtype=torch.int64)),
                    components=[Labels.range("xyz", 3)],
                    properties=properties,
                ),
            )
            return {
                "energy": TensorMap(
                    Labels("_", torch.tensor([[0]], dtype=torch.int64)),
                    [block],
                )
            }

    model = SymmetrizedModel(
        _AttachedGradientModel(),
        max_angular_momentum_target=0,
        max_angular_momentum_grid=2,
    )

    message = (
        "underlying output 'energy' contains unsupported explicit gradient 'positions'"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        model(
            [_forward_test_system([[1.0, 2.0, 3.0]])],
            {"energy": ModelOutput(sample_kind="system")},
            None,
        )
