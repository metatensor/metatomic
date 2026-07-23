import inspect
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
    System,
    load_atomistic_model,
)
from metatomic.torch.o3 import O3Transformation, transform_system
from metatomic.torch.symmetrized_model import (
    SymmetrizedModel,
    get_rotation_quadrature,
)
from metatomic.torch.symmetrized_model._decompose import (
    _add_o3_irrep_to_keys,
    _cartesian_vectors_to_spherical,
    _decompose_output,
    _o3_mu_labels,
    _symmetric_matrices_to_spherical,
)
from metatomic.torch.symmetrized_model._model import (
    _clamp_roundoff_negative_diagnostic,
    _component_norm_squared,
    _group_output_requests,
    _join_per_system_tensormaps,
    _mean_variance_over_components,
    _parse_output_request,
    _reduce_weighted_centered_batch,
    _transform_system_batch,
    _transform_system_geometry_batch,
    _variance_from_centered_moments,
)
from metatomic.torch.symmetrized_model._projections import (
    _character_projection_coefficients_from_rotation_batch,
    _character_projections_from_proper_and_improper_coefficients,
)
from metatomic.torch.symmetrized_model._quadrature import (
    _choose_quadrature,
    _rotations_from_euler_angles,
    get_euler_angles_quadrature,
)
from metatomic.torch.symmetrized_model._utils import (
    _group_samples_by_rotated_copy,
    _map_selected_atoms_to_rotated_copies,
    _restore_input_system_to_samples,
)
from metatomic.torch.symmetrized_model._wigner_storage import (
    _build_packed_wigner_matrices,
    _wigner_matrices_for_lambda,
)


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
    """
    Return one analytic polynomial response in every O(3) sector through
    ``lambda=3``.

    The homogeneous harmonic polynomials ``1``, ``x``, ``x*y``, and ``x*y*z``
    transform purely in the ``lambda=0``, ``1``, ``2``, and ``3`` sectors,
    respectively. These responses have ``sigma=+1``. Multiplying each polynomial
    by the determinant of the transformed Cartesian frame changes only its
    inversion parity, producing the corresponding ``sigma=-1`` response.

    The eight responses are returned as properties of one scalar TensorMap block,
    labeled by ``source_lambda`` and ``source_sigma``.
    """

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


def _system_with_neighbor_lists(dtype: torch.dtype) -> System:
    """Create a test system with populated and empty neighbor lists."""
    positions = torch.tensor(
        [[0.2, -0.1, 0.3], [1.1, 0.7, -0.4], [-0.3, 0.6, 1.2]],
        dtype=dtype,
    )
    cell = torch.tensor(
        [[2.5, 0.1, 0.0], [0.0, 2.2, 0.2], [0.1, 0.0, 2.7]],
        dtype=dtype,
    )
    system = System(
        types=torch.tensor([6, 1, 8]),
        positions=positions,
        cell=cell,
        pbc=torch.tensor([True, True, True]),
    )

    samples = Labels(
        [
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        torch.tensor([[0, 1, 0, 0, 0], [1, 2, 1, 0, 0]]),
    )
    components = [Labels.range("xyz", 3)]
    properties = Labels.range("distance", 1)
    system.add_neighbor_list(
        NeighborListOptions(3.0, False, True, "populated"),
        TensorBlock(
            values=torch.stack(
                [
                    positions[1] - positions[0],
                    positions[2] - positions[1] + cell[0],
                ]
            ).unsqueeze(-1),
            samples=samples,
            components=components,
            properties=properties,
        ),
    )
    system.add_neighbor_list(
        NeighborListOptions(1.0, True, False, "empty"),
        TensorBlock(
            values=torch.empty((0, 3, 1), dtype=dtype),
            samples=Labels(
                list(samples.names),
                torch.empty((0, len(samples.names)), dtype=torch.int64),
            ),
            components=components,
            properties=properties,
        ),
    )
    return system


class TestSystemGeometryBatch:
    """Test batched O(3) transformation of System geometry."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("n_matrices", [1, 3])
    def test_matches_individual_o3_transformations(self, dtype, n_matrices):
        """Batched geometry should match one transformation at a time."""
        proper = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        )
        if n_matrices == 1:
            matrices = proper.unsqueeze(0)
        else:
            matrices = torch.stack([torch.eye(3, dtype=dtype), proper, -proper])
        system = _system_with_neighbor_lists(dtype)

        transformed = _transform_system_geometry_batch(system, matrices)

        assert len(transformed) == len(matrices)
        for matrix, actual in zip(matrices, transformed, strict=True):
            expected = transform_system(
                system,
                O3Transformation(matrix, max_angular_momentum=0),
            )
            assert torch.equal(actual.positions, expected.positions)
            assert torch.equal(actual.cell, expected.cell)
            assert torch.equal(actual.types, expected.types)
            assert torch.equal(actual.pbc, expected.pbc)
            assert actual.known_neighbor_lists() == expected.known_neighbor_lists()
            for options in expected.known_neighbor_lists():
                actual_neighbors = actual.get_neighbor_list(options)
                expected_neighbors = expected.get_neighbor_list(options)
                assert torch.equal(actual_neighbors.values, expected_neighbors.values)
                assert actual_neighbors.samples == expected_neighbors.samples
                assert actual_neighbors.components == expected_neighbors.components
                assert actual_neighbors.properties == expected_neighbors.properties

    def test_preserves_neighbor_autograd(self):
        """Rotated neighbor vectors should differentiate through positions and cell."""
        positions = torch.tensor(
            [[0.2, -0.1, 0.3], [1.1, 0.7, -0.4]],
            dtype=torch.float64,
            requires_grad=True,
        )
        cell = torch.tensor(
            [[2.5, 0.1, 0.0], [0.0, 2.2, 0.2], [0.1, 0.0, 2.7]],
            dtype=torch.float64,
            requires_grad=True,
        )
        system = System(
            types=torch.tensor([6, 1]),
            positions=positions,
            cell=cell,
            pbc=torch.tensor([True, True, True]),
        )
        cell_shift = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)
        neighbor_vector = positions[1] - positions[0] + cell_shift @ cell
        options = NeighborListOptions(4.0, False, True)
        system.add_neighbor_list(
            options,
            TensorBlock(
                values=neighbor_vector.reshape(1, 3, 1),
                samples=Labels(
                    [
                        "first_atom",
                        "second_atom",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                    ],
                    torch.tensor([[0, 1, 1, -1, 0]]),
                ),
                components=[Labels.range("xyz", 3)],
                properties=Labels.range("distance", 1),
            ),
        )
        proper = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        matrices = torch.stack([proper, -proper])

        transformed = _transform_system_geometry_batch(system, matrices)
        loss = sum(
            transformed_system.get_neighbor_list(options).values.square().sum()
            for transformed_system in transformed
        )
        position_gradient, cell_gradient = torch.autograd.grad(
            loss,
            (positions, cell),
        )

        vector_gradient = 2 * len(matrices) * neighbor_vector.detach()
        assert torch.allclose(
            position_gradient,
            torch.stack([-vector_gradient, vector_gradient]),
        )
        assert torch.allclose(
            cell_gradient,
            torch.outer(cell_shift, vector_gradient),
        )

    def test_rejects_invalid_matrix_batches(self):
        """Matrix batches should have a non-empty shape and match the System."""
        system = _system_with_neighbor_lists(torch.float64)
        invalid_shapes = [(3, 3), (0, 3, 3), (2, 2, 3), (2, 3, 2)]
        for shape in invalid_shapes:
            with pytest.raises(ValueError, match="shape \\(N, 3, 3\\)"):
                _transform_system_geometry_batch(
                    system,
                    torch.empty(shape, dtype=torch.float64),
                )

        with pytest.raises(ValueError, match="same dtype and device"):
            _transform_system_geometry_batch(
                system,
                torch.eye(3, dtype=torch.float32).unsqueeze(0),
            )

    def test_is_scriptable(self):
        """The batched geometry transformation should compile and execute."""
        scripted = torch.jit.script(_transform_system_geometry_batch)
        system = _system_with_neighbor_lists(torch.float64)
        transformed = scripted(
            system,
            torch.eye(3, dtype=torch.float64).unsqueeze(0),
        )

        assert len(transformed) == 1
        assert torch.equal(transformed[0].positions, system.positions)
        assert torch.equal(transformed[0].cell, system.cell)


class TestSystemBatch:
    """Test batched O(3) transformation of complete Systems."""

    @pytest.mark.parametrize("is_improper", [False, True])
    def test_transforms_spherical_custom_data(self, is_improper):
        """Every transformed System should contain the corresponding custom data."""
        proper_matrices = torch.tensor(
            [
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [
                    [-2.0 / 3.0, 2.0 / 15.0, 11.0 / 15.0],
                    [2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0],
                    [1.0 / 3.0, 14.0 / 15.0, 2.0 / 15.0],
                ],
            ],
            dtype=torch.float64,
        )
        matrices = -proper_matrices if is_improper else proper_matrices
        packed_wigner = _build_packed_wigner_matrices(
            proper_matrices,
            max_o3_lambda=1,
        )
        wigner_matrices = [
            _wigner_matrices_for_lambda(
                packed_wigner,
                n_matrices=len(matrices),
                o3_lambda=o3_lambda,
            )
            for o3_lambda in range(2)
        ]

        system = System(
            types=torch.tensor([6, 8]),
            positions=torch.tensor(
                [[0.2, -0.1, 0.3], [1.1, 0.7, -0.4]],
                dtype=torch.float64,
            ),
            cell=torch.eye(3, dtype=torch.float64) * 4.0,
            pbc=torch.tensor([True, True, True]),
        )
        values = torch.tensor(
            [[[1.0], [2.0], [3.0]], [[-0.5], [1.5], [0.25]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        system.add_data(
            "mtt::field",
            TensorMap(
                Labels(
                    ["o3_lambda", "o3_sigma"],
                    torch.tensor([[1, 1]]),
                ),
                [
                    TensorBlock(
                        values=values,
                        samples=Labels.range("atom", 2),
                        components=[_o3_mu_labels(1, values.device)],
                        properties=Labels.range("property", 1),
                    )
                ],
            ),
        )

        transformed = torch.jit.script(_transform_system_batch)(
            system,
            matrices,
            wigner_matrices,
            max_o3_lambda_input=1,
            is_improper=is_improper,
        )

        assert len(transformed) == len(matrices)
        for matrix, transformed_system in zip(matrices, transformed, strict=True):
            expected_system = transform_system(
                system,
                O3Transformation(matrix, max_angular_momentum=1),
            )
            assert "mtt::field" in transformed_system.known_data()
            mts.allclose_raise(
                transformed_system.get_data("mtt::field"),
                expected_system.get_data("mtt::field"),
                rtol=0.0,
                atol=1.0e-12,
            )

        loss = sum(
            transformed_system.get_data("mtt::field").block().values.square().sum()
            for transformed_system in transformed
        )
        gradient = torch.autograd.grad(loss, values)[0]
        assert torch.allclose(
            gradient,
            2 * len(matrices) * values,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_input_limit_distinguishes_spherical_from_cartesian(self):
        """A zero spherical-rank limit should still allow Cartesian custom data."""
        matrix = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ).unsqueeze(0)
        packed_wigner = _build_packed_wigner_matrices(
            matrix,
            max_o3_lambda=0,
        )
        wigner_matrices = [
            _wigner_matrices_for_lambda(
                packed_wigner,
                n_matrices=1,
                o3_lambda=0,
            )
        ]
        system = System(
            types=torch.tensor([6]),
            positions=torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64),
            cell=torch.eye(3, dtype=torch.float64) * 4.0,
            pbc=torch.tensor([True, True, True]),
        )
        cartesian = TensorMap(
            Labels("_", torch.tensor([[0]])),
            [
                TensorBlock(
                    values=torch.tensor(
                        [[[1.0], [2.0], [3.0]]],
                        dtype=torch.float64,
                    ),
                    samples=Labels.range("atom", 1),
                    components=[Labels.range("xyz", 3)],
                    properties=Labels.range("property", 1),
                )
            ],
        )
        system.add_data("mtt::field", cartesian)
        scripted_transform = torch.jit.script(_transform_system_batch)

        transformed = scripted_transform(
            system,
            matrix,
            wigner_matrices,
            max_o3_lambda_input=0,
            is_improper=False,
        )
        expected = transform_system(
            system,
            O3Transformation(matrix[0], max_angular_momentum=0),
        )
        mts.allclose_raise(
            transformed[0].get_data("mtt::field"),
            expected.get_data("mtt::field"),
            rtol=0.0,
            atol=1.0e-12,
        )

        spherical_system = System(
            types=system.types,
            positions=system.positions,
            cell=system.cell,
            pbc=system.pbc,
        )
        spherical_system.add_data(
            "mtt::field",
            TensorMap(
                Labels(
                    ["o3_lambda", "o3_sigma"],
                    torch.tensor([[1, 1]]),
                ),
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
        with pytest.raises(
            torch.jit.Error,
            match=(
                "custom input 'mtt::field' contains o3_lambda=1, exceeding "
                "max_o3_lambda_input=0"
            ),
        ):
            scripted_transform(
                spherical_system,
                matrix,
                wigner_matrices,
                max_o3_lambda_input=0,
                is_improper=False,
            )


class TestCharacterProjections:
    """Test construction of character projections from rotated model responses."""

    @pytest.mark.parametrize("n_samples", [0, 2])
    def test_batch_coefficients_match_rotation_by_rotation_sum(self, n_samples):
        """Batching should match summing the weighted rotations individually."""
        torch.manual_seed(7)
        n_rotations = 4
        dimension = 3
        values = torch.randn(
            (n_rotations, n_samples, 2, 3),
            dtype=torch.float64,
        )
        weights = torch.tensor(
            [0.50, -0.25, 0.30, 0.45],
            dtype=torch.float32,
        )
        inverse_wigner_matrices = torch.randn(
            (n_rotations, dimension, dimension),
            dtype=torch.float32,
        )

        coefficients = _character_projection_coefficients_from_rotation_batch(
            values,
            weights,
            inverse_wigner_matrices,
        )

        expected = torch.zeros(
            (n_samples, dimension, dimension, 2, 3),
            dtype=torch.float64,
        )
        for rotation in range(n_rotations):
            expected += (
                weights[rotation].to(torch.float64)
                * inverse_wigner_matrices[rotation]
                .to(torch.float64)
                .reshape(1, dimension, dimension, 1, 1)
                * values[rotation].reshape(n_samples, 1, 1, 2, 3)
            )

        assert torch.allclose(coefficients, expected, rtol=0.0, atol=1e-12)

    @pytest.mark.parametrize("chi_lambda", [0, 1, 2])
    def test_factorization_matches_all_rotation_pairs(self, chi_lambda):
        """The factorization should match summing every pair of rotations."""
        torch.manual_seed(11 + chi_lambda)
        n_rotations = 4
        dimension = 2 * chi_lambda + 1
        proper_values = torch.randn(
            (n_rotations, 2, 2, 1),
            dtype=torch.float64,
            requires_grad=True,
        )
        improper_values = torch.randn(
            (n_rotations, 2, 2, 1),
            dtype=torch.float64,
            requires_grad=True,
        )
        weights = torch.tensor(
            [0.50, -0.25, 0.30, 0.45],
            dtype=torch.float64,
        )
        inverse_wigner_matrices = torch.randn(
            (n_rotations, dimension, dimension),
            dtype=torch.float64,
        )
        proper_coefficients = _character_projection_coefficients_from_rotation_batch(
            proper_values,
            weights,
            inverse_wigner_matrices,
        )
        improper_coefficients = _character_projection_coefficients_from_rotation_batch(
            improper_values,
            weights,
            inverse_wigner_matrices,
        )

        sigma_plus, sigma_minus = (
            _character_projections_from_proper_and_improper_coefficients(
                proper_coefficients,
                improper_coefficients,
                chi_lambda,
            )
        )

        expected = []
        for chi_sigma in (1, -1):
            combined_values = proper_values + (
                chi_sigma * (-1) ** chi_lambda * improper_values
            )
            direct_sum = torch.zeros_like(combined_values[0])
            for first_rotation in range(n_rotations):
                for second_rotation in range(n_rotations):
                    character = torch.sum(
                        inverse_wigner_matrices[first_rotation]
                        * inverse_wigner_matrices[second_rotation]
                    )
                    direct_sum += (
                        float(dimension)
                        / 4.0
                        * weights[first_rotation]
                        * weights[second_rotation]
                        * character
                        * combined_values[first_rotation]
                        * combined_values[second_rotation]
                    )
            expected.append(direct_sum)

        assert torch.allclose(sigma_plus, expected[0], rtol=0.0, atol=1e-12)
        assert torch.allclose(sigma_minus, expected[1], rtol=0.0, atol=1e-12)
        assert sigma_plus.shape == proper_values.shape[1:]
        assert sigma_minus.shape == improper_values.shape[1:]
        assert torch.all(sigma_plus >= 0)
        assert torch.all(sigma_minus >= 0)

        (sigma_plus.sum() + sigma_minus.sum()).backward()
        assert torch.all(torch.isfinite(proper_values.grad))
        assert torch.all(torch.isfinite(improper_values.grad))

    def test_rejects_mismatched_rotation_counts_and_coefficient_shapes(self):
        """Reject unequal rotation counts or proper/improper coefficient shapes."""
        with pytest.raises(ValueError, match="incompatible values"):
            _character_projection_coefficients_from_rotation_batch(
                torch.zeros((3, 1, 1), dtype=torch.float64),
                torch.ones(2, dtype=torch.float64),
                torch.ones((3, 1, 1), dtype=torch.float64),
            )

        with pytest.raises(ValueError, match="chi_lambda"):
            _character_projections_from_proper_and_improper_coefficients(
                torch.zeros((1, 3, 3, 1), dtype=torch.float64),
                torch.zeros((2, 3, 3, 1), dtype=torch.float64),
                chi_lambda=1,
            )

    def test_is_scriptable(self):
        """Both character-projection tensor operations should compile and run."""
        coefficient_function = torch.jit.script(
            _character_projection_coefficients_from_rotation_batch
        )
        projection_function = torch.jit.script(
            _character_projections_from_proper_and_improper_coefficients
        )
        values = torch.ones((1, 1, 1), dtype=torch.float64)
        coefficients = coefficient_function(
            values,
            torch.ones(1, dtype=torch.float64),
            torch.ones((1, 1, 1), dtype=torch.float64),
        )
        sigma_plus, sigma_minus = projection_function(
            coefficients,
            coefficients,
            0,
        )

        assert sigma_plus.item() == 1.0
        assert sigma_minus.item() == 0.0


class TestWignerStorage:
    """Test persistent Wigner-D storage for the quadrature grid."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_packed_matrices_match_o3(self, dtype):
        """Packing and rank views should preserve the public O(3) matrices."""
        proper_rotation = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=dtype,
        )
        matrices = torch.stack(
            [
                torch.eye(3, dtype=dtype),
                -proper_rotation,
            ]
        )
        max_o3_lambda = 2

        packed = _build_packed_wigner_matrices(matrices, max_o3_lambda)

        assert packed.dim() == 1
        assert packed.numel() == len(matrices) * sum(
            (2 * o3_lambda + 1) ** 2 for o3_lambda in range(max_o3_lambda + 1)
        )
        assert packed.dtype == matrices.dtype
        assert packed.device == matrices.device

        transformations = [
            O3Transformation(matrix, max_o3_lambda) for matrix in matrices.unbind(0)
        ]
        for o3_lambda in range(max_o3_lambda + 1):
            actual = _wigner_matrices_for_lambda(
                packed,
                len(matrices),
                o3_lambda,
            )
            expected = torch.stack(
                [
                    transformation.wigner_D_matrix(o3_lambda)
                    for transformation in transformations
                ]
            )
            assert torch.equal(actual, expected)

        rank_one = _wigner_matrices_for_lambda(packed, len(matrices), 1)
        previous = rank_one[0, 0, 0].clone()
        rank_one[0, 0, 0] += 1
        assert packed[len(matrices)] == previous + 1

    def test_builder_rejects_invalid_inputs(self):
        """The builder should reject invalid ranks, shapes, and dtypes."""
        matrices = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        with pytest.raises(ValueError, match="non-negative"):
            _build_packed_wigner_matrices(matrices, -1)

        for shape in ((0, 3, 3), (2, 3, 2)):
            with pytest.raises(ValueError, match="shape \\(N, 3, 3\\)"):
                _build_packed_wigner_matrices(
                    torch.empty(shape, dtype=torch.float64),
                    1,
                )

        with pytest.raises(TypeError, match="float32 or float64"):
            _build_packed_wigner_matrices(matrices.to(torch.float16), 1)

    def test_rank_view_rejects_invalid_inputs(self):
        """Rank views should reject invalid storage, counts, and ranks."""
        with pytest.raises(ValueError, match="one-dimensional"):
            _wigner_matrices_for_lambda(torch.empty((2, 2)), 1, 0)
        with pytest.raises(ValueError, match="n_matrices must be positive"):
            _wigner_matrices_for_lambda(torch.empty(1), 0, 0)
        with pytest.raises(ValueError, match="o3_lambda must be non-negative"):
            _wigner_matrices_for_lambda(torch.empty(1), 1, -1)
        with pytest.raises(ValueError, match="exceeds the packed"):
            _wigner_matrices_for_lambda(torch.empty(1), 1, 1)

    def test_rank_view_is_scriptable(self):
        """The runtime rank accessor should compile and execute in TorchScript."""
        scripted = torch.jit.script(_wigner_matrices_for_lambda)
        packed = torch.arange(70, dtype=torch.float64)

        assert torch.equal(
            scripted(packed, 2, 2),
            _wigner_matrices_for_lambda(packed, 2, 2),
        )


class TestQuadrature:
    """Test quadrature weights and grid properties."""

    def test_weights_sum(self):
        """Quadrature weights should sum to 1 (normalized Haar measure on SO(3))."""
        for L_max in [3, 5, 7]:
            lebedev_order, n_inplane = _choose_quadrature(L_max)
            _, _, _, w = get_euler_angles_quadrature(lebedev_order, n_inplane)
            # The weights are w_i / (4*pi*K) repeated K times, where w_i sum to 4*pi
            # So total sum = sum(w_i)/(4*pi*K) * K = sum(w_i)/(4*pi) = 1
            assert np.allclose(w.sum(), 1.0, atol=1e-12), (
                f"Weights don't sum to 1 for L_max={L_max}: sum={w.sum()}"
            )

    def test_choose_quadrature_monotone(self):
        """Higher L_max should give equal or larger quadrature grids."""
        prev_n = 0
        for L_max in [3, 5, 7, 11, 15]:
            n, K = _choose_quadrature(L_max)
            assert n >= prev_n
            assert K == L_max + 1
            prev_n = n

    def test_euler_angle_rotations_are_in_so3(self):
        """Euler-angle matrices should be orthogonal with determinant +1."""
        lebedev_order, n_inplane = _choose_quadrature(5)
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

    def test_choose_quadrature_too_large(self):
        with pytest.raises(ValueError, match="exceeds the largest"):
            _choose_quadrature(132)

    @pytest.mark.parametrize("value", [-1, -2])
    def test_choose_quadrature_rejects_negative_degree(self, value):
        with pytest.raises(ValueError, match="non-negative"):
            _choose_quadrature(value)

    @pytest.mark.parametrize("value", [1.5, True])
    def test_choose_quadrature_rejects_non_integer_degree(self, value):
        with pytest.raises(TypeError, match="must be an integer"):
            _choose_quadrature(value)

    @pytest.mark.parametrize("value", [0, -1])
    def test_rotation_quadrature_rejects_non_positive_rotation_count(self, value):
        with pytest.raises(ValueError, match="positive"):
            get_rotation_quadrature(3, value)

    @pytest.mark.parametrize("value", [1.5, True])
    def test_rotation_quadrature_rejects_non_integer_rotation_count(self, value):
        with pytest.raises(TypeError, match="must be an integer"):
            get_rotation_quadrature(3, value)

    def test_rotation_quadrature_rejects_unsupported_lebedev_order(self):
        with pytest.raises(ValueError, match="unsupported Lebedev order"):
            get_rotation_quadrature(4, 3)

    def test_degree_two_grid_resolves_l1_products(self):
        order, n_rotations = _choose_quadrature(2)
        rotations, weights = get_rotation_quadrature(order, n_rotations)
        function = rotations[:, 2, 0]

        norm = np.sum(weights * function**2)
        projection_matrix = np.einsum("g,gij,g->ij", weights, rotations, function)
        projected_norm = 3.0 * np.sum(projection_matrix**2)
        assert np.isclose(norm, 1.0 / 3.0, atol=1e-12)
        assert np.isclose(projected_norm, 1.0 / 3.0, atol=1e-12)

    def test_rotation_quadrature_matrices(self):
        """Return normalized proper matrices and optional improper partners."""
        rotations, weights = get_rotation_quadrature(11, 5)
        assert rotations.shape == (rotations.shape[0], 3, 3)
        assert np.isclose(weights.sum(), 1.0)
        assert np.allclose(
            rotations @ rotations.transpose(0, 2, 1),
            np.broadcast_to(np.eye(3), rotations.shape),
            atol=1e-12,
        )
        assert np.allclose(np.linalg.det(rotations), 1.0, atol=1e-12)

        o3_rotations, o3_weights = get_rotation_quadrature(
            11, 5, include_inversion=True
        )
        assert len(o3_rotations) == 2 * len(rotations)
        assert np.isclose(o3_weights.sum(), 1.0)
        dets = np.linalg.det(o3_rotations)
        assert np.allclose(np.sort(dets), np.repeat([-1.0, 1.0], len(rotations)))


class TestSymmetrizedModelConstruction:
    """Test construction of the quadrature and persistent Wigner-D storage."""

    def test_forward_has_the_exact_model_interface_signature(self):
        """
        Require the canonical ``ModelInterface.forward`` signature.

        The wrapper must accept only ``systems``, ``outputs``, and
        ``selected_atoms``, using the standard annotations and calling
        convention without default values.
        """
        signature = inspect.signature(SymmetrizedModel.forward)
        parameters = list(signature.parameters.values())

        assert [parameter.name for parameter in parameters] == [
            "self",
            "systems",
            "outputs",
            "selected_atoms",
        ]
        assert all(
            parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in parameters
        )
        assert all(
            parameter.default is inspect.Parameter.empty for parameter in parameters
        )
        assert [parameter.annotation for parameter in parameters] == [
            inspect.Parameter.empty,
            List[System],
            Dict[str, ModelOutput],
            Optional[Labels],
        ]
        assert signature.return_annotation == Dict[str, TensorMap]

    def test_constructs_registered_buffers(self):
        """Constructor limits should determine the grid and Wigner-D storage."""
        model = SymmetrizedModel(
            _EmptyModel(),
            max_o3_lambda_target=1,
            max_o3_lambda_input=2,
            max_o3_lambda_character=1,
            batch_size=7,
        )

        assert model.max_o3_lambda_target == 1
        assert model.max_o3_lambda_input == 2
        assert model.max_o3_lambda_character == 1
        assert model.max_o3_lambda_grid == 3
        assert model.batch_size == 7

        buffers = dict(model.named_buffers())
        assert set(buffers) == {
            "_rotation_matrices",
            "_rotation_weights",
            "_packed_wigner_matrices",
        }
        assert buffers["_rotation_matrices"].dtype == torch.float64
        assert buffers["_rotation_weights"].dtype == torch.float64
        assert buffers["_packed_wigner_matrices"].dtype == torch.float64
        assert torch.allclose(
            buffers["_rotation_weights"].sum(),
            torch.tensor(1.0, dtype=torch.float64),
        )

        n_rotations = len(buffers["_rotation_matrices"])
        expected_wigner_elements = n_rotations * sum(
            (2 * o3_lambda + 1) ** 2 for o3_lambda in range(3)
        )
        assert buffers["_packed_wigner_matrices"].numel() == expected_wigner_elements

    def test_character_limit_controls_default_grid(self):
        """Character sectors should raise the default grid degree when necessary."""
        model = SymmetrizedModel(
            _EmptyModel(),
            max_o3_lambda_target=0,
            max_o3_lambda_character=2,
        )

        assert model.max_o3_lambda_grid == 4

    def test_rejects_grid_too_small_for_character_sectors(self):
        """An explicit grid must resolve products for every requested sector."""
        with pytest.raises(ValueError, match="at least twice"):
            SymmetrizedModel(
                _EmptyModel(),
                max_o3_lambda_target=0,
                max_o3_lambda_character=2,
                max_o3_lambda_grid=3,
            )

    @pytest.mark.parametrize(
        ("argument", "value", "error", "message"),
        [
            ("max_o3_lambda_target", -1, ValueError, "non-negative"),
            ("max_o3_lambda_target", True, TypeError, "integer"),
            ("max_o3_lambda_input", 1.5, TypeError, "integer"),
            ("max_o3_lambda_character", -1, ValueError, "non-negative"),
            ("batch_size", 0, ValueError, "positive"),
            ("max_o3_lambda_grid", -1, ValueError, "non-negative"),
            ("max_wigner_storage_bytes", 0, ValueError, "positive"),
        ],
    )
    def test_rejects_invalid_constructor_arguments(
        self,
        argument,
        value,
        error,
        message,
    ):
        """Every integer constructor argument should enforce its documented range."""
        arguments = {"max_o3_lambda_target": 0, argument: value}

        with pytest.raises(error, match=message):
            SymmetrizedModel(_EmptyModel(), **arguments)

    def test_checks_wigner_storage_limit_before_building(self, monkeypatch):
        """An excessive Wigner-D allocation should be rejected before construction."""

        def fail_if_called(*args, **kwargs):
            raise AssertionError("Wigner-D construction should not have started")

        monkeypatch.setattr(
            "metatomic.torch.symmetrized_model._model._build_packed_wigner_matrices",
            fail_if_called,
        )

        with pytest.raises(ValueError, match="exceeding max_wigner_storage_bytes=1"):
            SymmetrizedModel(
                _EmptyModel(),
                max_o3_lambda_target=0,
                max_wigner_storage_bytes=1,
            )

    def test_rejects_a_model_stored_on_an_unsupported_device(self):
        """Reject direct construction from a model outside CPU or CUDA."""
        base_model = _EmptyModel()
        base_model.register_buffer("_device_marker", torch.empty(0, device="meta"))

        with pytest.raises(ValueError, match="supports CPU and CUDA"):
            SymmetrizedModel(base_model, max_o3_lambda_target=0)


class TestSymmetrizedModelForward:
    """Test how requested averages and diagnostics are computed and returned."""

    def test_character_projection_separates_sectors_through_lambda_three(self):
        """
        Separate every O(3) ``(lambda, sigma)`` sector through ``lambda=3``.

        Character projection of the eight analytic polynomial responses must
        produce eight ``(chi_lambda, chi_sigma)`` blocks. Each block must contain
        only the property belonging to the same sector, with squared norms
        ``1``, ``1/3``, ``1/15``, and ``1/105`` for ``lambda=0``, ``1``, ``2``,
        and ``3``. All projections onto the other seven sectors must vanish.
        """
        source_name = "mtt::o3_polynomial_sectors"
        requested_name = "o3::character_projection::" + source_name
        sectors = [
            (chi_lambda, chi_sigma) for chi_lambda in range(4) for chi_sigma in (1, -1)
        ]
        model = SymmetrizedModel(
            _O3PolynomialSectorModel(),
            max_o3_lambda_target=0,
            max_o3_lambda_character=3,
            max_o3_lambda_grid=6,
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
            max_o3_lambda_target=0,
            max_o3_lambda_character=1,
            max_o3_lambda_grid=2,
            batch_size=batch_size,
        )
        system = _forward_test_system([[1.0, 2.0, 3.0]])
        outputs = {
            "energy": ModelOutput(sample_kind="system"),
            "o3::variance::energy": ModelOutput(sample_kind="system"),
            "o3::character_projection::energy": ModelOutput(sample_kind="system"),
        }

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
            max_o3_lambda_target=2,
            max_o3_lambda_character=2,
            max_o3_lambda_grid=4,
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
            max_o3_lambda_target=0,
            max_o3_lambda_character=1,
            max_o3_lambda_grid=2,
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
            max_o3_lambda_target=1,
        )

        with pytest.raises(
            ValueError,
            match=(
                "output 'mtt::spherical_quadrupole' contains o3_lambda=2, "
                "exceeding max_o3_lambda_target=1"
            ),
        ):
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
        """
        Reject a spurious negative variance caused by insufficient quadrature.

        The degree-12 grid can not integrate the degree-14 squared response and
        yields a negative value. Raising the grid degree to 14 must recover the
        exact variance.
        """
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
            max_o3_lambda_target=0,
            max_o3_lambda_grid=12,
            batch_size=64,
        )
        with pytest.raises(ValueError, match="materially negative.*above 12"):
            underresolved([system], variance_request, None)

        resolved = SymmetrizedModel(
            _DegreeSevenEnergyModel(),
            max_o3_lambda_target=0,
            max_o3_lambda_grid=14,
            batch_size=64,
        )
        outputs = {
            "energy": ModelOutput(sample_kind="system"),
            variance_name: ModelOutput(sample_kind="system"),
        }
        result = resolved([system], outputs, None)
        expected_variance = 1.0e6 * 17.0 / 137280.0
        assert result["energy"].block().values.item() == pytest.approx(
            0.0,
            abs=1.0e-12,
        )
        assert result[variance_name].block().values.item() == pytest.approx(
            expected_variance,
            rel=1.0e-12,
        )

    @pytest.mark.parametrize("source_name", ["energy/pbe", "mtt::feature::node"])
    def test_preserves_variant_and_custom_output_names(self, source_name):
        """Return variants and custom outputs under their exact requested names."""
        model = SymmetrizedModel(
            _LinearEnergyModel(),
            max_o3_lambda_target=0,
            max_o3_lambda_grid=2,
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
        assert torch.allclose(
            result[variance_name].block().values,
            torch.tensor([[14.0 / 3.0]], dtype=torch.float64),
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
            max_o3_lambda_target=1,
            max_o3_lambda_grid=2,
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
            max_o3_lambda_target=1,
            max_o3_lambda_grid=2,
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

    def test_equivariant_outputs_preserve_values_metadata_and_zero_variance(self):
        """Return exact equivariant outputs unchanged and report zero variance."""
        system = _forward_test_system([[1.0, 2.0, 3.0], [-0.5, 0.25, 1.0]])
        sources = [
            "energy",
            "non_conservative_force",
            "non_conservative_stress",
            "mtt::spherical_vector",
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
            max_o3_lambda_target=2,
            max_o3_lambda_grid=2,
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

        expected_target_keys = {
            "o3::variance::energy": [[0, 1]],
            "o3::variance::non_conservative_force": [[1, 1]],
            "o3::variance::non_conservative_stress": [[0, 1], [2, 1]],
            "o3::variance::mtt::spherical_vector": [[1, 1]],
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
            max_o3_lambda_target=0,
            max_o3_lambda_grid=2,
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

        with torch.no_grad():
            inference_result = model(
                [system],
                {"energy": ModelOutput(sample_kind="system")},
                None,
            )
        assert not inference_result["energy"].block().values.requires_grad

    def test_rejects_invalid_requests_before_model_evaluation(self):
        """Invalid public requests should fail without running the source model."""
        base_model = _CountingLinearEnergyModel()
        model = SymmetrizedModel(base_model, max_o3_lambda_target=0)
        system = _forward_test_system([[1.0, 2.0, 3.0]])

        assert model([], {}, None) == {}
        with pytest.raises(ValueError, match="at least one System"):
            model([], {"energy": ModelOutput(sample_kind="system")}, None)
        with pytest.raises(ValueError, match="max_o3_lambda_character must be set"):
            model(
                [system],
                {"o3::character_projection::energy": ModelOutput(sample_kind="system")},
                None,
            )
        with pytest.raises(ValueError, match="does not support explicit gradients"):
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
        assert base_model.call_count == 0

    def test_is_scriptable_and_serializable(self, tmp_path):
        """The complete forward path should execute after scripting and reloading."""
        constructor_arguments = {
            "max_o3_lambda_target": 0,
            "max_o3_lambda_character": 1,
            "max_o3_lambda_grid": 2,
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


class TestSymmetrizedModelWrap:
    """Test exported-model capabilities, dependencies, and execution."""

    @pytest.mark.parametrize("max_o3_lambda_character", [None, 1])
    def test_transfers_metadata_and_declared_capabilities(
        self,
        max_o3_lambda_character,
    ):
        """Publish truthful diagnostics without duplicating deprecated aliases."""
        metadata = ModelMetadata(
            name="base model",
            description="Metadata that should remain unchanged.",
            authors=["A. Developer"],
            references={"implementation": ["doi:10.0000/example"]},
            extra={"version": "test"},
        )
        source_outputs = {
            "energy": ModelOutput(
                unit="eV",
                sample_kind="system",
                explicit_gradients=["positions"],
                description="Original energy description.",
            ),
            "mass": ModelOutput(
                unit="u",
                sample_kind="atom",
                description="Original mass description.",
            ),
            "mtt::pair": ModelOutput(
                sample_kind="atom_pair",
                description="Original pair description.",
            ),
        }
        base = AtomisticModel(
            _EmptyModel().eval(),
            metadata,
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
            max_o3_lambda_target=0,
            max_o3_lambda_character=max_o3_lambda_character,
            max_o3_lambda_grid=2,
        )

        actual_metadata = wrapped.metadata()
        assert actual_metadata.name == metadata.name
        assert actual_metadata.description == metadata.description
        assert actual_metadata.authors == metadata.authors
        assert actual_metadata.references == metadata.references
        assert actual_metadata.extra == metadata.extra

        capabilities = wrapped.capabilities()
        assert capabilities.atomic_types == [1, 6, 8]
        assert capabilities.interaction_range == 4.5
        assert capabilities.length_unit == "A"
        assert capabilities.supported_devices == ["cuda", "cpu"]
        assert capabilities.dtype == "float32"

        declared_names = set(wrapped._model_capabilities_outputs_names)
        expected_names = set(source_outputs)
        expected_names.update("o3::variance::" + name for name in source_outputs)
        if max_o3_lambda_character is not None:
            expected_names.update(
                "o3::character_projection::" + name for name in source_outputs
            )
        assert declared_names == expected_names

        # ``AtomisticModel`` adds this compatibility alias, but it must not become
        # another declared source with its own diagnostics.
        assert "masses" in capabilities.outputs
        assert "o3::variance::masses" not in capabilities.outputs
        assert "o3::character_projection::masses" not in capabilities.outputs

        source_units = {"energy": "eV", "mass": "u", "mtt::pair": ""}
        source_sample_kinds = {
            "energy": "system",
            "mass": "atom",
            "mtt::pair": "atom_pair",
        }
        for name, source_output in source_outputs.items():
            average = capabilities.outputs[name]
            assert average.unit == source_units[name]
            assert average.sample_kind == source_sample_kinds[name]
            assert average.explicit_gradients == []
            assert source_output.description in average.description

            squared_unit = (
                "" if source_output.unit == "" else f"({source_output.unit})^2"
            )
            variance = capabilities.outputs["o3::variance::" + name]
            assert variance.unit == squared_unit
            assert variance.sample_kind == source_output.sample_kind
            assert variance.explicit_gradients == []

            character_name = "o3::character_projection::" + name
            if max_o3_lambda_character is None:
                assert character_name not in capabilities.outputs
            else:
                character = capabilities.outputs[character_name]
                assert character.unit == squared_unit
                assert character.sample_kind == source_output.sample_kind
                assert character.explicit_gradients == []

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

        with pytest.raises(ValueError, match="prefix reserved"):
            SymmetrizedModel.wrap(base, max_o3_lambda_target=0)

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

        with pytest.raises(ValueError, match="supports CPU and CUDA"):
            SymmetrizedModel.wrap(base, max_o3_lambda_target=0)

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
            max_o3_lambda_target=0,
            max_o3_lambda_character=1,
            max_o3_lambda_grid=2,
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
        """Match CPU results after moving a saved float32 wrapper to CUDA."""
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
            max_o3_lambda_target=0,
            max_o3_lambda_character=1,
            max_o3_lambda_grid=2,
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

        assert cuda_system.positions.device.type == "cuda"
        cuda_neighbors = cuda_system.get_neighbor_list(neighbor_options)
        assert cuda_neighbors.values.device.type == "cuda"
        assert cuda_neighbors.samples.device.type == "cuda"
        cuda_field = cuda_system.get_data("mtt::field")
        assert cuda_field.keys.device.type == "cuda"
        assert cuda_field.block().values.device.type == "cuda"
        assert cuda_field.block().samples.device.type == "cuda"

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
        assert cuda_model.capabilities().dtype == "float32"
        assert cuda_model.capabilities().supported_devices == ["cpu", "cuda"]
        assert expected["energy"].block().values.item() == pytest.approx(
            295.2,
            rel=2.0e-5,
        )
        assert actual["energy"].keys.names == ["_"]
        variance = actual["o3::variance::energy"]
        assert variance.keys.names == ["o3_lambda", "o3_sigma"]
        assert variance.keys.values.cpu().tolist() == [[0, 1]]
        assert variance.block().components == []
        projection = actual["o3::character_projection::energy"]
        assert projection.keys.names == [
            "o3_lambda",
            "o3_sigma",
            "chi_lambda",
            "chi_sigma",
        ]
        assert projection.keys.values.cpu().tolist() == [
            [0, 1, 0, 1],
            [0, 1, 0, -1],
            [0, 1, 1, 1],
            [0, 1, 1, -1],
        ]
        for block in projection.blocks():
            assert len(block.components) == 1
            assert block.components[0].names == ["o3_mu"]
            assert len(block.components[0]) == 1
        for name, tensor in actual.items():
            assert tensor.keys.device.type == "cuda"
            for block in tensor.blocks():
                assert block.values.device.type == "cuda"
                assert block.values.dtype == torch.float32
                assert block.samples.device.type == "cuda"
                assert block.properties.device.type == "cuda"
                assert all(
                    component.device.type == "cuda" for component in block.components
                )
            mts.allclose_raise(
                tensor.to(device="cpu"),
                expected[name],
                rtol=2.0e-5,
                atol=2.0e-5,
            )


class TestSelectedAtomsColumnOrder:
    def test_system_column_found_by_name(self):
        # the rotated-copy index must go into the "system" column wherever it
        # is, not positionally into column 0
        selection = Labels(["atom", "system"], torch.tensor([[3, 0], [5, 0]]))
        rotated = _map_selected_atoms_to_rotated_copies(selection, 0, 2)
        assert rotated.names == ["atom", "system"]
        assert rotated.values[:, 0].tolist() == [3, 5, 3, 5]
        assert rotated.values[:, 1].tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize(
    ("sample_values", "message"),
    [
        ([[0, 0], [2, 0]], "out-of-range rotated-copy indices"),
        ([[0, 0], [0, 1], [1, 0]], "same sample labels"),
        ([[0, 0], [0, 1], [0, 2], [1, 0]], "same sample labels"),
        ([[0, 0], [0, 1], [1, 0], [1, 2]], "same sample labels"),
    ],
)
def test_rotated_copy_layout_rejects_inconsistent_samples(sample_values, message):
    """Samples from different rotated copies must never be mixed."""
    samples = Labels(["system", "atom"], torch.tensor(sample_values))
    block = TensorBlock(
        values=torch.zeros((len(samples), 1), dtype=torch.float64),
        samples=samples,
        components=[],
        properties=Labels.range("property", 1),
    )

    with pytest.raises(ValueError, match=message):
        _group_samples_by_rotated_copy(block, n_rotated_copies=2)


@pytest.mark.parametrize(
    ("sample_values", "values", "n_rotated_copies", "expected_values"),
    [
        (
            [[3, 0], [5, 0]],
            [3.0, 5.0],
            1,
            [[[3.0], [5.0]]],
        ),
        (
            [[3, 1], [3, 0], [5, 1], [5, 0]],
            [13.0, 3.0, 15.0, 5.0],
            2,
            [[[3.0], [5.0]], [[13.0], [15.0]]],
        ),
    ],
)
def test_group_samples_by_rotated_copy(
    sample_values, values, n_rotated_copies, expected_values
):
    """Values and shared labels should remain aligned after grouping."""
    samples = Labels(["atom", "system"], torch.tensor(sample_values))
    block = TensorBlock(
        values=torch.tensor(values, dtype=torch.float64).reshape(-1, 1),
        samples=samples,
        components=[],
        properties=Labels.range("property", 1),
    )

    grouped_values, shared_names, shared_values = _group_samples_by_rotated_copy(
        block, n_rotated_copies
    )

    assert torch.equal(
        grouped_values,
        torch.tensor(expected_values, dtype=torch.float64),
    )
    assert shared_names == ["atom"]
    assert shared_values.tolist() == [[3], [5]]


@pytest.mark.parametrize(
    ("sample_names", "sample_values", "expected_names", "expected_values"),
    [
        ([], [[]], ["system"], [[7]]),
        (["atom"], [[3], [5]], ["system", "atom"], [[7, 3], [7, 5]]),
    ],
)
def test_restore_input_system_to_samples(
    sample_names, sample_values, expected_names, expected_values
):
    """The original system index should be restored without changing samples."""
    samples = _restore_input_system_to_samples(
        sample_names,
        torch.tensor(sample_values, dtype=torch.int64),
        input_system_index=7,
        device=torch.device("cpu"),
    )

    assert samples.names == expected_names
    assert samples.values.tolist() == expected_values
    assert samples.device == torch.device("cpu")


@pytest.mark.parametrize("component_shape", [(), (2, 3)])
def test_weighted_centered_batch_moments(component_shape):
    """Compute weighted moments and reuse one fixed reference across batches."""
    n_rotated_copies = 3
    n_samples = 2
    n_properties = 2
    values = torch.arange(
        n_rotated_copies * n_samples * int(np.prod(component_shape)) * n_properties,
        dtype=torch.float64,
    ).reshape(n_rotated_copies * n_samples, *component_shape, n_properties)
    components = [
        Labels.range(name, size)
        for name, size in zip(("a", "b"), component_shape, strict=False)
    ]
    tensor = TensorMap(
        Labels("kind", torch.tensor([[0]])),
        [
            TensorBlock(
                values=values,
                samples=Labels(
                    ["system", "item"],
                    torch.tensor(
                        [
                            [copy, item]
                            for copy in range(n_rotated_copies)
                            for item in (5, 7)
                        ]
                    ),
                ),
                components=components,
                properties=Labels.range("property", n_properties),
            )
        ],
    )
    weights = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)

    moments = _reduce_weighted_centered_batch(
        tensor,
        weights,
        input_system_index=4,
        reference=None,
        compute_second_moments=True,
    )
    first_moment, second, absolute_second, reference = moments

    values_by_copy = values.reshape(
        n_rotated_copies, n_samples, *component_shape, n_properties
    )
    centered = values_by_copy - values_by_copy[0]
    weight_shape = (n_rotated_copies,) + (1,) * (centered.ndim - 1)
    assert torch.allclose(
        first_moment.block().values,
        torch.sum(weights.reshape(weight_shape) * centered, dim=0),
    )
    squared_norms = centered**2
    if component_shape:
        squared_norms = squared_norms.sum(dim=tuple(range(2, 2 + len(component_shape))))
    assert second is not None
    assert absolute_second is not None
    assert torch.allclose(
        second.block().values,
        torch.sum(weights.reshape(n_rotated_copies, 1, 1) * squared_norms, dim=0),
    )
    assert torch.allclose(
        absolute_second.block().values,
        torch.sum(
            torch.abs(weights).reshape(n_rotated_copies, 1, 1) * squared_norms,
            dim=0,
        ),
    )
    expected_samples = Labels(
        ["system", "item"],
        torch.tensor([[4, 5], [4, 7]]),
    )
    assert first_moment.keys == tensor.keys
    assert first_moment.block().samples == expected_samples
    assert first_moment.block().components == components
    assert first_moment.block().properties == tensor.block().properties
    assert second.block().samples == expected_samples
    assert second.block().components == []
    assert second.block().properties == tensor.block().properties
    assert absolute_second.block().samples == expected_samples
    assert absolute_second.block().components == []
    assert absolute_second.block().properties == tensor.block().properties

    initial_reference_values = values_by_copy[0].clone()
    assert torch.equal(reference.block().values, initial_reference_values)

    # Simulate a later batch with the same layout but different response values.
    tensor.block().values.add_(10.0)
    later_values_by_copy = tensor.block().values.reshape(
        n_rotated_copies, n_samples, *component_shape, n_properties
    )
    later_centered = later_values_by_copy - initial_reference_values.unsqueeze(0)

    later_moments = _reduce_weighted_centered_batch(
        tensor,
        weights,
        input_system_index=4,
        reference=reference,
        compute_second_moments=False,
    )
    first_moment, second, absolute_second, reused_reference = later_moments
    assert torch.allclose(
        first_moment.block().values,
        torch.sum(weights.reshape(weight_shape) * later_centered, dim=0),
    )
    assert second is None
    assert absolute_second is None
    assert reused_reference is reference
    assert torch.equal(reference.block().values, initial_reference_values)


def test_join_per_system_tensormaps_with_matching_keys(monkeypatch):
    """Systems with identical keys should be joined along samples."""
    tensors = [
        TensorMap(
            Labels("kind", torch.tensor([[0]])),
            [
                TensorBlock(
                    values=torch.tensor([[value]], dtype=torch.float64),
                    samples=Labels("system", torch.tensor([[system_index]])),
                    components=[],
                    properties=Labels.range("property", 1),
                )
            ],
        )
        for system_index, value in enumerate((1.0, 2.0))
    ]

    native_join = mts.join
    different_keys_arguments = []

    def record_join(tensors, axis, different_keys):
        different_keys_arguments.append(different_keys)
        return native_join(tensors, axis, different_keys=different_keys)

    monkeypatch.setattr(mts, "join", record_join)
    joined = _join_per_system_tensormaps(tensors)

    assert different_keys_arguments == ["error"]
    assert joined.keys == tensors[0].keys
    assert joined.block().samples.values.tolist() == [[0], [1]]
    assert joined.block().values.tolist() == [[1.0], [2.0]]


def test_join_per_system_tensormaps_with_different_keys(monkeypatch):
    """System-dependent keys should be joined through their union."""
    tensors = [
        TensorMap(
            Labels("kind", torch.tensor([[key]])),
            [
                TensorBlock(
                    values=torch.tensor([[value]], dtype=torch.float64),
                    samples=Labels("system", torch.tensor([[system_index]])),
                    components=[],
                    properties=Labels.range("property", 1),
                )
            ],
        )
        for system_index, (key, value) in enumerate(((0, 1.0), (1, 2.0)))
    ]

    native_join = mts.join
    different_keys_arguments = []

    def record_join(tensors, axis, different_keys):
        different_keys_arguments.append(different_keys)
        return native_join(tensors, axis, different_keys=different_keys)

    monkeypatch.setattr(mts, "join", record_join)
    joined = _join_per_system_tensormaps(tensors)

    assert different_keys_arguments == ["union"]
    assert joined.keys.values.tolist() == [[0], [1]]
    assert joined.block(0).samples.values.tolist() == [[0]]
    assert joined.block(0).values.tolist() == [[1.0]]
    assert joined.block(1).samples.values.tolist() == [[1]]
    assert joined.block(1).values.tolist() == [[2.0]]


@pytest.mark.parametrize(
    ("component_shape", "n_samples"),
    [((), 2), ((3,), 2), ((2, 3), 2), ((2, 3), 0)],
)
def test_component_norm_squared(component_shape, n_samples):
    """All component axes should be contracted without changing metadata."""
    shape = (n_samples, *component_shape, 2)
    values = torch.arange(int(np.prod(shape)), dtype=torch.float64).reshape(shape)
    tensor = _make_single_block_tensor_map(values)

    result = _component_norm_squared(tensor)

    expected = values.square()
    if component_shape:
        expected = expected.sum(dim=tuple(range(1, 1 + len(component_shape))))
    assert torch.equal(result.block().values, expected)
    assert result.keys == tensor.keys
    assert result.block().samples == tensor.block().samples
    assert result.block().components == []
    assert result.block().properties == tensor.block().properties


def test_variance_from_centered_moments():
    """Centered first and second moments should give component-summed variance."""
    component_shape = (2, 3)
    shape = (2, *component_shape, 2)
    centered_first_moment_values = (
        torch.arange(int(np.prod(shape)), dtype=torch.float64).reshape(shape) / 10
    )
    centered_first_moment = _make_single_block_tensor_map(centered_first_moment_values)

    norm_squared = centered_first_moment_values.square().sum(dim=(1, 2))
    expected_variance = torch.tensor([[0.25, 0.5], [0.75, 1.0]], dtype=torch.float64)
    centered_second_moment = _make_single_block_tensor_map(
        norm_squared + expected_variance
    )
    absolute_centered_second_moment = _make_single_block_tensor_map(
        norm_squared + expected_variance + 1.0
    )

    variance = _variance_from_centered_moments(
        centered_first_moment,
        centered_second_moment,
        absolute_centered_second_moment,
        n_grid_points=12,
        max_o3_lambda_grid=3,
    )

    assert torch.allclose(variance.block().values, expected_variance)
    assert variance.keys == centered_first_moment.keys
    assert variance.block().samples == centered_first_moment.block().samples
    assert variance.block().components == []
    assert variance.block().properties == centered_first_moment.block().properties


def test_centered_variance_is_stable_with_large_offset():
    """A common offset should not cause cancellation in the variance."""
    values = torch.tensor(
        [1.0e12, 1.0e12 + 1.0, 1.0e12 + 2.0, 1.0e12 + 3.0],
        dtype=torch.float64,
    ).reshape(-1, 1)
    tensor = _make_single_block_tensor_map(values, sample_name="system")
    weights = torch.tensor([0.125, 0.375, 0.375, 0.125], dtype=torch.float64)

    first, second, absolute_second, _ = _reduce_weighted_centered_batch(
        tensor,
        weights,
        input_system_index=7,
        reference=None,
        compute_second_moments=True,
    )
    assert second is not None
    assert absolute_second is not None
    variance = _variance_from_centered_moments(
        first,
        second,
        absolute_second,
        n_grid_points=4,
        max_o3_lambda_grid=3,
    )

    assert torch.allclose(
        variance.block().values,
        torch.tensor([[0.75]], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1.0e-12, 1.0e12])
def test_roundoff_negative_diagnostic_uses_its_scale(dtype, scale):
    """Only negative values within the summation tolerance should be clamped."""
    n_grid_points = 100
    n_epsilon = n_grid_points * torch.finfo(dtype).eps
    gamma = n_epsilon / (1.0 - n_epsilon)
    tolerance = 64.0 * gamma * scale

    cleaned = _clamp_roundoff_negative_diagnostic(
        _make_single_block_tensor_map(
            torch.tensor([[-0.5 * tolerance], [2.0]], dtype=dtype)
        ),
        _make_single_block_tensor_map(torch.tensor([[scale], [scale]], dtype=dtype)),
        n_grid_points=n_grid_points,
        quantity="variance",
        max_o3_lambda_grid=3,
    )
    assert cleaned.block().values[0, 0].item() == 0.0
    assert cleaned.block().values[1, 0].item() == 2.0

    with pytest.raises(ValueError, match="materially negative"):
        _clamp_roundoff_negative_diagnostic(
            _make_single_block_tensor_map(
                torch.tensor([[-2.0 * tolerance]], dtype=dtype)
            ),
            _make_single_block_tensor_map(torch.tensor([[scale]], dtype=dtype)),
            n_grid_points=n_grid_points,
            quantity="variance",
            max_o3_lambda_grid=3,
        )


@pytest.mark.parametrize(
    ("value", "scale"),
    [
        (float("nan"), 1.0),
        (0.0, float("inf")),
        (0.0, -1.0),
    ],
)
def test_roundoff_negative_diagnostic_rejects_invalid_input(value, scale):
    """Values and their numerical scales should be finite and scales non-negative."""
    with pytest.raises(ValueError, match="round-off scale is invalid"):
        _clamp_roundoff_negative_diagnostic(
            _make_single_block_tensor_map(torch.tensor([[value]], dtype=torch.float64)),
            _make_single_block_tensor_map(torch.tensor([[scale]], dtype=torch.float64)),
            n_grid_points=100,
            quantity="variance",
            max_o3_lambda_grid=3,
        )


def test_roundoff_negative_diagnostic_rejects_unsupported_dtype():
    """Diagnostics should use one of the supported floating-point dtypes."""
    with pytest.raises(TypeError, match="float32 or float64"):
        _clamp_roundoff_negative_diagnostic(
            _make_single_block_tensor_map(torch.tensor([[0.0]], dtype=torch.float16)),
            _make_single_block_tensor_map(torch.tensor([[1.0]], dtype=torch.float16)),
            n_grid_points=100,
            quantity="variance",
            max_o3_lambda_grid=3,
        )


def test_variance_from_centered_moments_is_scriptable():
    """The complete centered-variance calculation should compile with TorchScript."""
    torch.jit.script(_variance_from_centered_moments)


@pytest.mark.parametrize(
    ("component_shape", "n_samples"),
    [((), 2), ((3,), 2), ((2, 3), 2), ((3,), 0)],
)
def test_mean_variance_over_components(component_shape, n_samples):
    """Divide by component count without aggregating or creating samples."""
    variance_values = (
        torch.arange(n_samples * 2, dtype=torch.float64).reshape(n_samples, 2) + 1.0
    )
    variance = _make_single_block_tensor_map(variance_values, sample_name="atom")
    component_layout = _make_single_block_tensor_map(
        torch.zeros(n_samples, *component_shape, 2, dtype=torch.float64),
        sample_name="atom",
    )

    result = _mean_variance_over_components(variance, component_layout)

    n_components = int(np.prod(component_shape)) if component_shape else 1
    assert torch.equal(result.block().values, variance_values / n_components)
    assert result.keys == variance.keys
    assert result.block().samples == variance.block().samples
    assert result.block().components == []
    assert result.block().properties == variance.block().properties


@pytest.mark.parametrize(
    ("requested_name", "source_name", "calculation"),
    [
        ("energy", "energy", "average"),
        ("energy/pbe", "energy/pbe", "average"),
        ("mtt::aux::features", "mtt::aux::features", "average"),
        ("o3::variance::energy/pbe", "energy/pbe", "variance"),
        (
            "o3::variance::mtt::aux::features",
            "mtt::aux::features",
            "variance",
        ),
        (
            "o3::character_projection::mtt::feature::layer.0",
            "mtt::feature::layer.0",
            "character_projection",
        ),
        (
            "o3::variance_extra::energy",
            "o3::variance_extra::energy",
            "average",
        ),
    ],
)
def test_parse_output_request(requested_name, source_name, calculation):
    """Recognize only complete prefixes and preserve the remaining name."""
    assert _parse_output_request(requested_name) == (source_name, calculation)


@pytest.mark.parametrize(
    "requested_name",
    ["", "o3::variance::", "o3::character_projection::"],
)
def test_parse_output_request_requires_source_name(requested_name):
    """Every request should identify an underlying model output."""
    with pytest.raises(ValueError, match="does not identify"):
        _parse_output_request(requested_name)


def test_group_output_requests_by_source_and_calculation():
    """Group requests while retaining each requested output name and sample kind."""
    outputs = {
        "energy": ModelOutput(sample_kind="system"),
        "o3::variance::energy": ModelOutput(sample_kind="system"),
        "o3::character_projection::energy": ModelOutput(sample_kind="system"),
        "o3::variance::mtt::aux::pairs": ModelOutput(sample_kind="atom_pair"),
    }

    (
        source_sample_kinds,
        average_names,
        variance_names,
        character_projection_names,
    ) = _group_output_requests(outputs)

    assert source_sample_kinds == {
        "energy": "system",
        "mtt::aux::pairs": "atom_pair",
    }
    assert average_names == {"energy": "energy"}
    assert variance_names == {
        "energy": "o3::variance::energy",
        "mtt::aux::pairs": "o3::variance::mtt::aux::pairs",
    }
    assert character_projection_names == {"energy": "o3::character_projection::energy"}


def test_group_output_requests_rejects_mixed_sample_kinds():
    """One source cannot share an evaluation at two sample resolutions."""
    outputs = {
        "energy": ModelOutput(sample_kind="system"),
        "o3::variance::energy": ModelOutput(sample_kind="atom"),
    }

    with pytest.raises(ValueError, match="must use the same sample_kind"):
        _group_output_requests(outputs)


@pytest.mark.parametrize(
    ("o3_lambda", "expected"),
    [
        (0, [0]),
        (1, [-1, 0, 1]),
        (2, [-2, -1, 0, 1, 2]),
    ],
)
def test_o3_mu_labels(o3_lambda, expected):
    """Spherical components should be ordered from -lambda to +lambda."""
    labels = _o3_mu_labels(o3_lambda, torch.device("cpu"))

    assert labels.names == ["o3_mu"]
    assert labels.values[:, 0].tolist() == expected
    assert labels.device == torch.device("cpu")


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
    """Identity, traceless diagonal, and skew matrices should map as expected."""
    matrices = torch.zeros((3, 3, 3, 1), dtype=torch.float64)
    matrices[0, :, :, 0] = torch.eye(3, dtype=torch.float64)
    matrices[1, 0, 0, 0] = 1.0
    matrices[1, 1, 1, 0] = -1.0
    matrices[2, 0, 1, 0] = 2.0
    matrices[2, 1, 0, 0] = -2.0

    l0, l2 = _symmetric_matrices_to_spherical(matrices)

    expected_l0 = torch.zeros((3, 1, 1), dtype=torch.float64)
    expected_l0[0, 0, 0] = 3.0**0.5
    expected_l2 = torch.zeros((3, 5, 1), dtype=torch.float64)
    expected_l2[1, 4, 0] = 2.0**0.5
    assert torch.allclose(l0, expected_l0, rtol=0.0, atol=1.0e-12)
    assert torch.allclose(l2, expected_l2, rtol=0.0, atol=1.0e-12)


def test_symmetric_matrices_to_spherical_preserves_norm():
    """The spherical norm should equal the symmetric-part Frobenius norm."""
    generator = torch.Generator().manual_seed(1234)
    matrices = torch.randn(
        (4, 3, 3, 2),
        dtype=torch.float64,
        generator=generator,
    )
    symmetric = 0.5 * (matrices + matrices.transpose(1, 2))

    l0, l2 = _symmetric_matrices_to_spherical(matrices)

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
    (
        "names",
        "values",
        "o3_lambda",
        "o3_sigma",
        "expected_names",
        "expected_values",
    ),
    [
        (["_"], [[0]], 2, 1, ["o3_lambda", "o3_sigma"], [[2, 1]]),
        (
            ["channel"],
            [[3], [7]],
            1,
            -1,
            ["channel", "o3_lambda", "o3_sigma"],
            [[3, 1, -1], [7, 1, -1]],
        ),
        (
            ["channel", "o3_lambda"],
            [[3, 1], [7, 1]],
            1,
            -1,
            ["channel", "o3_lambda", "o3_sigma"],
            [[3, 1, -1], [7, 1, -1]],
        ),
    ],
)
def test_add_o3_irrep_to_keys(
    names,
    values,
    o3_lambda,
    o3_sigma,
    expected_names,
    expected_values,
):
    """Preserve semantic keys while assigning one O(3) irrep."""
    result = _add_o3_irrep_to_keys(
        Labels(names, torch.tensor(values)),
        o3_lambda,
        o3_sigma,
    )

    assert result.names == expected_names
    assert result.values.tolist() == expected_values


@pytest.mark.parametrize(
    ("names", "values", "message"),
    [
        (["_"], [[1]], "placeholder"),
        (["channel", "o3_lambda"], [[3, 1], [7, 2]], "o3_lambda"),
    ],
)
def test_add_o3_irrep_to_keys_rejects_conflicting_metadata(names, values, message):
    """Reject an invalid ``_`` placeholder or conflicting irrep key values."""
    with pytest.raises(ValueError, match=message):
        _add_o3_irrep_to_keys(
            Labels(names, torch.tensor(values)),
            o3_lambda=1,
            o3_sigma=1,
        )


@pytest.mark.parametrize(
    "source_name",
    [
        "energy",
        "energy/pbe",
        "energy_ensemble/member",
        "energy_uncertainty/direct",
    ],
)
def test_decompose_output_energy_like(source_name):
    """Energy-like variants should become one scalar spherical block."""
    values = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    tensor = _tensor_map_with_components(values, [])
    tensor.set_info("unit", "eV")

    result = _decompose_output(source_name, tensor)

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
        "non_conservative_forces/direct",
    ],
)
def test_decompose_output_non_conservative_force_preserves_autograd(source_name):
    """Both force spellings should become l=1 and preserve implicit autograd."""
    values = torch.tensor(
        [[[1.0], [2.0], [3.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    tensor = _tensor_map_with_components(values, ["xyz"])

    result = _decompose_output(source_name, tensor)

    assert result.keys.names == ["o3_lambda", "o3_sigma"]
    assert result.keys.values.tolist() == [[1, 1]]
    assert result.block().components == [_o3_mu_labels(1, values.device)]
    assert torch.equal(
        result.block().values,
        torch.tensor([[[2.0], [3.0], [1.0]]], dtype=torch.float64),
    )

    result.block().values.sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))


def test_decompose_output_non_conservative_stress_combines_irreps():
    """Stress should return l=0 and l=2 blocks and silently discard skew."""
    values = torch.zeros((2, 3, 3, 1), dtype=torch.float64)
    values[0, :, :, 0] = torch.eye(3, dtype=torch.float64)
    values[1, 0, 1, 0] = 2.0
    values[1, 1, 0, 0] = -2.0
    tensor = _tensor_map_with_components(values, ["xyz_1", "xyz_2"])

    result = _decompose_output("non_conservative_stress/direct", tensor)

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


def test_decompose_output_does_not_infer_custom_cartesian_semantics():
    """A generic 3x3 output should pass through unchanged."""
    tensor = _tensor_map_with_components(
        torch.rand((1, 3, 3, 1), dtype=torch.float64),
        ["xyz_1", "xyz_2"],
    )

    result = _decompose_output("mtt::custom", tensor)

    assert result is tensor


@pytest.mark.parametrize(
    ("source_name", "shape", "component_names", "message"),
    [
        ("energy", (1, 3, 1), ["xyz"], "must not have components"),
        (
            "non_conservative_force",
            (1, 3, 1),
            ["component"],
            "one 'xyz' component axis",
        ),
        (
            "non_conservative_stress",
            (1, 3, 3, 1),
            ["xyz_1", "component"],
            "'xyz_1' and 'xyz_2' component axes",
        ),
    ],
)
def test_decompose_output_rejects_invalid_standard_components(
    source_name,
    shape,
    component_names,
    message,
):
    """Standard quantities should use their required Cartesian component axes."""
    tensor = _tensor_map_with_components(
        torch.zeros(shape, dtype=torch.float64),
        component_names,
    )

    with pytest.raises(ValueError, match=message):
        _decompose_output(source_name, tensor)


def test_decompose_output_rejects_attached_gradients():
    """Decomposition should not silently discard explicit TensorBlock gradients."""
    properties = Labels.range("property", 1)
    block = TensorBlock(
        values=torch.ones((1, 1), dtype=torch.float64),
        samples=Labels.range("system", 1),
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
    tensor = TensorMap(
        Labels("_", torch.tensor([[0]], dtype=torch.int64)),
        [block],
    )

    with pytest.raises(ValueError, match="gradients attached to 'energy'"):
        _decompose_output("energy", tensor)


def test_decompose_output_is_scriptable():
    """The output decomposition should compile with TorchScript."""
    torch.jit.script(_decompose_output)
