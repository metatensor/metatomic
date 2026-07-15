"""Tests for symmetrized_model.py standalone functions and SymmetrizedModel class."""

import io
from typing import Dict, List, Optional

import metatensor.torch as mts
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from scipy.spatial.transform import Rotation

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
    load_atomistic_model,
    register_autograd_neighbors,
    unit_conversion_factor,
)
from metatomic.torch.o3 import O3Transformation, transform_system, transform_tensor
from metatomic.torch.o3 import _tranformations as transformation_module
from metatomic.torch.o3._wigner import (
    _complex_to_real_spherical_harmonics_transform,
    _compute_real_wigner_d_matrices,
    build_wigner_D_cache,
)
from metatomic.torch.symmetrized_model import (
    SymmetrizedModel,
    get_rotation_quadrature,
    per_system_character_fractions,
    per_system_equivariance_rmse,
)
from metatomic.torch.symmetrized_model import _model as symmetrized_model_module
from metatomic.torch.symmetrized_model._decompose import (
    _decompose_output,
    _l0_components_from_matrices,
    _l2_components_from_matrices,
)
from metatomic.torch.symmetrized_model._gradients import _evaluate_with_gradients
from metatomic.torch.symmetrized_model._model import (
    _reduce_weighted_batch_moments,
    _transform_system_batch,
    _validate_nonnegative_diagnostic,
)
from metatomic.torch.symmetrized_model._quadrature import (
    _choose_quadrature,
    _rotations_from_angles,
    get_euler_angles_quadrature,
)
from metatomic.torch.symmetrized_model._utils import (
    _reshape_block_by_local_system,
    _selected_atoms_for_local_systems,
)

from ._tests_utils import can_use_mps_backend


_ENERGY_OUTPUTS = {"energy": ModelOutput(sample_kind="system")}


def _symmetrized(base_model, **kwargs):
    """Build the float64 wrapper used by most behavioral tests."""
    kwargs.setdefault("max_o3_lambda_target", 1)
    kwargs.setdefault("batch_size", 4)
    return SymmetrizedModel(base_model, **kwargs).to(dtype=torch.float64)


def _evaluate_gradients(model, system, rotations=None, **kwargs):
    if rotations is None:
        rotations = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    return _evaluate_with_gradients(
        model,
        system,
        rotations,
        _ENERGY_OUTPUTS,
        None,
        torch.device("cpu"),
        torch.float64,
        **kwargs,
    )


def _single_block_tensor_map(
    values: torch.Tensor,
    *,
    samples: Labels,
    components: List[Labels],
    properties: Labels,
    keys: Optional[Labels] = None,
) -> TensorMap:
    """Build a TensorMap without repeating single-block test boilerplate."""
    if keys is None:
        keys = Labels(
            ["_"], torch.tensor([[0]], dtype=torch.int64, device=values.device)
        )
    return TensorMap(
        keys,
        [
            TensorBlock(
                values=values,
                samples=samples,
                components=components,
                properties=properties,
            )
        ],
    )


def _system_tensor_map(
    values: torch.Tensor,
    property_name: str = "energy",
    ell: int = -1,
) -> TensorMap:
    """Build the single-block, per-system outputs used by the mock models."""
    device = values.device
    if ell < 0:
        keys = Labels(["_"], torch.tensor([[0]], dtype=torch.int64, device=device))
        components: List[Labels] = []
    else:
        keys = Labels(
            ["o3_lambda", "o3_sigma"],
            torch.tensor([[ell, 1]], dtype=torch.int64, device=device),
        )
        components = [
            Labels(
                ["o3_mu"],
                torch.arange(-ell, ell + 1, dtype=torch.int64, device=device).reshape(
                    -1, 1
                ),
            )
        ]

    return TensorMap(
        keys,
        [
            TensorBlock(
                values=values,
                samples=Labels(
                    ["system"],
                    torch.arange(
                        values.shape[0], dtype=torch.int64, device=device
                    ).reshape(-1, 1),
                ),
                components=components,
                properties=Labels.range(property_name, values.shape[-1]).to(
                    device=device
                ),
            ),
        ],
    )


class TestStressDecomposition:
    """L=0 and L=2 together describe the symmetric part of a 3x3 tensor."""

    def test_trace_l2_norm_and_skew_contracts(self):
        identity = torch.eye(3, dtype=torch.float64)
        diagonal = torch.diag(torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64))
        traceless = torch.tensor(
            [[2.0, 1.0, 0.5], [1.0, -1.0, 0.3], [0.5, 0.3, -1.0]],
            dtype=torch.float64,
        )
        skew = torch.tensor(
            [[0.0, 2.0, -3.0], [-2.0, 0.0, 5.0], [3.0, -5.0, 0.0]],
            dtype=torch.float64,
        )
        matrices = torch.stack([identity, diagonal, traceless, skew]).unsqueeze(-1)
        matrices = torch.cat([matrices, 2.0 * matrices], dim=-1)

        l0 = _l0_components_from_matrices(matrices)
        l2 = _l2_components_from_matrices(matrices)
        assert l0.shape == (4, 1, 2)
        assert l2.shape == (4, 5, 2)
        assert torch.equal(l0[:, :, 1], 2.0 * l0[:, :, 0])
        assert torch.equal(l2[:, :, 1], 2.0 * l2[:, :, 0])
        assert torch.equal(
            l0[:, 0, 0], torch.tensor([3.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        )

        # Identity is purely L=0; the known diagonal tensor has only m=2.
        assert torch.count_nonzero(l2[0]) == 0
        assert l2[1, 2, 0] == 0.0
        assert l2[1, 4, 0] == 1.0

        # The five real components preserve the symmetric-traceless norm in
        # the normalization used by the stress decomposition.
        assert torch.allclose(
            torch.sum(l2[2, :, 0] ** 2),
            0.5 * torch.sum(traceless**2),
            atol=1e-12,
        )
        assert torch.count_nonzero(l2[3]) == 0

    def test_l0_l2_reconstruct_the_symmetric_part(self):
        matrix = torch.randn(1, 3, 3, 1, dtype=torch.float64)
        symmetric = 0.5 * (matrix + matrix.transpose(1, 2))
        l0 = _l0_components_from_matrices(symmetric)
        l2 = _l2_components_from_matrices(symmetric)
        a, b, c, d, e = l2[0, :, 0]
        sqrt3 = np.sqrt(3.0)
        traceless = torch.stack(
            [
                torch.stack([e - c / sqrt3, a, d]),
                torch.stack([a, -e - c / sqrt3, b]),
                torch.stack([d, b, 2 * c / sqrt3]),
            ]
        )
        reconstructed = l0.item() / 3 * torch.eye(3) + traceless
        assert torch.allclose(reconstructed, symmetric[0, :, :, 0], atol=1e-12)


class TestWeightedBatchMoments:
    @pytest.mark.parametrize("component_shape", [(), (2, 3)])
    def test_signed_weights_and_component_reduction(self, component_shape):
        n_systems = 3
        n_samples = 2
        n_properties = 2
        sample_ids = (2**40, 2**40 + 1)
        samples = Labels(
            ["system", "item"],
            torch.tensor(
                [[system, item] for system in range(n_systems) for item in sample_ids],
                dtype=torch.int64,
            ),
        )
        properties = Labels.range("property", n_properties)
        shape = (n_systems * n_samples, *component_shape, n_properties)
        values = torch.arange(int(np.prod(shape)), dtype=torch.float64).reshape(shape)
        components = [
            Labels.range(name, size)
            for name, size in zip(("a", "b"), component_shape, strict=False)
        ]
        tensor = _single_block_tensor_map(
            values,
            samples=samples,
            components=components,
            properties=properties,
            keys=Labels("kind", torch.tensor([[0]], dtype=torch.int64)),
        )
        weights = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
        mean, second, absolute_second, reference = _reduce_weighted_batch_moments(
            tensor, weights, system_index=7
        )
        expected_samples = Labels(
            ["system", "item"],
            torch.tensor([[7, sample_ids[0]], [7, sample_ids[1]]], dtype=torch.int64),
        )
        assert mean.block().samples == expected_samples
        assert second.block().samples == expected_samples
        assert second.block().components == []
        assert mean.block().components == components
        assert mean.block().properties == second.block().properties == properties

        reshaped = values.reshape(n_systems, n_samples, *component_shape, n_properties)
        centered = reshaped - reshaped[0]
        weight_shape = (n_systems,) + (1,) * (centered.ndim - 1)
        assert torch.allclose(
            mean.block().values,
            torch.sum(0.5 * weights.reshape(weight_shape) * centered, dim=0),
        )
        squared = centered**2
        if component_shape:
            squared = squared.sum(dim=tuple(range(2, 2 + len(component_shape))))
        assert torch.allclose(
            second.block().values,
            torch.sum(0.5 * weights.reshape(n_systems, 1, 1) * squared, dim=0),
        )
        assert torch.allclose(
            absolute_second.block().values,
            torch.sum(
                0.5 * torch.abs(weights).reshape(n_systems, 1, 1) * squared,
                dim=0,
            ),
        )
        assert torch.equal(reference.block().values, reshaped[0])


@pytest.mark.parametrize(
    ("sample_values", "message"),
    [
        ([[0, 0], [2, 0]], "out-of-range system indices"),
        ([[0, 0], [0, 1], [1, 0]], "same sample labels"),
        ([[0, 0], [0, 1], [0, 2], [1, 0]], "same sample labels"),
        ([[0, 0], [0, 1], [1, 0], [1, 2]], "same sample labels"),
    ],
)
def test_streaming_layout_rejects_inconsistent_rotated_samples(sample_values, message):
    """Streaming must never mix samples from different rotated copies."""
    samples = Labels(["system", "atom"], torch.tensor(sample_values, dtype=torch.int64))
    block = TensorBlock(
        values=torch.zeros((len(samples), 1), dtype=torch.float64),
        samples=samples,
        components=[],
        properties=Labels.range("p", 1),
    )

    with pytest.raises(ValueError, match=message):
        _reshape_block_by_local_system(block, n_local_systems=2)


@pytest.mark.parametrize("scale", [1.0e-12, 1.0, 1.0e12])
def test_nonnegative_diagnostic_uses_scale_aware_tolerance(scale):
    n_grid_points = 100
    gamma = n_grid_points * torch.finfo(torch.float64).eps
    gamma /= 1.0 - gamma
    tolerance = 64.0 * gamma * scale
    samples = Labels("system", torch.tensor([[0]]))
    properties = Labels.range("p", 1)

    def tensor(value):
        return _single_block_tensor_map(
            torch.tensor([[value]], dtype=torch.float64),
            samples=samples,
            components=[],
            properties=properties,
        )

    cleaned = _validate_nonnegative_diagnostic(
        tensor(-0.5 * tolerance),
        tensor(scale),
        n_grid_points=n_grid_points,
        quantity="variance",
        max_o3_lambda_grid=3,
    )
    assert cleaned.block().values.item() == 0.0

    with pytest.raises(ValueError, match="materially negative"):
        _validate_nonnegative_diagnostic(
            tensor(-2.0 * tolerance),
            tensor(scale),
            n_grid_points=n_grid_points,
            quantity="variance",
            max_o3_lambda_grid=3,
        )


@pytest.mark.parametrize(
    ("value", "scale"),
    [
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (0.0, -1.0),
    ],
)
def test_nonnegative_diagnostic_rejects_invalid_value_or_scale(value, scale):
    samples = Labels("system", torch.tensor([[0]]))
    properties = Labels.range("p", 1)

    def tensor(number):
        return _single_block_tensor_map(
            torch.tensor([[number]], dtype=torch.float64),
            samples=samples,
            components=[],
            properties=properties,
        )

    with pytest.raises(ValueError, match="variance or its error scale is invalid"):
        _validate_nonnegative_diagnostic(
            tensor(value),
            tensor(scale),
            n_grid_points=100,
            quantity="variance",
            max_o3_lambda_grid=3,
        )


class TestQuadratureWignerD:
    """Cross-check the quadrature pole rotations against their generating angles."""

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
    @pytest.mark.parametrize("inverted", [False, True])
    def test_scipy_quadrature_poles_recover_generating_wigner(self, dtype, inverted):
        """Check every polar node for gamma counts selected by degrees 0..16."""
        ell_max = 8
        transforms = {
            ell: _complex_to_real_spherical_harmonics_transform(ell)
            for ell in range(ell_max + 1)
        }
        atol = 2.0e-11 if dtype == torch.float64 else 4.0e-5

        for degree in range(17):
            order, n_rotations = _choose_quadrature(degree)
            alpha, beta, gamma, _ = get_euler_angles_quadrature(order, n_rotations)
            matrices = _rotations_from_angles(alpha, beta, gamma).as_matrix()
            polar_indices = np.flatnonzero(
                (np.abs(beta) < 1.0e-12) | (np.abs(beta - np.pi) < 1.0e-12)
            )
            for index in polar_indices:
                proper = torch.tensor(matrices[index], dtype=dtype)
                matrix = -proper if inverted else proper
                recovered = build_wigner_D_cache(
                    ell_max,
                    matrix,
                    device=matrix.device,
                    dtype=dtype,
                )
                expected = _compute_real_wigner_d_matrices(
                    ell_max,
                    (
                        float(alpha[index]),
                        float(beta[index]),
                        float(gamma[index]),
                    ),
                    transforms,
                )
                for ell in range(ell_max + 1):
                    assert torch.allclose(
                        recovered[ell],
                        expected[ell].to(dtype=dtype),
                        rtol=0.0,
                        atol=atol,
                    ), (
                        f"pole mismatch for degree={degree}, gamma index={index}, "
                        f"ell={ell}, dtype={dtype}, inverted={inverted}"
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

    def test_rotations_are_proper(self):
        """All rotation matrices from the quadrature should have det = +1."""
        lebedev_order, n_inplane = _choose_quadrature(5)
        alpha, beta, gamma, _ = get_euler_angles_quadrature(lebedev_order, n_inplane)
        R = _rotations_from_angles(alpha, beta, gamma)
        matrices = R.as_matrix()
        dets = np.linalg.det(matrices)
        assert np.allclose(dets, 1.0, atol=1e-10)

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

    @pytest.mark.parametrize("ell_max", range(5))
    def test_default_character_grid_resolves_o3_wigner_products_at_boundary(
        self, ell_max
    ):
        """Degree 2L, hence 2L+1 gamma points, resolves products through L.

        The Gram matrix includes both O(3) parities. It therefore checks the
        exact boundary used by character projections, rather than only an
        SO(3) scalar moment.
        """
        model = SymmetrizedModel(
            torch.nn.Identity(),
            max_o3_lambda_target=0,
            max_o3_lambda_character=ell_max,
        )
        assert model.max_o3_lambda_grid == 2 * ell_max

        weights = model._so3_weights_float64
        transformations = [
            O3Transformation(rotation, ell_max)
            for rotation in model.so3_rotations.transpose(-1, -2)
        ]
        columns = []
        expected_diagonal = []
        for ell in range(ell_max + 1):
            wigner = torch.stack(
                [
                    transformation.wigner_D_matrix(ell)
                    for transformation in transformations
                ]
            ).reshape(len(transformations), -1)
            for sigma in (1, -1):
                improper_factor = sigma * (-1) ** ell
                columns.append(torch.cat([wigner, improper_factor * wigner], dim=0))
                expected_diagonal.extend([1.0 / (2 * ell + 1)] * wigner.shape[1])

        basis = torch.cat(columns, dim=1)
        o3_weights = torch.cat([0.5 * weights, 0.5 * weights])
        gram = basis.T @ (o3_weights[:, None] * basis)
        expected = torch.diag(torch.tensor(expected_diagonal, dtype=torch.float64))
        # Wigner matrices are reconstructed from rotation matrices, so their
        # Euler-angle round trip currently limits this check to about 1e-8.
        assert torch.allclose(gram, expected, rtol=0.0, atol=2.0e-8)

    @pytest.mark.parametrize("ell_max", range(5))
    def test_default_target_grid_keeps_conservative_extra_degree(self, ell_max):
        model = SymmetrizedModel(
            torch.nn.Identity(),
            max_o3_lambda_target=ell_max,
        )
        assert model.max_o3_lambda_grid == 2 * ell_max + 1

    def test_rotation_quadrature_matrices(self):
        """get_rotation_quadrature returns orthogonal matrices with normalized
        weights, and doubles the grid with improper partners on request."""
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


_POSITIONS_A = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
_POSITIONS_B = [[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]]


def _two_atom_system(positions=_POSITIONS_A, dtype=torch.float64):
    """The diatomic test system shared by most tests in this module."""
    return System(
        types=torch.tensor([1, 1], dtype=torch.int32),
        positions=torch.tensor(positions, dtype=dtype),
        cell=torch.zeros((3, 3), dtype=dtype),
        pbc=torch.tensor([False, False, False]),
    )


@pytest.mark.parametrize("inversion", [1.0, -1.0])
@pytest.mark.parametrize("n_matrices", [1, 2])
def test_batched_quadrature_system_transform_matches_generic_path(
    inversion, n_matrices
):
    system = System(
        types=torch.tensor([1, 8], dtype=torch.int32),
        positions=torch.tensor(_POSITIONS_A, dtype=torch.float64),
        cell=3.0 * torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )
    options = NeighborListOptions(cutoff=2.0, full_list=True, strict=False)
    neighbors = TensorBlock(
        values=torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float64),
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
    )
    system.add_neighbor_list(options, neighbors)
    field = _single_block_tensor_map(
        torch.tensor([[[0.3], [-0.2], [0.7]]], dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[0]], dtype=torch.int64)),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("field", 1),
    )
    system.add_data("mtt::field", field)

    rotations = torch.tensor(
        [
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    matrices = (inversion * rotations)[:n_matrices]
    actual = _transform_system_batch(
        system,
        matrices,
        0,
        is_inverted=inversion < 0,
    )
    expected = [
        transform_system(system, O3Transformation(matrix, 0)) for matrix in matrices
    ]

    for actual_system, expected_system in zip(actual, expected, strict=True):
        assert torch.equal(actual_system.positions, expected_system.positions)
        assert torch.equal(actual_system.cell, expected_system.cell)
        assert torch.equal(
            actual_system.get_neighbor_list(options).values,
            expected_system.get_neighbor_list(options).values,
        )
        assert torch.equal(
            actual_system.get_data("mtt::field").block().values,
            expected_system.get_data("mtt::field").block().values,
        )


def test_batched_system_transform_reregisters_autograd_neighbors():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    system = System(
        types=torch.tensor([1, 1], dtype=torch.int32),
        positions=positions,
        cell=torch.zeros((3, 3), dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )
    options = NeighborListOptions(cutoff=2.0, full_list=True, strict=False)
    neighbors = TensorBlock(
        values=torch.tensor([[[1.0], [0.0], [0.0]]], dtype=torch.float64),
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            torch.tensor([[0, 1, 0, 0, 0]]),
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )
    register_autograd_neighbors(system, neighbors)
    system.add_neighbor_list(options, neighbors)

    matrices = torch.stack(
        [
            torch.eye(3, dtype=torch.float64),
            torch.tensor(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
        ]
    )
    transformed = _transform_system_batch(system, matrices, 0, is_inverted=False)
    loss = sum(
        torch.sum(item.get_neighbor_list(options).values ** 2) for item in transformed
    )

    gradient = torch.autograd.grad(loss, positions)[0]
    expected = torch.tensor([[-4.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(gradient, expected, atol=1e-12)


class _QuadraticEnergyModel(torch.nn.Module):
    """Minimal model where E = sum(positions^2). Analytical forces = -2*positions."""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        energies = torch.stack([torch.sum(system.positions**2) for system in systems])
        device = energies.device
        block = TensorBlock(
            values=energies.unsqueeze(-1),
            samples=Labels(
                ["system"],
                torch.arange(len(systems), dtype=torch.int64, device=device).reshape(
                    -1, 1
                ),
            ),
            components=[],
            properties=Labels(
                ["energy"], torch.tensor([[0]], dtype=torch.int64, device=device)
            ),
        )
        return {
            "energy": TensorMap(
                Labels(["_"], torch.tensor([[0]], dtype=torch.int64, device=device)),
                [block],
            )
        }

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return []


class _ConstantEnergyModel(torch.nn.Module):
    """Minimal model returning an energy independent of positions and cell."""

    def __init__(
        self,
        connected: bool = False,
        connect_first_only: bool = False,
        value: float = 1.0,
    ):
        super().__init__()
        self.connected = connected
        self.connect_first_only = connect_first_only
        self.value = value

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        energies = []
        for system_i, system in enumerate(systems):
            energy = torch.full(
                (),
                self.value,
                dtype=system.positions.dtype,
                device=system.positions.device,
            )
            if self.connected and (not self.connect_first_only or system_i == 0):
                # Keep a graph connection while remaining independent of values.
                energy = energy + 0.0 * system.positions.sum()
            energies.append(energy)

        return {"energy": _system_tensor_map(torch.stack(energies).unsqueeze(-1))}

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return []


class _ParameterOnlyEnergyModel(_ConstantEnergyModel):
    """Energy with an autograd graph that does not use positions or cell."""

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        return {"energy": _system_tensor_map(self.bias.expand(len(systems), 1))}


class _ExplicitGradientEnergyModel(torch.nn.Module):
    """Quadratic energy that attaches an unsolicited positions gradient."""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        energies = torch.stack([torch.sum(system.positions**2) for system in systems])
        device = energies.device
        properties = Labels(
            ["energy"], torch.tensor([[0]], dtype=torch.int64, device=device)
        )
        block = TensorBlock(
            values=energies.unsqueeze(-1),
            samples=Labels(
                ["system"],
                torch.arange(len(systems), dtype=torch.int64, device=device).reshape(
                    -1, 1
                ),
            ),
            components=[],
            properties=properties,
        )
        sample_index = torch.arange(len(systems), dtype=torch.int64, device=device)
        block.add_gradient(
            "positions",
            TensorBlock(
                values=torch.zeros(
                    (len(systems), 3, 1), dtype=energies.dtype, device=device
                ),
                samples=Labels(
                    ["sample", "system", "atom"],
                    torch.stack(
                        [sample_index, sample_index, torch.zeros_like(sample_index)],
                        dim=1,
                    ),
                ),
                components=[Labels.range("xyz", 3).to(device=device)],
                properties=properties,
            ),
        )
        return {
            "energy": TensorMap(
                Labels(["_"], torch.tensor([[0]], dtype=torch.int64, device=device)),
                [block],
            )
        }

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return []


class _RenamedEnergyModel(_QuadraticEnergyModel):
    """Same as _QuadraticEnergyModel, but with a non-standard output name."""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        out = super().forward(systems, outputs, selected_atoms)
        return {"mtt::energy": out["energy"]}


class _OffsetAnisotropicEnergyModel(torch.nn.Module):
    """Energy = 1e5 + sum(positions^2) + sum(positions[:, 0]).

    Like _AnisotropicEnergyModel (exact O(3) variance = |sum of positions|^2 / 3),
    but with a large invariant offset: in float32, the offset makes the
    second-moment variance estimator cancel catastrophically unless the
    statistics are accumulated in float64.
    """

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        energies = [
            1.0e5 + torch.sum(sys.positions**2) + torch.sum(sys.positions[:, 0])
            for sys in systems
        ]
        return {"energy": _system_tensor_map(torch.stack(energies).unsqueeze(-1))}

    def requested_neighbor_lists(self):
        return []


class _DegreeSevenEnergyModel(torch.nn.Module):
    """Smooth response whose degree-12 grid variance is materially aliased."""

    def forward(self, systems, outputs, selected_atoms=None):
        energies = []
        for system in systems:
            x, y, z = system.positions[0]
            fourth_order = 0.625 * (x**4 + y**4 + z**4)
            mixed = x**2 * y**2 + x**2 * z**2 + y**2 * z**2
            energies.append(1000.0 * x * y * z * (fourth_order - mixed))
        return {"energy": _system_tensor_map(torch.stack(energies).unsqueeze(-1))}

    def requested_neighbor_lists(self):
        return []


class _EnergyAndVectorModel(torch.nn.Module):
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        result = {}

        if "energy" in outputs:
            energies = torch.stack(
                [torch.sum(system.positions**2) for system in systems]
            )
            result["energy"] = _system_tensor_map(energies.unsqueeze(-1))

        if "non_conservative_forces" in outputs:
            values = torch.cat([sys.positions.unsqueeze(-1) for sys in systems], dim=0)
            samples = []
            for i_sys, sys in enumerate(systems):
                for atom in range(len(sys)):
                    samples.append([i_sys, atom])
            force_tmap = _single_block_tensor_map(
                values,
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor(samples, dtype=torch.int64),
                ),
                components=[
                    Labels(
                        names=["xyz"],
                        values=torch.arange(3, dtype=torch.int64).reshape(-1, 1),
                    )
                ],
                properties=Labels(
                    names=["p"],
                    values=torch.tensor([[0]], dtype=torch.int64),
                ),
            )
            if selected_atoms is not None:
                force_tmap = mts.slice(
                    force_tmap,
                    axis="samples",
                    selection=selected_atoms,
                )
            result["non_conservative_forces"] = force_tmap

        return result

    def requested_neighbor_lists(self):
        return []


class _NonLeadingSystemSampleModel(torch.nn.Module):
    """Return interleaved per-atom rows with ``system`` not in column zero."""

    def forward(self, systems, outputs, selected_atoms=None):
        values = torch.stack(
            [
                torch.sum(system.positions[atom] ** 2)
                for atom in range(len(systems[0]))
                for system in systems
            ]
        ).unsqueeze(-1)
        samples = [
            [atom, system_index]
            for atom in range(len(systems[0]))
            for system_index in range(len(systems))
        ]
        return {
            "mtt::feature": _single_block_tensor_map(
                values,
                samples=Labels(
                    ["atom", "system"],
                    torch.tensor(samples, dtype=torch.int64, device=values.device),
                ),
                components=[],
                properties=Labels.range("feature", 1).to(device=values.device),
            )
        }

    def requested_neighbor_lists(self):
        return []


class _CustomUnitInputEnergyModel(torch.nn.Module):
    """Return the sum of a custom mass input requested in kilograms."""

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        return {"mass": ModelOutput(unit="kg", sample_kind="atom")}

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        energies = [system.get_data("mass").block().values.sum() for system in systems]
        return {"energy": _system_tensor_map(torch.stack(energies).reshape(-1, 1))}

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return []


class _AnisotropicEnergyModel(torch.nn.Module):
    """Energy = sum(positions^2) + sum(positions[:, 0]).

    As a function of the O(3) transformation applied to the input positions, the
    output is band-limited: an invariant part (lambda=0) plus a vector part
    (lambda=1, odd under inversion). This makes exact statements about its
    character projections possible.
    """

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        energies = [
            torch.sum(sys.positions**2) + torch.sum(sys.positions[:, 0])
            for sys in systems
        ]
        return {"energy": _system_tensor_map(torch.stack(energies).unsqueeze(-1))}

    def requested_neighbor_lists(self):
        return []


class TestGradientForces:
    """Test conservative forces from autograd via _evaluate_with_gradients."""

    @staticmethod
    def _system(*, pbc=(False, False, False)):
        positions = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float64
        )
        return System(
            types=torch.tensor([1, 1]),
            positions=positions,
            cell=5.0 * torch.diag(torch.tensor(pbc, dtype=torch.float64)),
            pbc=torch.tensor(pbc),
        )

    @pytest.mark.parametrize(
        "periodic",
        [False, True],
    )
    def test_quadratic_derivatives_for_rotation_batch(self, periodic):
        """Autograd returns exact rotated-frame forces and periodic stress."""
        pbc = (periodic,) * 3
        system = self._system(pbc=pbc)
        rotations = torch.cat(
            [
                torch.eye(3, dtype=torch.float64).unsqueeze(0),
                torch.tensor(
                    Rotation.random(2, random_state=7).as_matrix(),
                    dtype=torch.float64,
                ),
            ]
        )
        result = _evaluate_gradients(_QuadraticEnergyModel(), system, rotations)

        n_atoms = len(system)
        forces = result["forces"].block().values.squeeze(-1)
        expected_forces = torch.cat(
            [-2.0 * (system.positions @ rotation.T) for rotation in rotations]
        )
        assert torch.allclose(forces, expected_forces, atol=1e-12)
        assert result["forces"].block().samples.names == ["system", "atom"]
        assert result["forces"].block().samples.values.tolist() == [
            [rotation, atom]
            for rotation in range(len(rotations))
            for atom in range(n_atoms)
        ]

        if not periodic:
            assert "stress" not in result
            return

        stresses = result["stress"].block().values.squeeze(-1)
        volume = torch.abs(torch.linalg.det(system.cell))
        expected_stresses = torch.stack(
            [
                2.0
                * (system.positions @ rotation.T).T
                @ (system.positions @ rotation.T)
                / volume
                for rotation in rotations
            ]
        )
        assert torch.allclose(stresses, expected_stresses, atol=1e-12)

    @pytest.mark.parametrize(
        ("pbc", "cell"),
        [
            ([False, False, False], torch.zeros(3, 3)),
            ([True, False, False], torch.diag(torch.tensor([5.0, 0.0, 0.0]))),
            ([True, True, False], torch.diag(torch.tensor([5.0, 5.0, 0.0]))),
        ],
    )
    def test_stress_is_omitted_without_three_periodic_dimensions(self, pbc, cell):
        positions = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64)
        system = System(
            types=torch.tensor([1]),
            positions=positions,
            cell=cell.to(dtype=torch.float64),
            pbc=torch.tensor(pbc),
        )
        result = _evaluate_gradients(_QuadraticEnergyModel(), system)

        assert "stress" not in result
        forces = result["forces"].block().values.squeeze(-1)
        assert torch.allclose(forces, -2.0 * positions, atol=1e-12)

    @pytest.mark.parametrize(
        "cell",
        [
            torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 5.0, 0.0]]),
            torch.diag(torch.tensor([float("nan"), 5.0, 5.0])),
            torch.diag(torch.tensor([float("inf"), 5.0, 5.0])),
        ],
    )
    def test_invalid_fully_periodic_cell_is_rejected(self, cell):
        system = System(
            types=torch.tensor([1]),
            positions=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64),
            cell=cell.to(dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        )

        with pytest.raises(ValueError, match="singular or non-finite"):
            _evaluate_gradients(_QuadraticEnergyModel(), system)

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(_ConstantEnergyModel(connected=True), id="connected"),
            pytest.param(
                _ConstantEnergyModel(connected=True, connect_first_only=True),
                id="partly-connected-batch",
            ),
            pytest.param(_ParameterOnlyEnergyModel(), id="parameter-only"),
        ],
    )
    def test_coordinate_independent_energy_has_zero_derivatives(self, model):
        result = _evaluate_gradients(
            model,
            self._system(pbc=(True, True, True)),
            torch.eye(3, dtype=torch.float64).repeat(2, 1, 1),
        )

        for name in ("forces", "stress"):
            values = result[name].block().values
            assert torch.equal(values, torch.zeros_like(values))

    def test_detached_constant_energy_has_zero_forces(self):
        model = _symmetrized(_ConstantEnergyModel(), batch_size=2)
        with pytest.warns(RuntimeWarning, match="detached.*zero derivatives"):
            out = model(
                [_two_atom_system()],
                {"energy": ModelOutput(sample_kind="system")},
                compute_gradients=True,
            )

        assert torch.count_nonzero(out["forces_l1_mean"].block().values) == 0


class TestSymmetrizedModelForward:
    def _make_system(self, dtype=torch.float64):
        return _two_atom_system(dtype=dtype)

    def _make_second_system(self, dtype=torch.float64):
        return _two_atom_system(_POSITIONS_B, dtype=dtype)

    def test_scalar_forward_outputs(self):
        model = _symmetrized(
            _QuadraticEnergyModel(),
            max_o3_lambda_character=1,
            max_o3_lambda_target=0,
            batch_size=2,
        )

        result = model([self._make_system()], _ENERGY_OUTPUTS)

        assert "energy_l0_mean" in result
        assert "energy_l0_var" in result
        assert "energy_l0_norm_squared" in result
        assert result["energy_l0_mean"].keys.names == ["o3_lambda", "o3_sigma"]
        assert torch.allclose(
            result["energy_l0_mean"].block().values,
            torch.tensor([[[5.0]]], dtype=torch.float64),
            atol=1e-10,
        )

    def test_forward_character_projections(self):
        model = _symmetrized(
            _QuadraticEnergyModel(),
            max_o3_lambda_character=1,
            max_o3_lambda_target=0,
            max_o3_lambda_grid=2,
            batch_size=1,
        )

        result = model(
            [self._make_system()],
            _ENERGY_OUTPUTS,
            compute_character_projections=True,
        )

        expected_keys = {
            "energy_l0_mean",
            "energy_l0_var",
            "energy_l0_norm_squared",
            "energy_l0_character_projection",
        }
        assert set(result.keys()) == expected_keys

    def test_character_projections_without_gradients_run_under_no_grad(self):
        class _GradStateModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.recorded_grad_states = []

            def forward(self, systems, outputs, selected_atoms):
                self.recorded_grad_states.append(torch.is_grad_enabled())
                values = torch.tensor(
                    [[1.0]],
                    dtype=systems[0].positions.dtype,
                    device=systems[0].positions.device,
                )
                return {"energy": _system_tensor_map(values)}

        base_model = _GradStateModel()
        model = _symmetrized(
            base_model,
            max_o3_lambda_character=1,
            max_o3_lambda_target=0,
            batch_size=1,
        )

        result = model(
            [self._make_system()],
            _ENERGY_OUTPUTS,
            compute_character_projections=True,
            compute_gradients=False,
        )

        assert "energy_l0_character_projection" in result
        assert len(base_model.recorded_grad_states) > 0
        assert all(state is False for state in base_model.recorded_grad_states)

    @pytest.mark.parametrize("compute_gradients", [False, True])
    def test_storage_device_does_not_change_outputs(self, compute_gradients):
        # Regression: setting storage_device must only move tensors, not
        # change numerical results, in both evaluation modes (outputs are
        # detached before back-rotation, so offloading is safe with gradients)
        systems = [self._make_system()]

        results = {}
        for storage_device in (None, "cpu"):
            base_model = _QuadraticEnergyModel()
            model = _symmetrized(
                base_model,
                max_o3_lambda_character=1,
                max_o3_lambda_target=1,
                batch_size=2,
                storage_device=storage_device,
            )
            results[storage_device] = model(
                systems,
                _ENERGY_OUTPUTS,
                compute_character_projections=not compute_gradients,
                compute_gradients=compute_gradients,
            )

        shared_keys = set(results[None].keys()) & set(results["cpu"].keys())
        assert shared_keys, "no shared output keys between storage modes"
        for name in shared_keys:
            tensor_false = results[None][name]
            tensor_true = results["cpu"][name]
            assert tensor_false.keys == tensor_true.keys
            for key in tensor_false.keys:
                block_false = tensor_false.block(key)
                block_true = tensor_true.block(key)
                assert torch.allclose(
                    block_false.values.cpu(),
                    block_true.values.cpu(),
                    atol=1e-12,
                ), f"storage device changed values for '{name}' / key {key}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    @pytest.mark.parametrize("compute_gradients", [False, True])
    def test_cuda_evaluation_can_offload_results_to_cpu(self, compute_gradients):
        """Exercise the split-device path for values, Wigner work, and gradients."""
        system = self._make_system().to(device="cuda")
        results = {}
        for storage_device in (None, "cpu"):
            model = _symmetrized(
                _QuadraticEnergyModel(),
                max_o3_lambda_character=1,
                max_o3_lambda_target=1,
                batch_size=2,
                storage_device=storage_device,
            ).to(device="cuda")
            results[storage_device] = model(
                [system],
                _ENERGY_OUTPUTS,
                compute_character_projections=not compute_gradients,
                compute_gradients=compute_gradients,
            )

        assert set(results[None]) == set(results["cpu"])
        for name, cpu_tensor in results["cpu"].items():
            cuda_tensor = results[None][name]
            assert cuda_tensor.keys == cpu_tensor.keys
            for cuda_block, cpu_block in zip(
                cuda_tensor.blocks(), cpu_tensor.blocks(), strict=True
            ):
                assert cuda_block.values.device.type == "cuda"
                assert cpu_block.values.device.type == "cpu"
                assert torch.allclose(
                    cuda_block.values.cpu(), cpu_block.values, atol=1e-11
                )

    def test_compute_gradients_produces_exact_forces_and_periodic_stress(self):
        positions = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float64
        )
        cell = torch.tensor(
            [[3.0, 0.0, 0.0], [0.2, 4.0, 0.0], [0.1, 0.3, 5.0]],
            dtype=torch.float64,
        )
        periodic = System(
            types=torch.tensor([1, 1]),
            positions=positions,
            cell=cell,
            pbc=torch.tensor([True, True, True]),
        )
        nonperiodic = self._make_second_system()
        model = _symmetrized(
            _QuadraticEnergyModel(),
            max_o3_lambda_character=1,
            batch_size=2,
        )

        result = model([periodic, nonperiodic], _ENERGY_OUTPUTS, compute_gradients=True)

        forces = result["forces_l1_mean"].block()
        expected_forces = torch.cat(
            [-2.0 * periodic.positions, -2.0 * nonperiodic.positions]
        ).roll(-1, dims=1)
        assert torch.equal(
            forces.samples.values,
            torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]),
        )
        assert torch.allclose(forces.values.squeeze(-1), expected_forces, atol=1e-11)

        expected_stress = (
            2.0 * positions.T @ positions / torch.abs(torch.linalg.det(cell))
        )
        expected_stress = expected_stress.reshape(1, 3, 3, 1)
        stress_l0 = result["stress_l0_mean"].block()
        stress_l2 = result["stress_l2_mean"].block()
        assert torch.equal(stress_l0.samples.values, torch.tensor([[0]]))
        assert torch.equal(stress_l2.samples.values, torch.tensor([[0]]))
        assert torch.allclose(
            stress_l0.values,
            _l0_components_from_matrices(expected_stress),
            atol=1e-11,
        )
        assert torch.allclose(
            stress_l2.values,
            _l2_components_from_matrices(expected_stress),
            atol=1e-11,
        )

        for name in ("forces_l1_var", "stress_l0_var", "stress_l2_var"):
            assert torch.max(torch.abs(result[name].block().values)) < 1e-11

    def test_compute_gradients_omits_3d_stress_for_partial_pbc(self):
        system = System(
            types=torch.tensor([1]),
            positions=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64),
            cell=torch.diag(torch.tensor([5.0, 5.0, 0.0], dtype=torch.float64)),
            pbc=torch.tensor([True, True, False]),
        )
        model = _symmetrized(_QuadraticEnergyModel())

        result = model(
            [system],
            _ENERGY_OUTPUTS,
            compute_gradients=True,
        )

        assert "forces_l1_mean" in result
        assert torch.all(torch.isfinite(result["forces_l1_mean"].block().values))
        assert all(not name.startswith("stress_") for name in result)

    def test_vector_like_forward_outputs(self):
        model = _symmetrized(
            _EnergyAndVectorModel(),
            max_o3_lambda_character=1,
            batch_size=2,
        )

        outputs = {
            "energy": ModelOutput(sample_kind="system"),
            "non_conservative_forces": ModelOutput(sample_kind="atom"),
        }
        result = model([self._make_system()], outputs)

        assert "non_conservative_forces_l1_mean" in result
        assert "non_conservative_forces_l1_var" in result
        assert "non_conservative_forces_l1_norm_squared" in result

    def test_selected_atoms_are_mapped_per_outer_system(self):
        systems = [self._make_system(), self._make_second_system()]
        model = _symmetrized(
            _EnergyAndVectorModel(),
            max_o3_lambda_character=1,
            batch_size=5,
        )

        outputs = {
            "energy": ModelOutput(sample_kind="system"),
            "non_conservative_forces": ModelOutput(sample_kind="atom"),
        }
        selected_atoms = Labels(
            names=["system", "atom"],
            values=torch.tensor([[0, 0], [1, 1]], dtype=torch.int64),
        )

        result = model(systems, outputs, selected_atoms=selected_atoms)

        energy_block = result["energy_l0_mean"].block()
        assert energy_block.samples.values.tolist() == [[0], [1]]
        assert torch.allclose(
            energy_block.values[:, 0, 0],
            torch.tensor([5.0, 25.0], dtype=torch.float64),
            atol=1e-10,
        )

        force_block = result["non_conservative_forces_l1_mean"].block()
        assert force_block.samples.values.tolist() == [[0, 0], [1, 1]]
        assert torch.allclose(
            force_block.values.roll(1, 1).squeeze(-1),
            torch.stack([systems[0].positions[0], systems[1].positions[1]]),
            atol=1e-10,
        )

    def test_selected_atoms_can_be_empty_for_some_systems(self):
        systems = [self._make_system(), self._make_second_system()]
        model = _symmetrized(
            _EnergyAndVectorModel(),
            max_o3_lambda_character=1,
            batch_size=5,
        )

        outputs = {
            "non_conservative_forces": ModelOutput(sample_kind="atom"),
        }
        selected_atoms = Labels(
            names=["system", "atom"],
            values=torch.tensor([[1, 1]], dtype=torch.int64),
        )

        result = model(systems, outputs, selected_atoms=selected_atoms)

        force_block = result["non_conservative_forces_l1_mean"].block()
        assert force_block.samples.values.tolist() == [[1, 1]]
        assert torch.allclose(
            force_block.values.roll(1, 1).squeeze(-1),
            systems[1].positions[1].unsqueeze(0),
            atol=1e-10,
        )


class TestCharacterProjectionValidation:
    def _make_system(self):
        return _two_atom_system()

    def test_equivariance_only_without_character_lambda(self):
        # the common use case: max_o3_lambda_character can be omitted entirely,
        # and the default grid then follows the target angular momentum
        model = _symmetrized(
            _QuadraticEnergyModel(),
            max_o3_lambda_target=0,
        )
        assert model.max_o3_lambda_grid == 1  # 2 * max_o3_lambda_target + 1

        result = model([self._make_system()], _ENERGY_OUTPUTS)
        assert set(result.keys()) == {
            "energy_l0_mean",
            "energy_l0_var",
            "energy_l0_norm_squared",
        }

    def test_character_projections_require_character_lambda(self):
        model = _symmetrized(
            _QuadraticEnergyModel(),
            max_o3_lambda_target=0,
        )

        with pytest.raises(ValueError, match="max_o3_lambda_character must be set"):
            model(
                [self._make_system()],
                _ENERGY_OUTPUTS,
                compute_character_projections=True,
            )

    def test_character_projections_insufficient_grid_raises(self):
        # a grid unable to integrate the projections exactly produces silently
        # wrong results (isotypical fractions far above 1), so it must be
        # rejected when projections are requested, and only then
        model = _symmetrized(
            _QuadraticEnergyModel(),
            max_o3_lambda_target=0,
            max_o3_lambda_character=1,
            max_o3_lambda_grid=1,  # K=2 gamma points, but products require 3
        )

        with pytest.raises(ValueError, match="too coarse for character projections"):
            model(
                [self._make_system()],
                _ENERGY_OUTPUTS,
                compute_character_projections=True,
            )

        result = model([self._make_system()], _ENERGY_OUTPUTS)
        assert "energy_l0_mean" in result


class TestExplicitGradientRejection:
    @pytest.mark.parametrize("compute_gradients", [False, True])
    def test_requested_explicit_gradient_is_rejected_before_evaluation(
        self, compute_gradients
    ):
        class _CountingModel(_QuadraticEnergyModel):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def forward(self, systems, outputs, selected_atoms=None):
                self.call_count += 1
                return super().forward(systems, outputs, selected_atoms)

        base_model = _CountingModel()
        model = SymmetrizedModel(base_model, max_o3_lambda_target=1)
        outputs = {
            "energy": ModelOutput(
                sample_kind="system", explicit_gradients=["positions"]
            )
        }

        with pytest.raises(ValueError, match="does not support explicit gradients"):
            model(
                [_two_atom_system()],
                outputs,
                compute_gradients=compute_gradients,
            )
        assert base_model.call_count == 0

    @pytest.mark.parametrize("compute_gradients", [False, True])
    def test_unsolicited_attached_gradient_is_rejected(self, compute_gradients):
        model = SymmetrizedModel(_ExplicitGradientEnergyModel(), max_o3_lambda_target=1)
        with pytest.raises(
            ValueError,
            match="output 'energy' contains explicit gradient 'positions'",
        ):
            model(
                [_two_atom_system()],
                {"energy": ModelOutput(sample_kind="system")},
                compute_gradients=compute_gradients,
            )


class TestBaseOutputValidation:
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_non_leading_system_sample_column(self, batch_size):
        model = _symmetrized(
            _NonLeadingSystemSampleModel(),
            max_o3_lambda_target=0,
            batch_size=batch_size,
        )
        result = model(
            [_two_atom_system()],
            {"mtt::feature": ModelOutput(sample_kind="atom")},
        )

        block = result["mtt::feature_mean"].block()
        assert block.samples.names == ["system", "atom"]
        assert block.samples.values.tolist() == [[0, 0], [0, 1]]

    @pytest.mark.parametrize("compute_gradients", [False, True])
    def test_missing_requested_output_is_rejected(self, compute_gradients):
        model = SymmetrizedModel(
            _QuadraticEnergyModel(), max_o3_lambda_target=1, batch_size=4
        )
        with pytest.raises(
            ValueError,
            match="did not return requested output.*mtt::missing",
        ):
            model(
                [_two_atom_system()],
                {
                    "energy": ModelOutput(sample_kind="system"),
                    "mtt::missing": ModelOutput(sample_kind="system"),
                },
                compute_gradients=compute_gradients,
            )

    @pytest.mark.parametrize(
        ("colliding_name", "compute_gradients"),
        [("energy_l0", False), ("forces_l1", True)],
    )
    def test_decomposed_output_name_collisions_are_rejected(
        self, colliding_name, compute_gradients
    ):
        model = _symmetrized(_ConstantEnergyModel(), batch_size=4)
        outputs = {
            "energy": ModelOutput(sample_kind="system"),
            colliding_name: ModelOutput(sample_kind="system"),
        }

        with pytest.raises(ValueError, match=f"both produce '{colliding_name}'"):
            model(
                [_two_atom_system()],
                outputs,
                compute_gradients=compute_gradients,
            )


class TestConstructorValidation:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("max_o3_lambda_target", 1.0),
            ("max_o3_lambda_target", True),
            ("max_o3_lambda_character", 1.0),
            ("max_o3_lambda_character", True),
            ("max_o3_lambda_grid", 1.0),
            ("max_o3_lambda_grid", True),
            ("batch_size", 1.0),
            ("batch_size", True),
            ("wigner_cache_max_bytes", 1.0),
            ("wigner_cache_max_bytes", True),
        ],
    )
    def test_rejects_non_integer_values(self, name, value):
        arguments = {"max_o3_lambda_target": 0, name: value}
        with pytest.raises(TypeError, match=f"{name} must be an integer"):
            SymmetrizedModel(_QuadraticEnergyModel(), **arguments)

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("max_o3_lambda_target", -1),
            ("max_o3_lambda_character", -1),
            ("max_o3_lambda_grid", -1),
            ("batch_size", 0),
            ("batch_size", -1),
            ("wigner_cache_max_bytes", -1),
        ],
    )
    def test_rejects_out_of_range_values(self, name, value):
        arguments = {"max_o3_lambda_target": 0, name: value}
        with pytest.raises(ValueError, match=name):
            SymmetrizedModel(_QuadraticEnergyModel(), **arguments)

    def test_accepts_numpy_integer_values(self):
        model = SymmetrizedModel(
            _QuadraticEnergyModel(),
            max_o3_lambda_target=np.int64(0),
            max_o3_lambda_character=np.int64(1),
            max_o3_lambda_grid=np.int64(3),
            batch_size=np.int64(2),
            wigner_cache_max_bytes=np.int64(1024),
        )
        assert model.max_o3_lambda_target == 0
        assert model.max_o3_lambda_character == 1
        assert model.max_o3_lambda_grid == 3
        assert model.batch_size == 2
        assert model.wigner_cache_max_bytes == 1024


class TestMPSPolicy:
    def test_storage_device_rejects_before_quadrature(self, monkeypatch):
        def quadrature_must_not_run(*args, **kwargs):
            raise AssertionError("quadrature construction was reached")

        monkeypatch.setattr(
            symmetrized_model_module,
            "_choose_quadrature",
            quadrature_must_not_run,
        )
        with pytest.raises(ValueError, match="does not support MPS"):
            SymmetrizedModel(
                _ConstantEnergyModel(),
                max_o3_lambda_target=0,
                storage_device="mps",
            )

    def test_to_mps_rejects_before_cache_or_subtree_mutation(self):
        model = SymmetrizedModel(
            _ConstantEnergyModel(),
            max_o3_lambda_target=0,
            max_o3_lambda_character=1,
            batch_size=4,
        ).to(dtype=torch.float64)
        model(
            [_two_atom_system()],
            {"energy": ModelOutput(sample_kind="system")},
            compute_character_projections=True,
        )
        before_rotations = model.so3_rotations.clone()
        before_cache = model._wigner_cache

        with pytest.raises(ValueError, match="does not support MPS"):
            model.to("mps")

        assert torch.equal(model.so3_rotations, before_rotations)
        assert model._wigner_cache is before_cache

    @pytest.mark.skipif(not can_use_mps_backend(), reason="MPS is not available")
    def test_mixed_base_state_rejects_before_quadrature(self, monkeypatch):
        class MixedStateModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cpu_parameter = torch.nn.Parameter(torch.ones(1))
                self.register_buffer("mps_buffer", torch.ones(1, device="mps"))

        def quadrature_must_not_run(*args, **kwargs):
            raise AssertionError("quadrature construction was reached")

        monkeypatch.setattr(
            symmetrized_model_module,
            "_choose_quadrature",
            quadrature_must_not_run,
        )
        with pytest.raises(ValueError, match="does not support MPS"):
            SymmetrizedModel(MixedStateModel(), max_o3_lambda_target=0)

    @pytest.mark.skipif(not can_use_mps_backend(), reason="MPS is not available")
    def test_mps_system_rejects_before_model_call(self):
        class CountingModel(_ConstantEnergyModel):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, *args, **kwargs):
                self.calls += 1
                return super().forward(*args, **kwargs)

        base = CountingModel()
        model = SymmetrizedModel(base, max_o3_lambda_target=0)
        system = _two_atom_system(dtype=torch.float32).to(device="mps")

        with pytest.raises(ValueError, match="does not support MPS"):
            model([system], {"energy": ModelOutput(sample_kind="system")})
        assert base.calls == 0


class TestPerSystemHelpers:
    def _make_systems(self):
        return [_two_atom_system(), _two_atom_system(_POSITIONS_B)]

    def test_character_fractions_capture_band_limited_output(self):
        # the energy of _AnisotropicEnergyModel has content only at lambda=0 and
        # lambda=1 (both in the proper chi_sigma=+1 sector), so with a sufficient
        # grid the fractions must sum to 1 and vanish everywhere else
        model = _symmetrized(
            _AnisotropicEnergyModel(),
            max_o3_lambda_target=0,
            max_o3_lambda_character=3,
            batch_size=64,
        )

        result = model(
            self._make_systems(), _ENERGY_OUTPUTS, compute_character_projections=True
        )

        proper, improper, lambdas = per_system_character_fractions(result, "energy_l0")

        assert lambdas.tolist() == [0, 1, 2, 3]
        assert proper.shape == (2, 4)
        total = (proper + improper).sum(dim=1)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-8)
        assert torch.all(improper.abs() < 1e-8)
        assert torch.all(proper[:, 2:].abs() < 1e-8)

    def test_negative_manual_diagnostics_are_rejected(self):
        samples = Labels("system", torch.tensor([[0]]))
        properties = Labels.range("p", 1)

        def single(value, keys=None):
            return _single_block_tensor_map(
                torch.tensor([[value]], dtype=torch.float64),
                samples=samples,
                components=[],
                properties=properties,
                keys=keys,
            )

        character_keys = Labels(["chi_lambda", "chi_sigma"], torch.tensor([[0, 1]]))
        with pytest.raises(ValueError, match="squared norm"):
            per_system_character_fractions(
                {
                    "x_character_projection": single(1.0, character_keys),
                    "x_norm_squared": single(-1.0),
                },
                "x",
            )
        with pytest.raises(ValueError, match="character-projection"):
            per_system_character_fractions(
                {
                    "x_character_projection": single(-1.0, character_keys),
                    "x_norm_squared": single(1.0),
                },
                "x",
            )
        with pytest.raises(ValueError, match="zero squared norm"):
            per_system_character_fractions(
                {
                    "x_character_projection": single(1.0, character_keys),
                    "x_norm_squared": single(0.0),
                },
                "x",
            )
        cancelling_norm = _single_block_tensor_map(
            torch.tensor([[-1.0, 2.0]], dtype=torch.float64),
            samples=samples,
            components=[],
            properties=Labels.range("p", 2),
        )
        cancelling_projection = _single_block_tensor_map(
            torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            samples=samples,
            components=[],
            properties=Labels.range("p", 2),
            keys=character_keys,
        )
        with pytest.raises(ValueError, match="squared norm"):
            per_system_character_fractions(
                {
                    "x_character_projection": cancelling_projection,
                    "x_norm_squared": cancelling_norm,
                },
                "x",
            )
        with pytest.raises(ValueError, match="variance values"):
            per_system_equivariance_rmse(
                {"x_var": single(-1.0), "x_mean": single(0.0)}, "x"
            )

    def test_character_fractions_support_empty_samples(self):
        samples = Labels("system", torch.empty((0, 1), dtype=torch.int64))
        properties = Labels.range("p", 1)
        character_keys = Labels(["chi_lambda", "chi_sigma"], torch.tensor([[0, 1]]))
        outputs = {
            "x_norm_squared": _single_block_tensor_map(
                torch.empty((0, 1), dtype=torch.float64),
                samples=samples,
                components=[],
                properties=properties,
            ),
            "x_character_projection": _single_block_tensor_map(
                torch.empty((0, 3, 1), dtype=torch.float64),
                samples=samples,
                components=[Labels.range("o3_mu", 3)],
                properties=properties,
                keys=character_keys,
            ),
        }

        proper, improper, lambdas = per_system_character_fractions(
            outputs, "x", n_systems=2
        )

        assert lambdas.tolist() == [0]
        assert proper.shape == improper.shape == (2, 1)
        assert torch.count_nonzero(proper) == torch.count_nonzero(improper) == 0

    def test_equivariance_rmse_vanishes_for_equivariant_output(self):
        # forces of _EnergyAndVectorModel are the (rotated) positions, which
        # back-rotate exactly: the equivariance RMSE must be zero per system
        model = _symmetrized(
            _EnergyAndVectorModel(),
            batch_size=8,
        )

        outputs = {"non_conservative_forces": ModelOutput(sample_kind="atom")}
        result = model(self._make_systems(), outputs)

        rmse = per_system_equivariance_rmse(result, "non_conservative_forces_l1")
        assert rmse.block().values.shape == (2, 1)
        # tolerance above the float64 cancellation floor of the variance,
        # sqrt(eps * max ||x||^2), which summation order can push past 1e-8
        assert torch.all(rmse.block().values.abs() < 1e-7)

    def test_equivariance_rmse_reduction(self):
        # RMSE = sqrt( mean over a system's samples of
        #              (component-summed variance) / (2l+1) ),
        # with 2l+1 read from the components of the matching _mean tensor
        keys = Labels(["_"], torch.tensor([[0]], dtype=torch.int64))
        samples = Labels(
            ["system", "atom"],
            torch.tensor([[0, 0], [0, 1], [1, 0]], dtype=torch.int64),
        )
        properties = Labels(["p"], torch.tensor([[0]], dtype=torch.int64))
        variance = _single_block_tensor_map(
            torch.tensor([[6.0], [15.0], [24.0]], dtype=torch.float64),
            samples=samples,
            components=[],
            properties=properties,
            keys=keys,
        )
        mean = _single_block_tensor_map(
            torch.zeros((3, 3, 1), dtype=torch.float64),
            samples=samples,
            components=[
                Labels(
                    ["o3_mu"],
                    torch.tensor([[-1], [0], [1]], dtype=torch.int64),
                )
            ],
            properties=properties,
            keys=keys,
        )

        rmse = per_system_equivariance_rmse(
            {"foo_var": variance, "foo_mean": mean}, "foo"
        )

        block = rmse.block()
        assert block.samples.names == ["system"]
        assert block.properties == properties
        # system 0: atoms contribute 6/3 and 15/3 -> mean 3.5; system 1: 24/3
        expected = torch.sqrt(torch.tensor([3.5, 8.0], dtype=torch.float64))
        assert torch.allclose(block.values.squeeze(1), expected, atol=1e-12)

    def test_equivariance_rmse_preserves_blocks_and_properties(self):
        # spherical targets keep their (o3_lambda, o3_sigma) block and property
        # (e.g. radial channel) structure: one RMSE per system, block, and
        # property, each block divided by its own 2l+1, so the values can be
        # aggregated later in whichever way the analysis needs
        keys = Labels(
            ["o3_lambda", "o3_sigma"],
            torch.tensor([[0, 1], [1, 1]], dtype=torch.int64),
        )
        properties = Labels(["n"], torch.tensor([[0], [1]], dtype=torch.int64))
        samples_l0 = Labels(
            ["system", "atom"],
            torch.tensor([[0, 0], [1, 0]], dtype=torch.int64),
        )
        # the l=1 block only has samples in system 0
        samples_l1 = Labels(
            ["system", "atom"], torch.tensor([[0, 0]], dtype=torch.int64)
        )

        variance = TensorMap(
            keys,
            [
                TensorBlock(
                    values=torch.tensor([[1.0, 4.0], [9.0, 16.0]], dtype=torch.float64),
                    samples=samples_l0,
                    components=[],
                    properties=properties,
                ),
                TensorBlock(
                    values=torch.tensor([[3.0, 12.0]], dtype=torch.float64),
                    samples=samples_l1,
                    components=[],
                    properties=properties,
                ),
            ],
        )
        mean = TensorMap(
            keys,
            [
                TensorBlock(
                    values=torch.zeros((2, 1, 2), dtype=torch.float64),
                    samples=samples_l0,
                    components=[
                        Labels(["o3_mu"], torch.tensor([[0]], dtype=torch.int64))
                    ],
                    properties=properties,
                ),
                TensorBlock(
                    values=torch.zeros((1, 3, 2), dtype=torch.float64),
                    samples=samples_l1,
                    components=[
                        Labels(
                            ["o3_mu"],
                            torch.tensor([[-1], [0], [1]], dtype=torch.int64),
                        )
                    ],
                    properties=properties,
                ),
            ],
        )

        rmse = per_system_equivariance_rmse(
            {"spherical_var": variance, "spherical_mean": mean},
            "spherical",
            n_systems=2,
        )

        assert rmse.keys == keys
        # l=0 block: multiplicity 1, one atom per system -> sqrt of the variance
        block_l0 = rmse.block({"o3_lambda": 0, "o3_sigma": 1})
        assert block_l0.properties == properties
        assert torch.allclose(
            block_l0.values,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
            atol=1e-12,
        )
        # l=1 block: multiplicity 3; system 1 has no samples -> zero
        block_l1 = rmse.block({"o3_lambda": 1, "o3_sigma": 1})
        assert torch.allclose(
            block_l1.values,
            torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float64),
            atol=1e-12,
        )


class TestEquivarianceErrorMethod:
    def _make_systems(self):
        return [_two_atom_system(), _two_atom_system(_POSITIONS_B)]

    def test_equivariant_output_has_zero_error(self):
        # forces of _EnergyAndVectorModel back-rotate exactly, so the reported
        # equivariance error must vanish; this is the invariant the metric is for
        model = _symmetrized(
            _EnergyAndVectorModel(),
            batch_size=8,
        )
        systems = self._make_systems()

        errors = model.equivariance_error(
            systems,
            {
                "energy": ModelOutput(sample_kind="system"),
                "non_conservative_forces": ModelOutput(sample_kind="atom"),
            },
        )

        assert set(errors.keys()) == {"energy_l0", "non_conservative_forces_l1"}
        block = errors["non_conservative_forces_l1"].block()
        assert block.samples.names == ["system"]
        assert block.samples.values[:, 0].tolist() == [0, 1]
        assert block.values.shape == (2, 1)
        # above the float64 cancellation floor of the variance
        assert torch.all(block.values.abs() < 1e-7)

    def test_non_equivariant_output_matches_helper(self):
        # a non-invariant energy must give a strictly positive error, equal to
        # the per_system_equivariance_rmse reduction of the raw forward outputs
        model = _symmetrized(
            _AnisotropicEnergyModel(),
            max_o3_lambda_target=0,
            batch_size=8,
        )
        systems = self._make_systems()

        errors = model.equivariance_error(systems, _ENERGY_OUTPUTS)
        raw = model(systems, _ENERGY_OUTPUTS)
        expected = per_system_equivariance_rmse(raw, "energy_l0", n_systems=2)

        values = errors["energy_l0"].block().values
        assert torch.all(values > 0.1)
        assert torch.allclose(values, expected.block().values, atol=1e-12)


class TestFloat64Accumulation:
    """Statistics are accumulated in float64 regardless of the model dtype."""

    def test_underresolved_signed_grid_is_rejected(self):
        position = torch.tensor(
            [[-1.12984253e-2, 3.64940445e-4, -9.99936104e-1]],
            dtype=torch.float64,
        )
        position = position / torch.linalg.norm(position)
        system = System(
            types=torch.tensor([1]),
            positions=position,
            cell=torch.zeros((3, 3), dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        underresolved = SymmetrizedModel(
            _DegreeSevenEnergyModel(),
            max_o3_lambda_target=0,
            max_o3_lambda_grid=12,
            batch_size=64,
        )
        with pytest.raises(ValueError, match="materially negative.*Increase"):
            underresolved([system], _ENERGY_OUTPUTS)

        resolved = SymmetrizedModel(
            _DegreeSevenEnergyModel(),
            max_o3_lambda_target=0,
            max_o3_lambda_character=7,
            max_o3_lambda_grid=14,
            batch_size=64,
        )
        result = resolved([system], _ENERGY_OUTPUTS, compute_character_projections=True)
        expected = 1.0e6 * 17.0 / 137280.0
        assert result["energy_l0_mean"].block().values.item() == pytest.approx(
            0.0, abs=1e-12
        )
        assert result["energy_l0_var"].block().values.item() == pytest.approx(
            expected, rel=1e-12
        )
        assert result["energy_l0_norm_squared"].block().values.item() == pytest.approx(
            expected, rel=1e-12
        )
        sigma_plus, sigma_minus, _ = per_system_character_fractions(result, "energy_l0")
        assert torch.sum(sigma_plus + sigma_minus).item() == pytest.approx(
            1.0, abs=1e-12
        )
        assert torch.max(torch.abs(sigma_minus)).item() < 1e-12

    def test_float32_model_with_large_energy_offset(self):
        # positions sum to v = (1, 2, 0), so the exact O(3) variance of the
        # energy is |v|^2 / 3, independent of the invariant 1e5 offset; with
        # float32 accumulation the offset wipes out the variance entirely
        # (second moments ~1e10 quantize at ~1e3)
        system = System(
            types=torch.tensor([1, 1], dtype=torch.int32),
            positions=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32
            ),
            cell=torch.zeros((3, 3), dtype=torch.float32),
            pbc=torch.tensor([False, False, False]),
        )
        model = SymmetrizedModel(
            _OffsetAnisotropicEnergyModel(),
            max_o3_lambda_target=1,
            batch_size=8,
        ).to(dtype=torch.float32)

        results = model([system], _ENERGY_OUTPUTS)

        variance = results["energy_l0_var"].block().values
        assert variance.dtype == torch.float64
        expected = 5.0 / 3.0
        # the residual error is the float32 round-off of the model outputs
        # themselves (~1e5 * 1e-7), far below the 5% tolerance
        assert abs(variance.item() - expected) < 0.05 * expected

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 64])
    def test_large_invariant_offset_is_batch_independent(self, batch_size):
        model = _symmetrized(
            _ConstantEnergyModel(value=1.0e12),
            max_o3_lambda_target=0,
            batch_size=batch_size,
        )

        result = model([_two_atom_system()], _ENERGY_OUTPUTS)

        assert result["energy_l0_mean"].block().values.item() == 1.0e12
        assert result["energy_l0_var"].block().values.item() == 0.0
        assert result["energy_l0_norm_squared"].block().values.item() == 1.0e24


class TestCustomEnergyName:
    """Gradients can be derived from an energy output with a non-standard name."""

    def _make_system(self):
        return _two_atom_system()

    def test_gradients_with_custom_energy_name(self):
        systems = [self._make_system()]

        reference = _symmetrized(_QuadraticEnergyModel())
        renamed = _symmetrized(_RenamedEnergyModel())

        reference_results = reference(
            systems,
            _ENERGY_OUTPUTS,
            compute_gradients=True,
        )
        renamed_results = renamed(
            systems,
            {"mtt::energy": ModelOutput(sample_kind="system")},
            compute_gradients=True,
            energy_name="mtt::energy",
        )

        # the derived forces are identical; only the energy naming differs
        # (non-standard energies pass through without the _l0 relabeling)
        for suffix in ("mean", "var", "norm_squared"):
            assert mts.allclose(
                reference_results[f"forces_l1_{suffix}"],
                renamed_results[f"forces_l1_{suffix}"],
                atol=1e-12,
            )
            assert torch.allclose(
                reference_results[f"energy_l0_{suffix}"].block().values.squeeze(1),
                renamed_results[f"mtt::energy_{suffix}"].block().values,
                atol=1e-12,
            )

    def test_missing_energy_name_raises(self):
        model = _symmetrized(_RenamedEnergyModel())

        with pytest.raises(ValueError, match="requires 'energy' in outputs"):
            model(
                [self._make_system()],
                {"mtt::energy": ModelOutput(sample_kind="system")},
                compute_gradients=True,
            )

    def test_equivariance_error_with_custom_energy_name(self):
        model = _symmetrized(_RenamedEnergyModel())

        errors = model.equivariance_error(
            [self._make_system()],
            {"mtt::energy": ModelOutput(sample_kind="system")},
            compute_gradients=True,
            energy_name="mtt::energy",
        )

        assert set(errors.keys()) == {"mtt::energy", "forces_l1"}
        # the underlying model is exactly equivariant; tolerance above the
        # float64 cancellation floor of the variance
        assert torch.all(errors["forces_l1"].block().values.abs() < 1e-7)


class TestAtomisticBaseModel:
    """SymmetrizedModel accepts exported AtomisticModel base models."""

    def _make_system(self):
        return _two_atom_system()

    def _export(self, module=None, explicit_gradients=None):
        if module is None:
            module = _QuadraticEnergyModel()
        if explicit_gradients is None:
            explicit_gradients = []
        return AtomisticModel(
            module.eval(),
            ModelMetadata(),
            ModelCapabilities(
                outputs={
                    "energy": ModelOutput(
                        sample_kind="system",
                        unit="eV",
                        description="energy",
                        explicit_gradients=explicit_gradients,
                    )
                },
                atomic_types=[1],
                interaction_range=0.0,
                length_unit="Angstrom",
                supported_devices=["cpu"],
                dtype="float64",
            ),
        )

    def _symmetrize(self, base_model):
        return _symmetrized(base_model)

    def _assert_same_results(self, reference, results):
        assert set(results.keys()) == set(reference.keys())
        for key in reference:
            assert mts.allclose(reference[key], results[key], atol=1e-12)

    def test_forward_matches_raw_module(self):
        outputs = {"energy": ModelOutput(sample_kind="system")}
        systems = [self._make_system()]

        reference = self._symmetrize(_QuadraticEnergyModel())(systems, outputs)
        results = self._symmetrize(self._export())(systems, outputs)

        self._assert_same_results(reference, results)

    def test_custom_input_unit_metadata_survives_rotation(self):
        system = self._make_system()
        masses = _single_block_tensor_map(
            torch.tensor([[1.0], [2.0]], dtype=torch.float64),
            samples=Labels(["system", "atom"], torch.tensor([[0, 0], [0, 1]])),
            components=[],
            properties=Labels(["mass"], torch.tensor([[0]])),
        )
        masses.set_info("unit", "u")
        system.add_data("mass", masses)

        model = self._symmetrize(self._export(_CustomUnitInputEnergyModel()))
        result = model([system], {"energy": ModelOutput(sample_kind="system")})

        expected = 3.0 * unit_conversion_factor("u", "kg")
        assert result["energy_l0_mean"].block().values.item() == pytest.approx(
            expected, rel=1e-12
        )

    def test_gradients_through_exported_model(self):
        outputs = {"energy": ModelOutput(sample_kind="system")}
        systems = [self._make_system()]

        reference = self._symmetrize(_QuadraticEnergyModel())(
            systems, outputs, compute_gradients=True
        )
        results = self._symmetrize(self._export())(
            systems, outputs, compute_gradients=True
        )

        assert "forces_l1_mean" in results
        self._assert_same_results(reference, results)

    @pytest.mark.parametrize("compute_gradients", [False, True])
    def test_exported_model_unsolicited_gradient_is_rejected(self, compute_gradients):
        exported = self._export(
            module=_ExplicitGradientEnergyModel(),
            explicit_gradients=["positions"],
        )
        model = self._symmetrize(exported)

        with pytest.raises(
            ValueError,
            match="output 'energy' contains explicit gradient 'positions'",
        ):
            model(
                [self._make_system()],
                {"energy": ModelOutput(sample_kind="system")},
                compute_gradients=compute_gradients,
            )

    def test_exported_model_converts_requested_units(self):
        # requested outputs are forwarded unchanged, so each base-model kind
        # keeps its usual metatomic semantics: exported models convert to the
        # requested unit, raw modules ignore units entirely
        outputs = {"energy": ModelOutput(sample_kind="system", unit="kcal/mol")}
        systems = [self._make_system()]

        raw = self._symmetrize(_QuadraticEnergyModel())(systems, outputs)
        exported = self._symmetrize(self._export())(systems, outputs)

        conversion = unit_conversion_factor("eV", "kcal/mol")
        assert torch.allclose(
            exported["energy_l0_mean"].block().values,
            conversion * raw["energy_l0_mean"].block().values,
            atol=1e-12,
        )

    def test_loaded_model_matches_raw_module(self, tmp_path):
        outputs = {"energy": ModelOutput(sample_kind="system")}
        systems = [self._make_system()]

        path = str(tmp_path / "exported.pt")
        self._export().save(path)
        loaded = load_atomistic_model(path)

        reference = self._symmetrize(_QuadraticEnergyModel())(systems, outputs)
        results = self._symmetrize(loaded)(systems, outputs)

        self._assert_same_results(reference, results)


class TestGradientPathIntegrity:
    """The compute_gradients path must transform inputs exactly like the
    no-grad path: fresh neighbor-list storage per rotated copy, and custom
    system data rotated along."""

    def _make_system_with_neighbor_list(self):
        system = System(
            types=torch.tensor([1, 1], dtype=torch.int32),
            positions=torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
            ),
            cell=torch.zeros((3, 3), dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        options = NeighborListOptions(cutoff=3.5, full_list=True, strict=True)
        neighbors = TensorBlock(
            values=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
            ).reshape(-1, 3, 1),
            samples=Labels(
                [
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                torch.tensor([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.int32),
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        )
        system.add_neighbor_list(options, neighbors)
        return system, options

    def test_neighbor_lists_are_not_mutated(self):
        # regression test: the rotated copies used to write through storage
        # shared with the original system, compounding rotations across copies
        # and corrupting the caller's neighbor list
        class _CapturingModel(_QuadraticEnergyModel):
            def __init__(self):
                super().__init__()
                self.captured: List[System] = []

            def forward(self, systems, outputs, selected_atoms=None):
                self.captured = list(systems)
                return super().forward(systems, outputs, selected_atoms)

        system, options = self._make_system_with_neighbor_list()
        original = system.get_neighbor_list(options).values.detach().clone()

        # +90 degrees about z, then +90 degrees about x
        rotations = torch.tensor(
            [
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            ],
            dtype=torch.float64,
        )

        model = _CapturingModel()
        _evaluate_with_gradients(
            model,
            system,
            rotations,
            {"energy": ModelOutput(sample_kind="system")},
            None,
            torch.device("cpu"),
            torch.float64,
        )

        assert torch.equal(system.get_neighbor_list(options).values, original)
        for i, captured in enumerate(model.captured):
            expected = original.squeeze(-1) @ rotations[i].T
            assert torch.allclose(
                captured.get_neighbor_list(options).values.squeeze(-1),
                expected,
                atol=1e-14,
            )

    def test_custom_data_is_rotated(self, monkeypatch):
        # energy = sum(pos^2) + field . sum(pos) is invariant when the stored
        # field rotates with the system: the forces equivariance error must
        # vanish (and the run must not crash on the data-carrying system)
        class _FieldModel(torch.nn.Module):
            def forward(
                self,
                systems: List[System],
                outputs: Dict[str, ModelOutput],
                selected_atoms: Optional[Labels] = None,
            ) -> Dict[str, TensorMap]:
                energies = []
                for sys in systems:
                    field = sys.get_data("mtt::field").block().values[0, :, 0]
                    energies.append(
                        torch.sum(sys.positions**2)
                        + torch.dot(field, sys.positions.sum(dim=0))
                    )
                return {
                    "energy": _system_tensor_map(torch.stack(energies).unsqueeze(-1))
                }

            def requested_neighbor_lists(self):
                return []

        system = System(
            types=torch.tensor([1, 1], dtype=torch.int32),
            positions=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64
            ),
            cell=torch.zeros((3, 3), dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        field = _single_block_tensor_map(
            torch.tensor([[[0.3], [0.7], [-0.2]]], dtype=torch.float64),
            samples=Labels(["system"], torch.tensor([[0]])),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("field", 1),
        )
        system.add_data("mtt::field", field)

        checked_constructor_calls = 0
        original_init = O3Transformation.__init__

        def counted_init(self, *args, **kwargs):
            nonlocal checked_constructor_calls
            checked_constructor_calls += 1
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(O3Transformation, "__init__", counted_init)

        model = _symmetrized(_FieldModel(), batch_size=8)
        errors = model.equivariance_error(
            [system],
            _ENERGY_OUTPUTS,
            compute_gradients=True,
        )

        assert torch.all(errors["forces_l1"].block().values.abs() < 1e-7)
        # All matrices in this path come from the wrapper's fixed quadrature;
        # none should repeat the public constructor's synchronization-heavy
        # orthogonality and determinant validation.
        assert checked_constructor_calls == 0

    def test_reserved_output_names_raise(self):
        model = _symmetrized(_QuadraticEnergyModel(), batch_size=8)
        with pytest.raises(ValueError, match="reserved for the autograd-derived"):
            model(
                [_two_atom_system()],
                {
                    "energy": ModelOutput(sample_kind="system"),
                    "forces": ModelOutput(sample_kind="atom"),
                },
                compute_gradients=True,
            )

    def test_results_carry_no_autograd_graph(self):
        # regression test: the accumulators used to keep every rotated copy's
        # forward graph alive across the whole grid in compute_gradients mode
        model = _symmetrized(_QuadraticEnergyModel())
        results = model(
            [_two_atom_system()],
            _ENERGY_OUTPUTS,
            compute_gradients=True,
        )
        for name in ("energy_l0_mean", "energy_l0_var", "forces_l1_mean"):
            assert results[name].block().values.grad_fn is None


class TestLambdaValidation:
    """Input data uses its actual rank; outputs obey max_o3_lambda_target."""

    def _spherical_map(self, ell):
        return _system_tensor_map(
            torch.rand(1, 2 * ell + 1, 1, dtype=torch.float64),
            property_name="p",
            ell=ell,
        )

    def _make_system(self):
        return _two_atom_system()

    def test_system_data_rank_is_independent_of_output_target(self):
        system = self._make_system()
        spherical_field = self._spherical_map(3)
        expected = torch.sum(spherical_field.block().values ** 2)
        system.add_data("mtt::spherical_field", spherical_field)

        class _SphericalNormModel(torch.nn.Module):
            def forward(self, systems, outputs, selected_atoms=None):
                energies = torch.stack(
                    [
                        torch.sum(
                            item.get_data("mtt::spherical_field").block().values ** 2
                        )
                        for item in systems
                    ]
                )
                return {"energy": _system_tensor_map(energies.unsqueeze(-1))}

            def requested_neighbor_lists(self):
                return []

        model = _symmetrized(
            _SphericalNormModel(), max_o3_lambda_target=0, batch_size=8
        )
        result = model([system], _ENERGY_OUTPUTS)

        assert torch.allclose(
            result["energy_l0_mean"].block().values.squeeze(),
            expected,
            atol=1e-10,
        )
        assert torch.all(result["energy_l0_var"].block().values.abs() < 1e-10)

    def test_output_beyond_target_raises(self):
        spherical_map = self._spherical_map(2)

        class _SphericalOutputModel(torch.nn.Module):
            def forward(self, systems, outputs, selected_atoms=None):
                return {"mtt::spherical": spherical_map}

            def requested_neighbor_lists(self):
                return []

        model = _symmetrized(_SphericalOutputModel(), batch_size=8)

        with pytest.raises(ValueError, match="larger than max_o3_lambda_target=1"):
            model(
                [self._make_system()],
                {"mtt::spherical": ModelOutput(sample_kind="system")},
            )

    @pytest.mark.parametrize("spherical_first", [False, True])
    @pytest.mark.parametrize("compute_character_projections", [False, True])
    def test_wigner_work_uses_actual_output_and_character_ranks(
        self, monkeypatch, spherical_first, compute_character_projections
    ):
        class _ScalarAndSphericalModel(torch.nn.Module):
            def forward(self, systems, outputs, selected_atoms=None):
                device = systems[0].positions.device
                dtype = systems[0].positions.dtype
                scalar = _system_tensor_map(
                    torch.ones(len(systems), 1, dtype=dtype, device=device)
                )
                spherical = _system_tensor_map(
                    torch.ones(len(systems), 5, 1, dtype=dtype, device=device),
                    property_name="p",
                    ell=2,
                )
                available = {"energy": scalar, "mtt::spherical": spherical}
                return {name: available[name] for name in outputs}

            def requested_neighbor_lists(self):
                return []

        built_ranks = []
        original_build = transformation_module.build_wigner_D_cache

        def tracked_build(ell_max, *args, **kwargs):
            built_ranks.append(ell_max)
            return original_build(ell_max, *args, **kwargs)

        monkeypatch.setattr(
            transformation_module, "build_wigner_D_cache", tracked_build
        )

        names = ["energy", "mtt::spherical"]
        if spherical_first:
            names.reverse()
        outputs = {
            name: ModelOutput(sample_kind="system" if name == "energy" else "atom")
            for name in names
        }
        model = _symmetrized(
            _ScalarAndSphericalModel(),
            max_o3_lambda_target=4,
            max_o3_lambda_character=0,
            batch_size=64,
        )

        result = model(
            [self._make_system()],
            outputs,
            compute_character_projections=compute_character_projections,
        )

        assert built_ranks and set(built_ranks) == {2}
        assert len(built_ranks) == model.so3_rotations.shape[0]
        assert (
            "energy_l0_character_projection" in result
        ) is compute_character_projections


class TestPersistentWignerCache:
    """User-visible cache contracts: bounded, optional, reusable, and derived."""

    @staticmethod
    def _model(cache_max_bytes=64 * 1024**2):
        return _symmetrized(
            _ConstantEnergyModel(),
            max_o3_lambda_target=0,
            max_o3_lambda_character=2,
            max_o3_lambda_grid=5,
            batch_size=32,
            wigner_cache_max_bytes=cache_max_bytes,
        )

    @staticmethod
    def _run(model, *, dtype=torch.float64):
        return model(
            [_two_atom_system(dtype=dtype)],
            _ENERGY_OUTPUTS,
            compute_character_projections=True,
        )

    @staticmethod
    def _assert_same(first, second):
        assert set(first) == set(second)
        for name in first:
            assert mts.equal(first[name], second[name])

    def test_budget_controls_cache_without_changing_results(self):
        cached = self._model()
        uncached = self._model(cache_max_bytes=0)

        expected = self._run(cached)
        self._assert_same(expected, self._run(uncached))
        retained_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in cached._wigner_cache.values()
        )
        assert 0 < retained_bytes <= cached.wigner_cache_max_bytes
        assert uncached._wigner_cache == {}

        # A lower-rank request must not retain a higher-rank cache that no
        # longer fits a reduced public budget.
        cached.wigner_cache_max_bytes = retained_bytes // 2
        lower_rank = cached._persistent_wigner_stacks(
            1, device=torch.device("cpu"), dtype=torch.float64
        )
        assert lower_rank is not None and set(lower_rank) == {0, 1}

        # Changing the public budget at runtime releases a warm cache while
        # preserving numerical results through the bounded fallback path.
        cached.wigner_cache_max_bytes = 0
        self._assert_same(expected, self._run(cached))
        assert cached._wigner_cache == {}

    def test_one_build_is_reused_across_original_systems(self, monkeypatch):
        model = self._model()
        build_calls = 0
        original_build = transformation_module.build_wigner_D_cache

        def counted_build(*args, **kwargs):
            nonlocal build_calls
            build_calls += 1
            return original_build(*args, **kwargs)

        monkeypatch.setattr(
            transformation_module, "build_wigner_D_cache", counted_build
        )
        model(
            [_two_atom_system(), _two_atom_system(_POSITIONS_B)],
            {"energy": ModelOutput(sample_kind="system")},
            compute_character_projections=True,
        )

        # Systems are still evaluated separately, but the second one reuses
        # the fixed grid representation built while processing the first.
        assert build_calls == model.so3_rotations.shape[0]

    def test_warm_cache_to_dtype_matches_fresh_model(self):
        model = self._model()
        self._run(model)

        model.to(dtype=torch.float32)
        assert model._wigner_cache == {}

        fresh = self._model().to(dtype=torch.float32)
        expected = self._run(fresh, dtype=torch.float32)
        actual = self._run(model, dtype=torch.float32)
        self._assert_same(expected, actual)

    def test_grid_mutation_invalidates_and_preserves_cache_values(self):
        model = self._model()
        self._run(model)
        old_key = model._wigner_cache_key
        old_stack = model._wigner_cache[2]

        # Read-only forwards must leave the persistent matrices unchanged.
        expected_cache = {
            ell: tensor.clone() for ell, tensor in model._wigner_cache.items()
        }
        self._run(model)
        for ell, expected in expected_cache.items():
            assert torch.equal(model._wigner_cache[ell], expected)

        # An in-place grid mutation increments Tensor._version and must rebuild
        # the Wigner matrices derived from the canonical grid.
        with torch.no_grad():
            model.so3_rotations[0].copy_(model.so3_rotations[1])
        self._run(model)
        assert model._wigner_cache_key != old_key
        assert model._wigner_cache[2] is not old_stack
        assert torch.equal(model._wigner_cache[2][0], model._wigner_cache[2][1])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_cuda_cached_and_uncached_results_match(self, monkeypatch):
        build_devices = []
        original_build = transformation_module.build_wigner_D_cache

        def record_build_device(*args, **kwargs):
            build_devices.append(args[1].device.type)
            return original_build(*args, **kwargs)

        monkeypatch.setattr(
            transformation_module,
            "build_wigner_D_cache",
            record_build_device,
        )
        cached = self._model().to(device="cuda")
        uncached = self._model(cache_max_bytes=0).to(device="cuda")
        system = _two_atom_system().to(device="cuda")

        expected = cached(
            [system],
            _ENERGY_OUTPUTS,
            compute_character_projections=True,
        )
        actual = uncached(
            [system],
            _ENERGY_OUTPUTS,
            compute_character_projections=True,
        )

        self._assert_same(expected, actual)
        assert cached._wigner_cache
        assert uncached._wigner_cache == {}
        assert len(build_devices) == 2 * cached.so3_rotations.shape[0]
        assert set(build_devices) == {"cpu"}

    def test_cache_is_rebuilt_after_serialization(self):
        model = self._model()
        expected = self._run(model)
        assert model._wigner_cache
        state = model.state_dict()
        assert "so3_rotations" in state
        assert all("wigner_cache" not in name for name in state)

        payload = io.BytesIO()
        torch.save(model, payload)
        payload.seek(0)
        loaded = torch.load(payload, weights_only=False)

        assert loaded._wigner_cache == {}
        actual = self._run(loaded)
        self._assert_same(expected, actual)


class TestDecomposeOutputNames:
    """All spellings of the force/stress output names decompose to spherical
    blocks; unknown names pass through."""

    def _vector_map(self):
        return _single_block_tensor_map(
            torch.rand(2, 3, 1, dtype=torch.float64),
            samples=Labels(["system", "atom"], torch.tensor([[0, 0], [0, 1]])),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("p", 1),
        )

    def _scalar_map(self, keys=None):
        return _single_block_tensor_map(
            torch.rand(1, 2, dtype=torch.float64),
            samples=Labels(["system"], torch.tensor([[0]])),
            components=[],
            properties=Labels.range("p", 2),
            keys=keys,
        )

    def _stress_map(self):
        return _single_block_tensor_map(
            torch.rand(1, 3, 3, 1, dtype=torch.float64),
            samples=Labels(["system"], torch.tensor([[0]])),
            components=[Labels.range("xyz_1", 3), Labels.range("xyz_2", 3)],
            properties=Labels.range("p", 1),
        )

    @staticmethod
    def _assert_irrep_keys(tensor, ell):
        assert tensor.keys.names == ["o3_lambda", "o3_sigma"]
        assert tensor.keys.values.tolist() == [[ell, 1]]

    @pytest.mark.parametrize(
        "name",
        [
            "non_conservative_force",
            "non_conservative_forces",
            "non_conservative_force/direct",
        ],
    )
    def test_vector_names_are_decomposed(self, name):
        output_name = name + "_l1"
        result = _decompose_output(name, self._vector_map())
        assert set(result) == {output_name}
        assert result[output_name].block().components[0].names == ["o3_mu"]
        self._assert_irrep_keys(result[output_name], 1)

    @pytest.mark.parametrize(
        "name",
        ["energy/pbe", "energy_ensemble/member", "energy_uncertainty/direct"],
    )
    def test_energy_variants_are_decomposed(self, name):
        result = _decompose_output(name, self._scalar_map())
        assert set(result) == {name + "_l0"}
        self._assert_irrep_keys(result[name + "_l0"], 0)

    def test_stress_variant_is_decomposed(self):
        name = "non_conservative_stress/direct"
        result = _decompose_output(name, self._stress_map())
        assert set(result) == {name + "_l0", name + "_l2"}
        self._assert_irrep_keys(result[name + "_l0"], 0)
        self._assert_irrep_keys(result[name + "_l2"], 2)

    def test_semantic_keys_are_preserved_and_placeholder_is_validated(self):
        tensor = self._scalar_map(Labels(["channel"], torch.tensor([[7]])))
        result = _decompose_output("energy", tensor)["energy_l0"]
        assert result.keys.names == ["channel", "o3_lambda", "o3_sigma"]
        assert result.keys.values.tolist() == [[7, 0, 1]]

        with pytest.raises(ValueError, match="canonical.*placeholder"):
            _decompose_output(
                "energy", self._scalar_map(Labels(["_"], torch.tensor([[1]])))
            )
        with pytest.raises(ValueError, match="existing 'o3_lambda'.*inconsistent"):
            _decompose_output(
                "energy",
                self._scalar_map(
                    Labels(
                        ["channel", "o3_lambda"],
                        torch.tensor([[7, 1]]),
                    )
                ),
            )

    @pytest.mark.parametrize("inversion", [1.0, -1.0])
    @pytest.mark.parametrize(
        ("name", "tensor_factory"),
        [
            ("non_conservative_force", "_vector_map"),
            ("non_conservative_stress", "_stress_map"),
        ],
    )
    def test_decomposition_commutes_with_o3_transform(
        self, name, tensor_factory, inversion
    ):
        tensor = getattr(self, tensor_factory)()
        matrix = inversion * torch.tensor(
            Rotation.from_rotvec([0.2, -0.4, 0.3]).as_matrix(),
            dtype=torch.float64,
        )
        transformation = O3Transformation(matrix, max_angular_momentum=2)
        system = _two_atom_system()

        decomposed_first = _decompose_output(name, tensor)
        transformed_first = {
            output_name: transform_tensor(output, [system], [transformation])
            for output_name, output in decomposed_first.items()
        }
        transformed_cartesian = transform_tensor(tensor, [system], [transformation])
        decomposed_second = _decompose_output(name, transformed_cartesian)

        assert set(transformed_first) == set(decomposed_second)
        for output_name in transformed_first:
            assert mts.allclose(
                transformed_first[output_name],
                decomposed_second[output_name],
                atol=1e-12,
            )

    def test_unknown_name_passes_through(self):
        tensor = self._vector_map()
        result = _decompose_output("mtt::custom", tensor)
        assert set(result.keys()) == {"mtt::custom"}


class TestSelectedAtomsColumnOrder:
    def test_system_column_found_by_name(self):
        # the local copy index must go into the "system" column wherever it
        # is, not positionally into column 0
        selection = Labels(["atom", "system"], torch.tensor([[3, 0], [5, 0]]))
        local = _selected_atoms_for_local_systems(selection, 0, 2)
        assert local.names == ["atom", "system"]
        assert local.values[:, 0].tolist() == [3, 5, 3, 5]
        assert local.values[:, 1].tolist() == [0, 0, 1, 1]
