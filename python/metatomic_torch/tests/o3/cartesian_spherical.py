import itertools
import math
import os

import metatensor.torch as mts
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch.o3 import (
    calculate_cg_coefficients,
    cartesian_to_spherical,
    random_transformations,
    spherical_to_cartesian,
)

from .._tests_utils import can_use_mps_backend


ALL_DEVICE_DTYPE = [("cpu", "float64"), ("cpu", "float32")]

if torch.cuda.is_available():
    ALL_DEVICE_DTYPE.append(("cuda", "float64"))
    ALL_DEVICE_DTYPE.append(("cuda", "float32"))

if can_use_mps_backend():
    ALL_DEVICE_DTYPE.append(("mps", "float32"))


def cartesian_tensor(*, rank, dtype, device):
    """A single block tensor with ``rank`` Cartesian components and random values."""
    values = torch.rand((6,) + (3,) * rank + (3,), dtype=dtype, device=device)
    return TensorMap(
        keys=Labels.single().to(device),
        blocks=[
            TensorBlock(
                values=values,
                samples=Labels.range("system", 6).to(device),
                components=[
                    Labels.range(f"xyz_{i}", 3).to(device) for i in range(1, rank + 1)
                ],
                properties=Labels.range("p", 3).to(device),
            )
        ],
    )


def add_gradients(tensor: TensorMap) -> TensorMap:
    """Copy ``tensor``, adding ``positions`` gradients full of zeros to all blocks."""
    blocks = []
    for block in tensor.blocks():
        block = block.copy()
        components = [Labels.range("xyz", 3).to(block.values.device)] + block.components
        shape = [2] + [len(c) for c in components] + [len(block.properties)]
        block.add_gradient(
            "positions",
            TensorBlock(
                values=torch.zeros(
                    shape, dtype=block.values.dtype, device=block.values.device
                ),
                samples=Labels(
                    ["sample", "atom"],
                    torch.tensor([[0, 0], [1, 0]], device=block.values.device),
                ),
                components=components,
                properties=block.properties,
            ),
        )
        blocks.append(block)

    return TensorMap(tensor.keys, blocks)


@torch.jit.script
def dtype_to_int(dtype: torch.dtype) -> int:
    return dtype


@pytest.fixture
def cartesian():
    # the first block is completely symmetric
    values_1 = torch.rand((10, 4, 3, 3, 3, 2), dtype=torch.float64)
    symmetric_values_1 = torch.zeros_like(values_1)
    for permutation in itertools.permutations([2, 3, 4]):
        symmetric_values_1 += values_1.permute(0, 1, *permutation, 5)
    symmetric_values_1 /= 6

    block_1 = TensorBlock(
        values=symmetric_values_1,
        samples=Labels.range("s", 10),
        components=[
            Labels.range("other", 4),
            Labels.range("xyz_1", 3),
            Labels.range("xyz_2", 3),
            Labels.range("xyz_3", 3),
        ],
        properties=Labels.range("p", 2),
    )

    # second block does not have any specific symmetry
    block_2 = TensorBlock(
        values=torch.rand((12, 6, 3, 3, 3, 7), dtype=torch.float64),
        samples=Labels.range("s", 12),
        components=[
            Labels.range("other", 6),
            Labels.range("xyz_1", 3),
            Labels.range("xyz_2", 3),
            Labels.range("xyz_3", 3),
        ],
        properties=Labels.range("p", 7),
    )

    return TensorMap(Labels.range("key", 2), [block_1, block_2])


def test_cartesian_to_spherical_rank_1(cartesian):
    spherical = cartesian_to_spherical(cartesian, components=["xyz_1"])

    assert spherical.component_names == ["other", "o3_mu", "xyz_2", "xyz_3"]
    assert spherical.keys.names == ["o3_lambda", "o3_sigma", "key"]

    spherical = cartesian_to_spherical(
        cartesian,
        components=["xyz_1"],
        keep_l_in_keys=True,
    )
    assert spherical.keys.names == ["o3_lambda", "o3_sigma", "l_1", "key"]


def test_cartesian_to_spherical_rank_2(cartesian):
    spherical = cartesian_to_spherical(cartesian, components=["xyz_1", "xyz_2"])

    assert spherical.component_names == ["other", "o3_mu", "xyz_3"]
    assert spherical.keys == Labels(
        ["o3_lambda", "o3_sigma", "key"],
        torch.tensor(
            [
                # only o3_sigma=1 in the symmetric block
                [0, 1, 0],
                [2, 1, 0],
                # all o3_sigma in the non-symmetric block
                [0, 1, 1],
                [1, -1, 1],
                [2, 1, 1],
            ]
        ),
    )

    spherical = cartesian_to_spherical(
        cartesian,
        components=["xyz_1", "xyz_2"],
        keep_l_in_keys=True,
        remove_blocks_threshold=None,
    )
    assert spherical.keys.names == ["o3_lambda", "o3_sigma", "l_2", "l_1", "key"]
    # all blocks are kept
    assert len(spherical.keys) == 6


def test_cartesian_to_spherical_rank_3(cartesian):
    spherical = cartesian_to_spherical(
        cartesian, components=["xyz_1", "xyz_2", "xyz_3"]
    )

    assert spherical.component_names == ["other", "o3_mu"]
    assert spherical.keys == Labels(
        ["o3_lambda", "o3_sigma", "l_3", "k_1", "l_2", "l_1", "key"],
        torch.tensor(
            [
                # only o3_sigma=1 for the symmetric block, but there are multiple "path"
                # ("l_3", "k_1", "l_2", "l_1") that lead to o3_lambda=1
                [1, 1, 1, 0, 1, 1, 0],
                [1, 1, 1, 2, 1, 1, 0],
                [3, 1, 1, 2, 1, 1, 0],
                # all possible o3_sigma for the non-symmetric block
                [1, 1, 1, 0, 1, 1, 1],
                [0, -1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [2, -1, 1, 1, 1, 1, 1],
                [1, 1, 1, 2, 1, 1, 1],
                [2, -1, 1, 2, 1, 1, 1],
                [3, 1, 1, 2, 1, 1, 1],
            ]
        ),
    )

    spherical = cartesian_to_spherical(
        cartesian,
        components=["xyz_1", "xyz_2", "xyz_3"],
        remove_blocks_threshold=None,
    )
    # all blocks are kept, even those with norm=0
    assert len(spherical.keys) == 14


@pytest.mark.parametrize(
    "components", [["xyz_1"], ["xyz_1", "xyz_2"], ["xyz_1", "xyz_2", "xyz_3"]]
)
@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
def test_cartesian_to_spherical_roundtrip(cartesian, components, device, dtype):
    dtype = getattr(torch, dtype)
    cartesian = cartesian.to(device=device, dtype=dtype)

    assert cartesian.device.type == device
    assert cartesian.dtype == dtype_to_int(dtype)

    spherical = cartesian_to_spherical(
        cartesian,
        components=components,
        keep_l_in_keys=True,
        # keep every block so the transformation stays exactly invertible
        remove_blocks_threshold=None,
    )

    assert spherical.device.type == device
    assert spherical.dtype == dtype_to_int(dtype)

    assert "o3_lambda" in spherical.keys.names

    # going back to cartesian should recover the original tensor exactly
    back = spherical_to_cartesian(spherical)

    assert back.device.type == device
    assert back.dtype == dtype_to_int(dtype)

    assert mts.equal_metadata(back, cartesian)

    if dtype == torch.float64:
        assert mts.allclose(back, cartesian, atol=1e-12, rtol=1e-12)
    else:
        assert dtype == torch.float32
        assert mts.allclose(back, cartesian, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
def test_spherical_to_cartesian_roundtrip(rank, device, dtype):
    dtype = getattr(torch, dtype)
    cartesian = cartesian_tensor(rank=rank, device=device, dtype=dtype)

    assert cartesian.device.type == device
    assert cartesian.dtype == dtype_to_int(dtype)

    component_names = [f"xyz_{i}" for i in range(1, rank + 1)]
    spherical = cartesian_to_spherical(
        cartesian,
        components=component_names,
        keep_l_in_keys=True,
        remove_blocks_threshold=None,
    )

    assert spherical.device.type == device
    assert spherical.dtype == dtype_to_int(dtype)

    # the spherical representation should no longer contain cartesian components
    assert spherical.component_names == ["o3_mu"]

    back = spherical_to_cartesian(spherical)

    assert back.device.type == device
    assert back.dtype == dtype_to_int(dtype)

    assert mts.equal_metadata(back, cartesian)

    if dtype == torch.float64:
        assert mts.allclose(back, cartesian, atol=1e-12, rtol=1e-12)
    else:
        assert dtype == torch.float32
        assert mts.allclose(back, cartesian, atol=1e-6, rtol=1e-6)


def test_roundtrip_with_removed_blocks():
    # an antisymmetric rank 2 tensor only contains a o3_lambda=1 block
    cartesian = cartesian_tensor(rank=2, dtype=torch.float64, device="cpu")
    values = cartesian.block().values
    antisymmetric = TensorMap(
        cartesian.keys,
        [
            TensorBlock(
                values=values - values.transpose(1, 2),
                samples=cartesian.block().samples,
                components=cartesian.block().components,
                properties=cartesian.block().properties,
            )
        ],
    )

    spherical = cartesian_to_spherical(
        antisymmetric,
        components=["xyz_1", "xyz_2"],
        keep_l_in_keys=True,
    )
    assert spherical.keys == Labels(
        ["o3_lambda", "o3_sigma", "l_2", "l_1", "_"],
        torch.tensor([[1, -1, 1, 1, 0]]),
    )

    # the rank of the tensor is taken from the `l_{i}` dimensions, and the removed
    # blocks are treated as zeros
    back = spherical_to_cartesian(spherical)
    assert mts.equal_metadata(back, antisymmetric)
    assert mts.allclose(back, antisymmetric)

    # symmetric rank 3 tensor, without any o3_sigma=-1 block
    cartesian = cartesian_tensor(rank=3, dtype=torch.float64, device="cpu")
    values = cartesian.block().values
    symmetric_values = torch.zeros_like(values)
    for permutation in itertools.permutations([1, 2, 3]):
        symmetric_values += values.permute(0, *permutation, 4)
    symmetric = TensorMap(
        cartesian.keys,
        [
            TensorBlock(
                values=symmetric_values,
                samples=cartesian.block().samples,
                components=cartesian.block().components,
                properties=cartesian.block().properties,
            )
        ],
    )

    spherical = cartesian_to_spherical(
        symmetric, components=["xyz_1", "xyz_2", "xyz_3"]
    )
    assert torch.all(spherical.keys.column("o3_sigma") == 1)

    back = spherical_to_cartesian(spherical)
    assert mts.equal_metadata(back, symmetric)
    assert mts.allclose(back, symmetric)


def test_remove_all_blocks():
    cartesian = cartesian_tensor(rank=2, dtype=torch.float64, device="cpu")
    zeros = TensorMap(
        cartesian.keys,
        [
            TensorBlock(
                values=torch.zeros_like(cartesian.block().values),
                samples=cartesian.block().samples,
                components=cartesian.block().components,
                properties=cartesian.block().properties,
            )
        ],
    )

    spherical = cartesian_to_spherical(zeros, components=["xyz_1", "xyz_2"])
    assert len(spherical) == 0
    assert spherical.keys.names == ["o3_lambda", "o3_sigma", "_"]


def test_spherical_to_cartesian_without_other_keys():
    # spherical tensors created by hand only have o3_lambda and o3_sigma keys
    spherical = TensorMap(
        Labels(["o3_lambda", "o3_sigma"], torch.tensor([[0, 1], [1, -1], [2, 1]])),
        [
            TensorBlock(
                values=torch.rand((5, 2 * o3_lambda + 1, 2), dtype=torch.float64),
                samples=Labels.range("system", 5),
                components=[
                    Labels(
                        "o3_mu",
                        torch.arange(-o3_lambda, o3_lambda + 1).reshape(-1, 1),
                    )
                ],
                properties=Labels.range("p", 2),
            )
            for o3_lambda in [0, 1, 2]
        ],
    )

    cartesian = spherical_to_cartesian(spherical, new_component_names=["a", "b"])
    assert cartesian.keys == Labels.single()
    assert cartesian.component_names == ["a", "b"]

    back = cartesian_to_spherical(cartesian, components=["a", "b"])
    back = mts.remove_dimension(back, "keys", "_")
    assert mts.equal_metadata(back, spherical)
    assert mts.allclose(back, spherical)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_equivariance(rank):
    cartesian = cartesian_tensor(rank=rank, dtype=torch.float64, device="cpu")
    transformations = random_transformations(1, rank, dtype=torch.float64, device="cpu")

    components = [f"xyz_{i}" for i in range(1, rank + 1)]
    spherical = cartesian_to_spherical(
        cartesian,
        components=components,
        keep_l_in_keys=True,
        remove_blocks_threshold=None,
    )

    for add_inversion in [False, True]:
        rotated_spherical = cartesian_to_spherical(
            transformations.transform_tensormap(cartesian, add_inversion=add_inversion),
            components=components,
            keep_l_in_keys=True,
            remove_blocks_threshold=None,
        )
        expected = transformations.transform_tensormap(
            spherical, add_inversion=add_inversion
        )

        assert mts.equal_metadata(rotated_spherical, expected)
        assert mts.allclose(rotated_spherical, expected)


def _l0_components_from_matrix(A):
    """
    The reference equations for projecting a cartesian rank 2 tensor to the irreducible
    spherical component with lambda = 0.
    """
    # the sign comes from the convention used for Clebsch-Gordan coefficients, which
    # matches featomic
    return -torch.stack([A[0, 0] + A[1, 1] + A[2, 2]]) / math.sqrt(3)


def _l1_components_from_matrix(A):
    """
    The reference equations for projecting a cartesian rank 2 tensor to the irreducible
    spherical component with lambda = 1 and sigma = -1.
    """
    l1_A = torch.stack(
        [
            A[2, 0] - A[0, 2],  # y
            A[0, 1] - A[1, 0],  # z
            A[1, 2] - A[2, 1],  # x
        ]
    )

    return -l1_A / math.sqrt(2)


def _l2_components_from_matrix(A):
    """
    The reference equations for projecting a cartesian rank 2 tensor to the irreducible
    spherical component with lambda = 2.
    """
    l2_A = torch.stack(
        [
            (A[0, 1] + A[1, 0]) / 2.0,
            (A[1, 2] + A[2, 1]) / 2.0,
            (2.0 * A[2, 2] - A[0, 0] - A[1, 1]) / (2.0 * math.sqrt(3.0)),
            (A[0, 2] + A[2, 0]) / 2.0,
            (A[0, 0] - A[1, 1]) / 2.0,
        ]
    )

    return l2_A * math.sqrt(2)


def _l3_components_from_matrix(A):
    """
    The reference equations for projecting a cartesian rank 3 tensor to the irreducible
    spherical component with lambda = 3.
    """
    return torch.stack(
        [
            # o3_mu = -3
            (A[0, 1, 0] + A[1, 0, 0] + A[0, 0, 1] - A[1, 1, 1]) / 2.0,
            # o3_mu = -2
            (
                A[0, 1, 2]
                + A[1, 0, 2]
                + A[1, 2, 0]
                + A[2, 1, 0]
                + A[0, 2, 1]
                + A[2, 0, 1]
            )
            / math.sqrt(6),
            # o3_mu = -1
            (
                4.0 * A[1, 2, 2]
                + 4.0 * A[2, 1, 2]
                + 4.0 * A[2, 2, 1]
                - 3.0 * A[1, 1, 1]
                - A[0, 0, 1]
                - A[0, 1, 0]
                - A[1, 0, 0]
            )
            / math.sqrt(60),
            # o3_mu = 0
            (
                2.0 * A[2, 2, 2]
                - A[0, 2, 0]
                - A[2, 0, 0]
                - A[0, 0, 2]
                - A[1, 2, 1]
                - A[2, 1, 1]
                - A[1, 1, 2]
            )
            / math.sqrt(10),
            # o3_mu = 1
            (
                4.0 * A[0, 2, 2]
                + 4.0 * A[2, 0, 2]
                + 4.0 * A[2, 2, 0]
                - 3.0 * A[0, 0, 0]
                - A[1, 1, 0]
                - A[0, 1, 1]
                - A[1, 0, 1]
            )
            / math.sqrt(60),
            # o3_mu = 2
            (
                A[0, 0, 2]
                + A[0, 2, 0]
                + A[2, 0, 0]
                - A[1, 1, 2]
                - A[1, 2, 1]
                - A[2, 1, 1]
            )
            / math.sqrt(6),
            # o3_mu = 3
            (A[0, 0, 0] - A[1, 1, 0] - A[0, 1, 1] - A[1, 0, 1]) / 2.0,
        ]
    )


def test_cartesian_to_spherical_rank_2_by_equation():
    """
    Tests cartesian_to_spherical for a random rank-2 tensor by comparing the result to
    the result from the reference equations.
    """
    cartesian = cartesian_tensor(rank=2, dtype=torch.float64, device="cpu")
    spherical = cartesian_to_spherical(cartesian, ["xyz_1", "xyz_2"])
    assert spherical.keys.names == ["o3_lambda", "o3_sigma", "_"]
    assert spherical.keys.values[:, :2].tolist() == [[0, 1], [1, -1], [2, 1]]

    values = cartesian.block().values
    for o3_lambda, reference_equation in enumerate(
        [
            _l0_components_from_matrix,
            _l1_components_from_matrix,
            _l2_components_from_matrix,
        ]
    ):
        block = spherical.block(o3_lambda)
        assert block.components == [
            Labels("o3_mu", torch.arange(-o3_lambda, o3_lambda + 1).reshape(-1, 1))
        ]

        for sample in range(values.shape[0]):
            for prop in range(values.shape[-1]):
                reference = reference_equation(values[sample, :, :, prop])
                torch.testing.assert_close(block.values[sample, :, prop], reference)


def test_cartesian_to_spherical_rank_3_by_equation():
    """
    Tests cartesian_to_spherical for a random rank-3 tensor by comparing the result to
    the result from the reference equation.
    """
    cartesian = cartesian_tensor(rank=3, dtype=torch.float64, device="cpu")
    spherical = cartesian_to_spherical(cartesian, ["xyz_1", "xyz_2", "xyz_3"])

    block = spherical.block({"o3_lambda": 3})
    values = cartesian.block().values
    for sample in range(values.shape[0]):
        for prop in range(values.shape[-1]):
            reference = _l3_components_from_matrix(values[sample, :, :, :, prop])
            torch.testing.assert_close(block.values[sample, :, prop], reference)


def test_keep_tensor_info():
    cartesian = cartesian_tensor(rank=2, dtype=torch.float64, device="cpu")
    cartesian.set_info("unit", "eV")

    spherical = cartesian_to_spherical(cartesian, ["xyz_1", "xyz_2"])
    assert spherical.info() == {"unit": "eV"}

    back = spherical_to_cartesian(spherical)
    assert back.info() == {"unit": "eV"}


def test_torchscript(tmpdir, cartesian):
    components = ["xyz_1", "xyz_2", "xyz_3"]
    cg_coefficients = calculate_cg_coefficients(
        3, dtype=torch.float64, device=torch.device("cpu")
    )

    scripted_cartesian_to_spherical = torch.jit.script(cartesian_to_spherical)
    scripted_spherical_to_cartesian = torch.jit.script(spherical_to_cartesian)

    spherical = scripted_cartesian_to_spherical(
        cartesian,
        components,
        cg_coefficients=cg_coefficients,
        remove_blocks_threshold=None,
        keep_l_in_keys=None,
    )
    reference_spherical = cartesian_to_spherical(
        cartesian,
        components,
        remove_blocks_threshold=None,
        keep_l_in_keys=None,
    )
    assert mts.equal_metadata(spherical, reference_spherical)
    assert mts.allclose(spherical, reference_spherical)

    back = scripted_spherical_to_cartesian(
        spherical,
        new_component_names=components,
        cg_coefficients=cg_coefficients,
    )
    assert mts.equal_metadata(back, cartesian)
    assert mts.allclose(back, cartesian)

    path = os.path.join(tmpdir, "c2s.pt")
    torch.jit.save(scripted_cartesian_to_spherical, path)
    loaded_cartesian_to_spherical = torch.jit.load(path)

    path = os.path.join(tmpdir, "s2c.pt")
    torch.jit.save(scripted_spherical_to_cartesian, path)
    loaded_spherical_to_cartesian = torch.jit.load(path)

    spherical = loaded_cartesian_to_spherical(
        cartesian,
        components,
        cg_coefficients=cg_coefficients,
        remove_blocks_threshold=None,
        keep_l_in_keys=None,
    )
    assert mts.equal_metadata(spherical, reference_spherical)
    assert mts.allclose(spherical, reference_spherical)

    back = loaded_spherical_to_cartesian(
        spherical,
        new_component_names=components,
        cg_coefficients=cg_coefficients,
    )
    assert mts.equal_metadata(back, cartesian)
    assert mts.allclose(back, cartesian)

    # rank 1 tensors do not need CG coefficients
    spherical = scripted_cartesian_to_spherical(cartesian, ["xyz_2"])
    back = scripted_spherical_to_cartesian(spherical, new_component_names=["xyz_2"])
    assert mts.allclose(back, cartesian)

    message = (
        "in TorchScript mode, `cg_coefficients` must be pre-computed "
        "and given to this function explicitly"
    )
    with pytest.raises(torch.jit.Error, match=message):
        scripted_cartesian_to_spherical(cartesian, components)

    spherical = cartesian_to_spherical(cartesian, components)
    with pytest.raises(torch.jit.Error, match=message):
        scripted_spherical_to_cartesian(spherical)


def test_gradients_errors():
    cartesian = cartesian_tensor(rank=2, dtype=torch.float64, device="cpu")
    components = ["xyz_1", "xyz_2"]

    message = "gradients are not supported yet by `cartesian_to_spherical`"
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(add_gradients(cartesian), components)

    spherical = cartesian_to_spherical(cartesian, components)
    message = "gradients are not supported yet by `spherical_to_cartesian`"
    with pytest.raises(ValueError, match=message):
        spherical_to_cartesian(add_gradients(spherical))


def test_cartesian_to_spherical_errors(cartesian):
    message = "`components` should be a list, got <class 'str'>"
    with pytest.raises(TypeError, match=message):
        cartesian_to_spherical(cartesian, components="xyz_1")

    message = "'1' is not part of this tensor components"
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components=[1, 2])

    message = "'not_there' is not part of this tensor components"
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components=["not_there"])

    message = (
        "this function only supports consecutive components, "
        "\\['xyz_2', 'xyz_1'\\] are not"
    )
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components=["xyz_2", "xyz_1"])

    message = (
        "this function only supports consecutive components, "
        "\\['xyz_1', 'xyz_3'\\] are not"
    )
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components=["xyz_1", "xyz_3"])

    message = (
        "component 'other' in block for \\(key=0\\) should have \\[0, 1, 2\\] "
        "as values, got \\[0, 1, 2, 3\\] instead"
    )
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components=["other"])

    message = "`keep_l_in_keys` must be `True` for tensors of rank 3 and above"
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(
            cartesian,
            components=["xyz_1", "xyz_2", "xyz_3"],
            keep_l_in_keys=False,
        )

    spherical = cartesian_to_spherical(cartesian, components=["xyz_1"])
    message = (
        "this tensor already has an `o3_lambda` key, is it already in spherical form\\?"
    )
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(spherical, components=["xyz_2"])

    with_l = mts.insert_dimension(
        cartesian, "keys", 0, "l_2", torch.tensor([1, 1], dtype=torch.int32)
    )
    message = "this tensor already has an `l_2` key, is it already in spherical form\\?"
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(with_l, components=["xyz_1", "xyz_2"])


def test_cg_coefficients_errors(cartesian):
    components = ["xyz_1", "xyz_2", "xyz_3"]

    cg_coefficients = calculate_cg_coefficients(
        2, dtype=torch.float64, device=torch.device("cpu")
    )
    message = (
        "`cg_coefficients` only contains coefficients up to lambda=2, "
        "but this tensor requires lambda_max=3"
    )
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components, cg_coefficients=cg_coefficients)

    cg_coefficients = calculate_cg_coefficients(
        3, dtype=torch.float32, device=torch.device("cpu")
    )
    message = "`cg_coefficients` and the tensor must have the same dtype and device"
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components, cg_coefficients=cg_coefficients)

    message = "`cg_coefficients` should be computed with `calculate_cg_coefficients`"
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components, cg_coefficients=cartesian)

    message = "lambda_max must be non-negative, got -1"
    with pytest.raises(ValueError, match=message):
        calculate_cg_coefficients(-1, dtype=torch.float64, device=torch.device("cpu"))

    message = "dtype must be torch.float32 or torch.float64, got torch.int32"
    with pytest.raises(ValueError, match=message):
        calculate_cg_coefficients(2, dtype=torch.int32, device=torch.device("cpu"))


def test_spherical_to_cartesian_errors(cartesian):
    message = (
        "this tensor is missing one of `o3_lambda` or `o3_sigma` in the keys, "
        "is it already in Cartesian form\\?"
    )
    with pytest.raises(ValueError, match=message):
        spherical_to_cartesian(cartesian)

    spherical = cartesian_to_spherical(
        cartesian,
        ["xyz_1", "xyz_2", "xyz_3"],
        remove_blocks_threshold=None,
    )

    message = (
        "this tensor does not have an `o3_mu` component, "
        "is it already in Cartesian form\\?"
    )
    renamed = mts.rename_dimension(spherical, "components", "o3_mu", "other_mu")
    with pytest.raises(ValueError, match=message):
        spherical_to_cartesian(renamed)

    message = (
        "`new_component_names` should have 3 entries \\(the rank of the tensor\\), "
        "got 2"
    )
    with pytest.raises(ValueError, match=message):
        spherical_to_cartesian(spherical, new_component_names=["a", "b"])

    message = "expected a k_1 dimension in the keys"
    renamed = mts.rename_dimension(spherical, "keys", "k_1", "other")
    with pytest.raises(ValueError, match=message):
        spherical_to_cartesian(renamed)

    message = "expected only 1 as value for the l_2 dimension"
    keys = spherical.keys
    l_2 = keys.names.index("l_2")
    values = keys.values.clone()
    values[:, l_2] = 2
    wrong_l = TensorMap(
        Labels(keys.names, values), [block.copy() for block in spherical.blocks()]
    )
    with pytest.raises(ValueError, match=message):
        spherical_to_cartesian(wrong_l)

    rank_1 = cartesian_to_spherical(cartesian, ["xyz_1"], keep_l_in_keys=True)
    keys = rank_1.keys
    values = keys.values.clone()
    values[:, 0] = 2
    wrong_lambda = TensorMap(
        Labels(keys.names, values), [block.copy() for block in rank_1.blocks()]
    )
    message = (
        "found o3_lambda=2 in the keys, which is not possible for a tensor of rank 1 "
        "\\(with 1 `l_\\{i\\}` dimensions in the keys\\)"
    )
    with pytest.raises(ValueError, match=message):
        spherical_to_cartesian(wrong_lambda)

    # rank 3 tensors need the l_{i} dimensions in the keys
    no_l = TensorMap(
        Labels(["o3_lambda", "o3_sigma"], torch.tensor([[3, 1]])),
        [spherical.block({"o3_lambda": 3, "key": 0}).copy()],
    )
    message = "expected a l_1 dimension in the keys"
    with pytest.raises(ValueError, match=message):
        spherical_to_cartesian(no_l)
