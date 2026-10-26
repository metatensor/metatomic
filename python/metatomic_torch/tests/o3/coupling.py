import metatensor.torch as mts
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch.o3 import (
    calculate_cg_coefficients,
    cartesian_to_spherical,
    coupled_to_uncoupled,
    random_transformations,
    uncoupled_to_coupled,
)

from .._tests_utils import can_use_mps_backend


ALL_DEVICE_DTYPE = [("cpu", "float64"), ("cpu", "float32")]

if torch.cuda.is_available():
    ALL_DEVICE_DTYPE.append(("cuda", "float64"))
    ALL_DEVICE_DTYPE.append(("cuda", "float32"))

if can_use_mps_backend():
    ALL_DEVICE_DTYPE.append(("mps", "float32"))


# (o3_lambda, o3_sigma) of the spherical components in each block, for tensors with
# one, two and three "xyz" components
IRREPS = {
    1: [[(0, 1)], [(1, 1)], [(2, -1)]],
    2: [
        [(0, 1), (0, 1)],
        [(1, 1), (1, 1)],
        [(1, -1), (2, 1)],
        [(2, 1), (1, -1)],
        [(2, 1), (2, 1)],
    ],
    3: [
        [(1, 1), (1, 1), (1, 1)],
        [(0, 1), (2, -1), (1, 1)],
        [(2, 1), (1, -1), (2, 1)],
    ],
}


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


def uncoupled_tensor(irreps, other_component=False, dtype=torch.float64, device="cpu"):
    """
    Create a tensor in uncoupled form with random values, with one block for each entry
    in ``irreps``, containing the ``(o3_lambda, o3_sigma)`` of each spherical component.
    """
    n_components = len(irreps[0])
    key_names = []
    for i in range(1, n_components + 1):
        key_names += [f"o3_lambda_{i}", f"o3_sigma_{i}"]

    keys = []
    blocks = []
    for block_irreps in irreps:
        key = []
        shape = [5]
        components = []
        if other_component:
            shape.append(2)
            components.append(Labels.range("other", 2))

        for i, (o3_lambda, o3_sigma) in enumerate(block_irreps, start=1):
            key += [o3_lambda, o3_sigma]
            shape.append(2 * o3_lambda + 1)
            components.append(
                Labels(
                    f"o3_mu_{i}",
                    torch.arange(-o3_lambda, o3_lambda + 1).reshape(-1, 1),
                )
            )
        shape.append(3)

        keys.append(key)
        blocks.append(
            TensorBlock(
                values=torch.rand(shape, dtype=dtype),
                samples=Labels.range("system", 5),
                components=components,
                properties=Labels.range("p", 3),
            )
        )

    return TensorMap(Labels(key_names, torch.tensor(keys)), blocks).to(device)


def test_coupled_keys():
    uncoupled = uncoupled_tensor(IRREPS[2])
    coupled = uncoupled_to_coupled(
        uncoupled, ["o3_mu_1", "o3_mu_2"], remove_blocks_threshold=None
    )

    # the degree and parity of the coupled components are kept as `l_i` and `s_i`
    assert coupled.keys.names == ["o3_lambda", "o3_sigma", "l_1", "s_1", "l_2", "s_2"]
    assert coupled.component_names == ["o3_mu"]

    # all the possible o3_lambda are created for each block
    expected_n_blocks = 0
    for (l1, _), (l2, _) in IRREPS[2]:
        expected_n_blocks += 2 * min(l1, l2) + 1
    assert len(coupled) == expected_n_blocks

    for key, block in coupled.items():
        o3_lambda, o3_sigma, l1, s1, l2, s2 = [int(v) for v in key.values]
        assert abs(l1 - l2) <= o3_lambda <= l1 + l2
        assert o3_sigma == s1 * s2 * (-1) ** (l1 + l2 + o3_lambda)
        assert block.components == [
            Labels("o3_mu", torch.arange(-o3_lambda, o3_lambda + 1).reshape(-1, 1))
        ]

    uncoupled = uncoupled_tensor(IRREPS[3])
    coupled = uncoupled_to_coupled(
        uncoupled, ["o3_mu_1", "o3_mu_2", "o3_mu_3"], remove_blocks_threshold=None
    )
    assert coupled.keys.names == [
        "o3_lambda",
        "o3_sigma",
        "k_1",
        "l_1",
        "s_1",
        "l_2",
        "s_2",
        "l_3",
        "s_3",
    ]


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("device,dtype", ALL_DEVICE_DTYPE)
def test_roundtrip(rank, device, dtype):
    dtype = getattr(torch, dtype)
    uncoupled = uncoupled_tensor(
        IRREPS[rank],
        other_component=True,
        dtype=dtype,
        device=device,
    )

    components = [f"o3_mu_{i}" for i in range(1, rank + 1)]
    coupled = uncoupled_to_coupled(
        uncoupled,
        components,
        # keep every block so the transformation stays exactly invertible
        remove_blocks_threshold=None,
    )
    assert coupled.component_names == ["other", "o3_mu"]
    assert coupled.device.type == device
    assert coupled.block(0).values.dtype == dtype

    back = coupled_to_uncoupled(coupled, components)

    assert mts.equal_metadata(back, uncoupled)
    if dtype == torch.float64:
        assert mts.allclose(back, uncoupled, atol=1e-12, rtol=1e-12)
    else:
        assert mts.allclose(back, uncoupled, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_equivariance(rank):
    uncoupled = uncoupled_tensor(IRREPS[rank])
    transformations = random_transformations(
        n=1, max_angular_momentum=6, dtype=torch.float64, device="cpu"
    )

    components = [f"o3_mu_{i}" for i in range(1, rank + 1)]
    coupled = uncoupled_to_coupled(uncoupled, components, remove_blocks_threshold=None)

    for add_inversion in [False, True]:
        transformed_coupled = uncoupled_to_coupled(
            transformations.transform_tensormap(uncoupled, add_inversion=add_inversion),
            components,
            remove_blocks_threshold=None,
        )
        expected = transformations.transform_tensormap(
            coupled, add_inversion=add_inversion
        )

        assert mts.equal_metadata(transformed_coupled, expected)
        assert mts.allclose(transformed_coupled, expected)


@pytest.mark.parametrize("rank", [2, 3])
def test_same_as_cartesian_to_spherical(rank):
    # a Cartesian tensor is the same as a product of o3_lambda=1, o3_sigma=1 spherical
    # harmonics, once the (x, y, z) axes are re-ordered to (y, z, x) = (mu=-1, 0, 1)
    values = torch.rand((5,) + (3,) * rank + (2,), dtype=torch.float64)

    cartesian = TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values=values,
                samples=Labels.range("system", 5),
                components=[Labels.range(f"xyz_{i}", 3) for i in range(1, rank + 1)],
                properties=Labels.range("p", 2),
            )
        ],
    )
    from_cartesian = cartesian_to_spherical(
        cartesian,
        [f"xyz_{i}" for i in range(1, rank + 1)],
        keep_l_in_keys=True,
        remove_blocks_threshold=None,
    )

    key_names = []
    for i in range(1, rank + 1):
        key_names += [f"o3_lambda_{i}", f"o3_sigma_{i}"]

    mu = torch.arange(-1, 2).reshape(-1, 1)
    uncoupled = TensorMap(
        Labels(key_names, torch.ones((1, 2 * rank), dtype=torch.int32)),
        [
            TensorBlock(
                values=values.roll(shifts=[-1] * rank, dims=list(range(1, rank + 1))),
                samples=Labels.range("system", 5),
                components=[Labels(f"o3_mu_{i}", mu) for i in range(1, rank + 1)],
                properties=Labels.range("p", 2),
            )
        ],
    )
    from_uncoupled = uncoupled_to_coupled(
        uncoupled,
        [f"o3_mu_{i}" for i in range(1, rank + 1)],
        remove_blocks_threshold=None,
    )

    # both functions use the same coupling order, and produce blocks in the same order
    assert len(from_cartesian) == len(from_uncoupled)
    if rank == 2:
        names = ["o3_lambda", "o3_sigma"]
    else:
        names = ["o3_lambda", "o3_sigma", "k_1"]

    for i in range(len(from_cartesian)):
        cartesian_key = from_cartesian.keys.entry(i)
        uncoupled_key = from_uncoupled.keys.entry(i)
        for name in names:
            assert cartesian_key[name] == uncoupled_key[name]

        torch.testing.assert_close(
            from_cartesian.block(i).values, from_uncoupled.block(i).values
        )


def test_remove_blocks():
    # the product of a vector with itself is symmetric, so the o3_lambda=1 block is zero
    vector = torch.rand((5, 3, 2), dtype=torch.float64)
    mu = torch.arange(-1, 2).reshape(-1, 1)
    uncoupled = TensorMap(
        Labels(
            ["o3_lambda_1", "o3_sigma_1", "o3_lambda_2", "o3_sigma_2"],
            torch.tensor([[1, 1, 1, 1]]),
        ),
        [
            TensorBlock(
                values=vector[:, :, None, :] * vector[:, None, :, :],
                samples=Labels.range("system", 5),
                components=[Labels("o3_mu_1", mu), Labels("o3_mu_2", mu)],
                properties=Labels.range("p", 2),
            )
        ],
    )

    coupled = uncoupled_to_coupled(uncoupled, ["o3_mu_1", "o3_mu_2"])
    assert coupled.keys.column("o3_lambda").tolist() == [0, 2]

    # missing blocks are treated as zeros
    back = coupled_to_uncoupled(coupled, ["o3_mu_1", "o3_mu_2"])
    assert mts.equal_metadata(back, uncoupled)
    assert mts.allclose(back, uncoupled)


def test_keep_tensor_info():
    uncoupled = uncoupled_tensor(IRREPS[2])
    uncoupled.set_info("unit", "eV")

    coupled = uncoupled_to_coupled(uncoupled, ["o3_mu_1", "o3_mu_2"])
    assert coupled.info() == {"unit": "eV"}

    back = coupled_to_uncoupled(coupled, ["o3_mu_1", "o3_mu_2"])
    assert back.info() == {"unit": "eV"}


def test_torchscript():
    uncoupled = uncoupled_tensor(IRREPS[3])
    components = ["o3_mu_1", "o3_mu_2", "o3_mu_3"]
    cg_coefficients = calculate_cg_coefficients(
        5, dtype=torch.float64, device=torch.device("cpu")
    )

    scripted_uncoupled_to_coupled = torch.jit.script(uncoupled_to_coupled)
    scripted_coupled_to_uncoupled = torch.jit.script(coupled_to_uncoupled)

    coupled = scripted_uncoupled_to_coupled(
        uncoupled, components, cg_coefficients=cg_coefficients
    )
    reference = uncoupled_to_coupled(uncoupled, components)
    assert mts.equal_metadata(coupled, reference)
    assert mts.allclose(coupled, reference)

    back = scripted_coupled_to_uncoupled(
        coupled, components, cg_coefficients=cg_coefficients
    )
    assert mts.equal_metadata(back, uncoupled)
    assert mts.allclose(back, uncoupled)

    message = (
        "in TorchScript mode, `cg_coefficients` must be pre-computed "
        "and given to this function explicitly"
    )
    with pytest.raises(torch.jit.Error, match=message):
        scripted_uncoupled_to_coupled(uncoupled, components)

    with pytest.raises(torch.jit.Error, match=message):
        scripted_coupled_to_uncoupled(coupled, components)


def test_gradients_errors():
    uncoupled = uncoupled_tensor(IRREPS[2])
    components = ["o3_mu_1", "o3_mu_2"]

    message = (
        "gradients are not supported yet by `uncoupled_to_coupled`, but the block for "
        "\\(o3_lambda_1=0, o3_sigma_1=1, o3_lambda_2=0, o3_sigma_2=1\\) has "
        "gradients with respect to \\['positions'\\]"
    )
    with pytest.raises(ValueError, match=message):
        uncoupled_to_coupled(add_gradients(uncoupled), components)

    coupled = uncoupled_to_coupled(uncoupled, components)
    message = "gradients are not supported yet by `coupled_to_uncoupled`"
    with pytest.raises(ValueError, match=message):
        coupled_to_uncoupled(add_gradients(coupled), components)


def test_uncoupled_to_coupled_errors():
    uncoupled = uncoupled_tensor(IRREPS[2], other_component=True)

    message = "`components` should be a list, got <class 'str'>"
    with pytest.raises(TypeError, match=message):
        uncoupled_to_coupled(uncoupled, "o3_mu_1")

    message = "'not_there' is not part of this tensor components"
    with pytest.raises(ValueError, match=message):
        uncoupled_to_coupled(uncoupled, ["not_there"])

    message = (
        "this function only supports consecutive components, "
        "\\['o3_mu_2', 'o3_mu_1'\\] are not"
    )
    with pytest.raises(ValueError, match=message):
        uncoupled_to_coupled(uncoupled, ["o3_mu_2", "o3_mu_1"])

    message = "spherical components should be named 'o3_mu_\\{suffix\\}', got 'other'"
    with pytest.raises(ValueError, match=message):
        uncoupled_to_coupled(uncoupled, ["other", "o3_mu_1"])

    renamed = mts.rename_dimension(uncoupled, "keys", "o3_sigma_2", "sigma")
    message = "expected `o3_lambda_2` and `o3_sigma_2` in the keys for the 'o3_mu_2'"
    with pytest.raises(ValueError, match=message):
        uncoupled_to_coupled(renamed, ["o3_mu_1", "o3_mu_2"])

    keys = uncoupled.keys
    values = keys.values.clone()
    values[:, keys.names.index("o3_sigma_1")] = 2
    wrong_sigma = TensorMap(
        Labels(keys.names, values), [block.copy() for block in uncoupled.blocks()]
    )
    message = "expected all o3_sigma to be -1 or 1, got 2 in block for"
    with pytest.raises(ValueError, match=message):
        uncoupled_to_coupled(wrong_sigma, ["o3_mu_1", "o3_mu_2"])

    values = keys.values.clone()
    values[:, keys.names.index("o3_lambda_1")] += 1
    wrong_lambda = TensorMap(
        Labels(keys.names, values), [block.copy() for block in uncoupled.blocks()]
    )
    message = "component 'o3_mu_1' in block for .* should have values from -1 to 1"
    with pytest.raises(ValueError, match=message):
        uncoupled_to_coupled(wrong_lambda, ["o3_mu_1", "o3_mu_2"])

    coupled = uncoupled_to_coupled(uncoupled, ["o3_mu_1", "o3_mu_2"])
    message = (
        "this tensor already has `o3_lambda` or `o3_sigma` keys, "
        "is it already in coupled form\\?"
    )
    with pytest.raises(ValueError, match=message):
        uncoupled_to_coupled(coupled, ["o3_mu"])

    with_l = mts.insert_dimension(
        uncoupled, "keys", 0, "l_2", torch.zeros(len(uncoupled), dtype=torch.int32)
    )
    message = "this tensor already has a `l_2` key, is it already in coupled form\\?"
    with pytest.raises(ValueError, match=message):
        uncoupled_to_coupled(with_l, ["o3_mu_1", "o3_mu_2"])


def test_coupled_to_uncoupled_errors():
    uncoupled = uncoupled_tensor(IRREPS[3])
    components = ["o3_mu_1", "o3_mu_2", "o3_mu_3"]

    message = (
        "this tensor is missing one of `o3_lambda` or `o3_sigma` in the keys, "
        "is it already in uncoupled form\\?"
    )
    with pytest.raises(ValueError, match=message):
        coupled_to_uncoupled(uncoupled, components)

    coupled = uncoupled_to_coupled(uncoupled, components, remove_blocks_threshold=None)

    renamed = mts.rename_dimension(coupled, "components", "o3_mu", "other_mu")
    message = (
        "this tensor does not have an `o3_mu` component, "
        "is it already in uncoupled form\\?"
    )
    with pytest.raises(ValueError, match=message):
        coupled_to_uncoupled(renamed, components)

    renamed = mts.rename_dimension(coupled, "keys", "k_1", "other")
    message = "expected a k_1 dimension in the keys"
    with pytest.raises(ValueError, match=message):
        coupled_to_uncoupled(renamed, components)

    message = "expected `l_4` and `s_4` in the keys for the 'o3_mu_4' component"
    with pytest.raises(ValueError, match=message):
        coupled_to_uncoupled(coupled, ["o3_mu_2", "o3_mu_3", "o3_mu_4"])

    with_lambda = mts.insert_dimension(
        coupled,
        "keys",
        0,
        "o3_lambda_1",
        torch.zeros(len(coupled), dtype=torch.int32),
    )
    message = (
        "this tensor already has a `o3_lambda_1` key, "
        "is it already in uncoupled form\\?"
    )
    with pytest.raises(ValueError, match=message):
        coupled_to_uncoupled(with_lambda, components)
