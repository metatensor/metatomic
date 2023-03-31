import torch

from equistore.torch import Labels, TensorBlock, TensorMap


def tensor():
    """A dummy tensor map to be used in tests"""
    block_1 = TensorBlock(
        values=torch.full((3, 1, 1), 1.0),
        samples=Labels(["s"], torch.IntTensor([[0], [2], [4]])),
        components=[Labels(["c"], torch.IntTensor([[0]]))],
        properties=Labels(["p"], torch.IntTensor([[0]])),
    )
    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            samples=Labels(["sample", "g"], torch.IntTensor([[0, -2], [2, 3]])),
            values=torch.full((2, 1, 1), 11.0),
            components=block_1.components,
            properties=block_1.properties,
        ),
    )

    block_2 = TensorBlock(
        values=torch.full((3, 1, 3), 2.0),
        samples=Labels(["s"], torch.IntTensor([[0], [1], [3]])),
        components=[Labels(["c"], torch.IntTensor([[0]]))],
        properties=Labels(["p"], torch.IntTensor([[3], [4], [5]])),
    )
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.full((3, 1, 3), 12.0),
            samples=Labels(
                ["sample", "g"], torch.IntTensor([[0, -2], [0, 3], [2, -2]])
            ),
            components=block_2.components,
            properties=block_2.properties,
        ),
    )

    block_3 = TensorBlock(
        values=torch.full((4, 3, 1), 3.0),
        samples=Labels(["s"], torch.IntTensor([[0], [3], [6], [8]])),
        components=[Labels(["c"], torch.IntTensor([[0], [1], [2]]))],
        properties=Labels(["p"], torch.IntTensor([[0]])),
    )
    block_3.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.full((1, 3, 1), 13.0),
            samples=Labels(["sample", "g"], torch.IntTensor([[1, -2]])),
            components=block_3.components,
            properties=block_3.properties,
        ),
    )

    block_4 = TensorBlock(
        values=torch.full((4, 3, 1), 4.0),
        samples=Labels(["s"], torch.IntTensor([[0], [1], [2], [5]])),
        components=[Labels(["c"], torch.IntTensor([[0], [1], [2]]))],
        properties=Labels(["p"], torch.IntTensor([[0]])),
    )
    block_4.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.full((2, 3, 1), 14.0),
            samples=Labels(["sample", "g"], torch.IntTensor([[0, 1], [3, 3]])),
            components=block_4.components,
            properties=block_4.properties,
        ),
    )

    keys = Labels(
        names=["key_1", "key_2"],
        values=torch.IntTensor([[0, 0], [1, 0], [2, 2], [2, 3]]),
    )

    return TensorMap(keys, [block_1, block_2, block_3, block_4])
