import torch

from ..o3 import O3Transformation


def _build_packed_wigner_matrices(
    matrices: torch.Tensor,
    max_o3_lambda: int,
) -> torch.Tensor:
    """Build and pack proper Wigner-D matrices through ``max_o3_lambda``."""
    output_device = matrices.device
    output_dtype = matrices.dtype
    calculation_matrices = matrices.detach().to(device="cpu")
    n_matrices = matrices.size(0)
    n_elements_per_matrix = (
        (max_o3_lambda + 1) * (2 * max_o3_lambda + 1) * (2 * max_o3_lambda + 3) // 3
    )
    packed = torch.empty(
        n_matrices * n_elements_per_matrix,
        dtype=output_dtype,
        device="cpu",
    )

    for matrix_index, matrix in enumerate(calculation_matrices.unbind(0)):
        transformation = O3Transformation(matrix, max_o3_lambda)
        for o3_lambda in range(max_o3_lambda + 1):
            dimension = 2 * o3_lambda + 1
            elements_before = o3_lambda * (4 * o3_lambda * o3_lambda - 1) // 3
            offset = n_matrices * elements_before + matrix_index * dimension * dimension
            packed[offset : offset + dimension * dimension].copy_(
                transformation.wigner_D_matrix(o3_lambda).reshape(-1)
            )

    return packed.to(
        device=output_device,
        dtype=output_dtype,
    )


def _wigner_matrices_for_lambda(
    packed: torch.Tensor,
    n_matrices: int,
    o3_lambda: int,
) -> torch.Tensor:
    """Return the packed Wigner-D stack for one ``o3_lambda`` as a view."""
    # the packed layout is rank-major then matrix-major: all matrices for
    # o3_lambda=0 come first, then all matrices for o3_lambda=1, and so on
    dimension = 2 * o3_lambda + 1
    elements_before = o3_lambda * (4 * o3_lambda * o3_lambda - 1) // 3
    offset = n_matrices * elements_before
    length = n_matrices * dimension * dimension
    if offset + length > packed.numel():
        raise ValueError("o3_lambda exceeds the packed Wigner-D storage")

    return packed[offset : offset + length].view(
        n_matrices,
        dimension,
        dimension,
    )
