"""
Clebsch-Gordan coefficients for real spherical harmonics, and the kernels using them to
couple a product of two spherical harmonics into a single one, and back.
"""

from typing import List, Optional

import numpy as np
import torch
import wigners
from metatensor.torch import Labels, TensorBlock, TensorMap

from ._utils import validate_integer
from ._wigner import _complex_to_real_spherical_harmonics_transform


def calculate_cg_coefficients(
    lambda_max: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> TensorMap:
    """
    Calculate the Clebsch-Gordan coefficients coupling two real spherical harmonics of
    degrees ``l1`` and ``l2`` into a single real spherical harmonic of degree
    ``lambda``, for all ``l1``, ``l2`` and ``lambda`` up to ``lambda_max``.

    These coefficients are used by :py:func:`cartesian_to_spherical` and
    :py:func:`spherical_to_cartesian`, and must be computed in advance and given to
    these functions explicitly when calling them from TorchScript. Converting a tensor
    of rank ``N`` requires ``lambda_max >= N``.

    The coefficients are stored in a :py:class:`~metatensor.torch.TensorMap` with
    ``l1``, ``l2`` and ``lambda`` keys. Each block contains a single sample (``_``), the
    ``m1``, ``m2`` and ``mu`` components (with values from ``-l`` to ``l`` for the
    corresponding degree), and a single property (``cg_coefficient``).

    :param lambda_max: non-negative maximal degree of the spherical harmonics
    :param dtype: dtype of the coefficients, either :py:attr:`torch.float32` or
        :py:attr:`torch.float64`
    :param device: device on which the coefficients should be stored
    :return: :py:class:`~metatensor.torch.TensorMap` containing the Clebsch-Gordan
        coefficients
    """
    lambda_max = validate_integer("lambda_max", lambda_max, 0)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}")

    keys: List[List[int]] = []
    blocks: List[TensorBlock] = []
    for l1 in range(lambda_max + 1):
        for l2 in range(lambda_max + 1):
            for o3_lambda in range(abs(l1 - l2), min(lambda_max, l1 + l2) + 1):
                coefficients = torch.tensor(
                    _real_cg_coefficients(l1, l2, o3_lambda),
                    dtype=dtype,
                    device=device,
                )
                keys.append([l1, l2, o3_lambda])
                blocks.append(
                    TensorBlock(
                        values=coefficients.reshape(
                            1, 2 * l1 + 1, 2 * l2 + 1, 2 * o3_lambda + 1, 1
                        ),
                        samples=Labels(
                            "_", torch.zeros((1, 1), dtype=torch.int32, device=device)
                        ),
                        components=[
                            Labels(
                                "m1",
                                torch.arange(
                                    -l1, l1 + 1, dtype=torch.int32, device=device
                                ).reshape(-1, 1),
                                assume_unique=True,
                            ),
                            Labels(
                                "m2",
                                torch.arange(
                                    -l2, l2 + 1, dtype=torch.int32, device=device
                                ).reshape(-1, 1),
                                assume_unique=True,
                            ),
                            Labels(
                                "mu",
                                torch.arange(
                                    -o3_lambda,
                                    o3_lambda + 1,
                                    dtype=torch.int32,
                                    device=device,
                                ).reshape(-1, 1),
                                assume_unique=True,
                            ),
                        ],
                        properties=Labels(
                            "cg_coefficient",
                            torch.zeros((1, 1), dtype=torch.int32, device=device),
                        ),
                    )
                )

    return TensorMap(
        Labels(
            ["l1", "l2", "lambda"],
            torch.tensor(keys, dtype=torch.int32, device=device),
            assume_unique=True,
        ),
        blocks,
    )


def _real_cg_coefficients(l1: int, l2: int, o3_lambda: int) -> np.ndarray:
    """
    Get the Clebsch-Gordan coefficients for real spherical harmonics, as an array of
    shape ``(2 * l1 + 1, 2 * l2 + 1, 2 * o3_lambda + 1)``.
    """
    complex_cg = wigners.clebsch_gordan_array(l1, l2, o3_lambda)

    real_cg = (
        _complex_to_real_spherical_harmonics_transform(l1).conj()
        @ complex_cg.reshape(2 * l1 + 1, -1)
    ).reshape(complex_cg.shape)

    real_cg = real_cg.swapaxes(0, 1)
    real_cg = (
        _complex_to_real_spherical_harmonics_transform(l2).conj()
        @ real_cg.reshape(2 * l2 + 1, -1)
    ).reshape(real_cg.shape)
    real_cg = real_cg.swapaxes(0, 1)

    # the transformation is applied as T @ C along the last axis, i.e. C @ T^T
    real_cg = real_cg @ _complex_to_real_spherical_harmonics_transform(o3_lambda).T

    # The result is either purely real or purely imaginary, depending on the parity of
    # l1 + l2 + lambda.
    if (l1 + l2 + o3_lambda) % 2 == 0:
        return real_cg.real
    else:
        return -real_cg.imag


def resolve_cg_coefficients(
    cg_coefficients: Optional[TensorMap],
    lambda_max: int,
    values: torch.Tensor,
) -> TensorMap:
    """
    Get the CG coefficients to use for a tensor requiring up to ``lambda_max``.

    The coefficients are computed if ``cg_coefficients`` is ``None`` (which is only
    possible outside of TorchScript), and checked against ``lambda_max`` and the dtype
    and device of ``values`` otherwise.
    """
    if cg_coefficients is None:
        if torch.jit.is_scripting():
            raise ValueError(
                "in TorchScript mode, `cg_coefficients` must be pre-computed "
                "and given to this function explicitly"
            )
        else:
            return calculate_cg_coefficients(
                lambda_max, dtype=values.dtype, device=values.device
            )

    if cg_coefficients.keys.names != ["l1", "l2", "lambda"]:
        raise ValueError(
            "`cg_coefficients` should be computed with `calculate_cg_coefficients`"
        )

    available_lambda_max = int(torch.max(cg_coefficients.keys.column("lambda")))
    if available_lambda_max < lambda_max:
        raise ValueError(
            "`cg_coefficients` only contains coefficients up to "
            f"lambda={available_lambda_max}, but this tensor requires "
            f"lambda_max={lambda_max}"
        )

    reference = cg_coefficients.block(0).values
    if reference.dtype != values.dtype or reference.device != values.device:
        raise ValueError(
            "`cg_coefficients` and the tensor must have the same dtype and device"
        )

    return cg_coefficients


def cg_coupling_matrix(
    cg_coefficients: TensorMap,
    l1: int,
    l2: int,
    o3_lambdas: List[int],
    cartesian_1: bool,
    cartesian_2: bool,
) -> torch.Tensor:
    """
    Get the coefficients coupling ``l1`` and ``l2`` to all the ``o3_lambdas`` as a
    single matrix, with shape ``((2 * l1 + 1) * (2 * l2 + 1), sum(2 * L + 1))``.

    If ``cartesian_1`` (respectively ``cartesian_2``) is ``True``, the coefficients act
    on the ``l1=1`` (respectively ``l2=1``) axis in the Cartesian ``(x, y, z)`` order,
    instead of the spherical ``(mu=-1, 0, 1) = (y, z, x)`` order. This allows to work
    directly with Cartesian arrays, without re-ordering them first.

    :param cg_coefficients: coefficients from :py:func:`calculate_cg_coefficients`
    :param l1: degree of the first spherical harmonic
    :param l2: degree of the second spherical harmonic
    :param o3_lambdas: degrees ``L`` of the coupled spherical harmonics
    :param cartesian_1: is the first axis using the Cartesian order?
    :param cartesian_2: is the second axis using the Cartesian order?
    """
    matrices: List[torch.Tensor] = []
    for o3_lambda in o3_lambdas:
        block = cg_coefficients.block({"l1": l1, "l2": l2, "lambda": o3_lambda})
        matrices.append(block.values[0, :, :, :, 0])
    matrix = torch.cat(matrices, dim=2)

    # (x, y, z) is the same as mu = (1, -1, 0), so rolling the coefficients by one along
    # a l=1 axis makes them act on the Cartesian order
    if cartesian_1:
        assert l1 == 1
        matrix = matrix.roll(1, 0)

    if cartesian_2:
        assert l2 == 1
        matrix = matrix.roll(1, 1)

    return matrix.reshape((2 * l1 + 1) * (2 * l2 + 1), matrix.shape[2])


def cg_couple_array(
    array: torch.Tensor,
    o3_lambdas: List[int],
    coefficients: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Go from an uncoupled product basis that behave like a product of spherical harmonics
    to a coupled basis that behaves like a single spherical harmonic.

    The two middle axes of ``array`` behave like spherical harmonics of degrees ``l1``
    and ``l2``, and are projected onto spherical harmonics of degree ``L`` for each
    ``L`` in ``o3_lambdas``, using Clebsch-Gordan coefficients:

    ``T^{L}_{M} = \\sum_{m1 m2} C^{l1 l2 L}_{m1 m2 M} T^{l1}_{m1} T^{l2}_{m2}``

    :param array: array with shape ``(n_s, 2 * l1 + 1, 2 * l2 + 1, n_p)``
    :param o3_lambdas: degrees of the spherical harmonics to compute
    :param coefficients: coefficients for ``l1``, ``l2`` and all ``o3_lambdas``, from
        :py:func:`cg_coupling_matrix`
    :return: one array with shape ``(n_s, 2 * L + 1, n_p)`` for each ``L`` in
        ``o3_lambdas``
    """
    assert len(array.shape) == 4

    n_samples = array.shape[0]
    n_m1 = array.shape[1]
    n_m2 = array.shape[2]
    n_properties = array.shape[3]

    # move properties next to samples, so the projection on all o3_lambda is a single
    # matrix product: (n_s, 2*l1 + 1, 2*l2 + 1, n_p) => (n_s * n_p, (2*l1+1)*(2*l2+1))
    array = array.permute(0, 3, 1, 2).reshape(n_samples * n_properties, n_m1 * n_m2)
    coupled = array @ coefficients

    results: List[torch.Tensor] = []
    start = 0
    for o3_lambda in o3_lambdas:
        n_mu = 2 * o3_lambda + 1
        # (n_s * n_p, 2 * o3_lambda + 1) => (n_s, 2 * o3_lambda + 1, n_p)
        result = coupled[:, start : start + n_mu]
        result = result.reshape(n_samples, n_properties, n_mu).transpose(1, 2)
        results.append(result)
        start += n_mu

    return results


def cg_uncouple_array(
    arrays: List[torch.Tensor],
    l1: int,
    l2: int,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """
    Exact inverse of :py:func:`cg_couple_array`, going from a coupled basis that
    behaves like a single spherical harmonic back to the uncoupled product basis that
    behaves like a product of two spherical harmonics of degrees ``l1`` and ``l2``.

    The Clebsch-Gordan coefficients form an orthogonal transformation over the full set
    of ``L`` in ``range(|l1 - l2|, l1 + l2 + 1)``, which is inverted by its transpose:

    ``T^{l1}_{m1} T^{l2}_{m2} = \\sum_{L M} C^{l1 l2 L}_{m1 m2 M} T^{L}_{M}``

    :param arrays: arrays with shape ``(n_s, 2 * L + 1, n_p)``. ``n_s`` and ``n_p``
        must be the same for all arrays. Any missing ``L`` is treated as an array of
        zeros.
    :param l1: degree of the first spherical harmonic to recover
    :param l2: degree of the second spherical harmonic to recover
    :param coefficients: coefficients for ``l1``, ``l2`` and the ``L`` of all
        ``arrays`` (in the same order), from :py:func:`cg_coupling_matrix`
    :return: array with shape ``(n_s, 2 * l1 + 1, 2 * l2 + 1, n_p)``
    """
    assert len(arrays) > 0

    n_samples = arrays[0].shape[0]
    n_properties = arrays[0].shape[2]

    # move properties next to samples and concatenate all the arrays, so the sum over
    # o3_lambda is a single matrix product: (n_s, 2 * L + 1, n_p) for all L =>
    # (n_s * n_p, sum(2 * L + 1))
    transposed: List[torch.Tensor] = []
    for array in arrays:
        transposed.append(array.transpose(1, 2))
    coupled = torch.cat(transposed, dim=2).reshape(n_samples * n_properties, -1)

    output = coupled @ coefficients.t()

    # (n_s * n_p, (2*l1 + 1) * (2*l2 + 1)) => (n_s, 2*l1 + 1, 2*l2 + 1, n_p)
    output = output.reshape(n_samples, n_properties, 2 * l1 + 1, 2 * l2 + 1)
    return output.permute(0, 2, 3, 1)
