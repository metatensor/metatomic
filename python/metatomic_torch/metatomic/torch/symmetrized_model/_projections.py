from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from ..o3 import O3Transformation
from ._utils import (
    _infer_n_systems,
    _key_to_tuple,
    _prepend_system_to_samples,
    _reshape_block_by_local_system,
)


@dataclass
class _ProjectionBlock:
    """Block metadata plus the running quadrature sums of
    ``D^ell(R^{-1}) u(sR)`` (one ``(2 ell + 1, 2 ell + 1, ...)`` tensor per
    ``ell``) for one block of one direct orbit output, where ``R`` is the SO(3)
    representative and ``s`` selects the proper or improper coset."""

    key_names: List[str]
    key_values: torch.Tensor
    sample_names: List[str]
    sample_values: torch.Tensor
    components: List[Labels]
    properties: Labels
    coefficients: Dict[int, torch.Tensor]


# per-output accumulator for character-projection coefficients, keyed by the
# block key (as a tuple of integers)
_ProjectionAccumulator = Dict[Tuple[int, ...], _ProjectionBlock]


def _wigner_stacks(
    transformations: List[O3Transformation], ell_max: int
) -> Dict[int, torch.Tensor]:
    """Stack cached Wigner-D matrices without public defensive copies."""
    return {
        ell: torch.stack([t._wigner_D_matrix(ell) for t in transformations])
        for ell in range(ell_max + 1)
    }


def _wigner_stacks_from_matrices(
    matrices: torch.Tensor,
    ell_max: int,
    *,
    is_inverted: bool,
    output_device: torch.device,
    output_dtype: torch.dtype,
) -> Dict[int, torch.Tensor]:
    """Build one validated Wigner batch on CPU and transfer it in stacks.

    Wigner-D construction is CPU-backed. Recovering Euler angles separately
    from CUDA matrices would therefore synchronize the device for every
    rotation and transfer every small matrix separately. Staging the current
    matrix batch once keeps the same calculation while requiring only one
    device-to-host transfer per batch and one host-to-device transfer per rank.
    """
    calculation_matrices = matrices.detach().to(
        device=torch.device("cpu"), dtype=output_dtype
    )
    transformations = [
        O3Transformation._from_validated_matrix(
            matrix,
            ell_max,
            is_inverted=is_inverted,
        )
        for matrix in calculation_matrices.unbind(0)
    ]
    cpu_stacks = _wigner_stacks(transformations, ell_max)
    return {
        ell: stack.to(device=output_device, dtype=output_dtype)
        for ell, stack in cpu_stacks.items()
    }


def _compute_batch_projection_contributions(
    tensor: TensorMap,
    weights: torch.Tensor,
    wigner_matrices: Dict[int, torch.Tensor],
    max_o3_lambda_character: int,
) -> _ProjectionAccumulator:
    """Compute one batch's contribution to the projection coefficients.

    For every block and every ``ell`` up to ``max_o3_lambda_character``, this
    computes the weighted partial quadrature sum
    ``sum_R w_R D^ell_mn(R^{-1}) u(sR)`` over the batch's SO(3)
    representatives ``R`` for one proper or improper coset ``s``. These are
    the Fourier coefficients of the direct (un-back-rotated) orbit output
    ``u`` in the convention used by the public convolution formula; the
    inversion parity is inserted only when the two cosets are finalized.
    Coefficients have shape ``(2 ell + 1, 2 ell + 1)`` in front of the block's
    sample/component/property axes. Block metadata is carried along so the
    final TensorMap can be rebuilt once all batches are merged.
    """
    n_local_systems = weights.numel()
    block_contributions: _ProjectionAccumulator = {}
    for key, block in tensor.items():
        key_tuple = _key_to_tuple(key)
        values, sample_names, sample_values = _reshape_block_by_local_system(
            block, n_local_systems
        )
        weight = weights.to(dtype=values.dtype, device=values.device)
        weighted_values = (
            weight.view([weight.shape[0]] + [1] * (values.ndim - 1)) * values
        )

        coefficients: Dict[int, torch.Tensor] = {}
        for ell in range(max_o3_lambda_character + 1):
            D = wigner_matrices[ell].to(dtype=values.dtype, device=values.device)
            coefficients[ell] = torch.einsum("imn,i...->mn...", D, weighted_values)

        block_contributions[key_tuple] = _ProjectionBlock(
            key_names=list(tensor.keys.names),
            key_values=key.values.clone(),
            sample_names=sample_names,
            sample_values=sample_values.clone(),
            components=list(block.components),
            properties=block.properties,
            coefficients=coefficients,
        )

    return block_contributions


def _merge_projection_contributions(
    accumulator: _ProjectionAccumulator,
    contribution: _ProjectionAccumulator,
) -> None:
    """Add a batch's coefficient sums into ``accumulator`` in place.

    Entries are matched by block key; within an entry, the per-``ell``
    quadrature sums are added element-wise (the quadrature is linear, so
    batches can be accumulated in any order).
    """
    for key_tuple, entry in contribution.items():
        if key_tuple not in accumulator:
            accumulator[key_tuple] = entry
            continue
        existing = accumulator[key_tuple].coefficients
        for ell, tensor in entry.coefficients.items():
            existing[ell] = existing[ell] + tensor


def _accumulate_batch(
    positive_accumulators: Dict[str, _ProjectionAccumulator],
    negative_accumulators: Dict[str, _ProjectionAccumulator],
    decomposed_direct: Dict[str, TensorMap],
    weights: torch.Tensor,
    wigner_stacks: Dict[int, torch.Tensor],
    max_o3_lambda_character: int,
    inversion: int,
) -> None:
    """Accumulate one batch of direct (un-back-rotated) outputs into the
    character-projection sums for the proper (+1) or improper (-1) coset."""
    accumulators = positive_accumulators if inversion == 1 else negative_accumulators
    for name, tensor in decomposed_direct.items():
        contribution = _compute_batch_projection_contributions(
            tensor,
            weights,
            wigner_stacks,
            max_o3_lambda_character,
        )
        _merge_projection_contributions(accumulators.setdefault(name, {}), contribution)


def _finalize_system(
    positive_accumulators: Dict[str, _ProjectionAccumulator],
    negative_accumulators: Dict[str, _ProjectionAccumulator],
    system_index: int,
    max_o3_lambda_character: int,
) -> Dict[str, TensorMap]:
    """Turn one system's accumulated coefficients into character-projection
    TensorMaps, keyed by output name."""
    results: Dict[str, TensorMap] = {}
    for name in set(positive_accumulators) | set(negative_accumulators):
        tensor = _finalize_projection_tensor(
            positive_accumulators.get(name, {}),
            negative_accumulators.get(name, {}),
            system_index,
            max_o3_lambda_character,
        )
        if tensor is not None:
            results[name] = tensor
    return results


def _finalize_projection_tensor(
    positive: _ProjectionAccumulator,
    negative: _ProjectionAccumulator,
    system_index: int,
    max_o3_lambda_character: int,
) -> Optional[TensorMap]:
    """Combine the two coset sums into squared isotypical-projection norms.

    ``positive``/``negative`` hold the quadrature sums of
    ``D^ell(R^{-1}) u(R)`` and ``D^ell(R^{-1}) u(-R)``, respectively, with
    ``R`` in SO(3). For each ``ell`` and ``sigma``, the O(3) isotypical
    projection inserts the inversion parity and combines them as
    ``plus + sigma * (-1)^ell * minus``, and its squared norm is
    ``(2 ell + 1) / 4 * sum_mn combined^2``, one value per sample. Results are
    packed into a TensorMap keyed by the original block keys extended with
    ``chi_lambda``/``chi_sigma`` (``None`` if there is nothing to finalize).
    """
    all_keys = list(positive.keys())
    for key in negative.keys():
        if key not in positive:
            all_keys.append(key)

    if len(all_keys) == 0:
        return None

    blocks: List[TensorBlock] = []
    key_values: List[torch.Tensor] = []
    key_names: Optional[List[str]] = None
    for key_tuple in all_keys:
        plus_entry = positive.get(key_tuple)
        minus_entry = negative.get(key_tuple)
        meta = plus_entry if plus_entry is not None else minus_entry
        assert meta is not None

        key_names = list(meta.key_names)
        key_tensor = meta.key_values
        plus_coeffs = plus_entry.coefficients if plus_entry is not None else {}
        minus_coeffs = minus_entry.coefficients if minus_entry is not None else {}

        for ell in range(max_o3_lambda_character + 1):
            plus_tensor = plus_coeffs.get(ell)
            minus_tensor = minus_coeffs.get(ell)
            if plus_tensor is None and minus_tensor is None:
                continue
            if plus_tensor is None:
                plus_tensor = torch.zeros_like(minus_tensor)
            if minus_tensor is None:
                minus_tensor = torch.zeros_like(plus_tensor)

            parity = (-1) ** ell
            for sigma in [1, -1]:
                combined = plus_tensor + sigma * parity * minus_tensor
                values = (
                    0.25 * (2 * ell + 1) * torch.sum(combined * combined, dim=(0, 1))
                )
                blocks.append(
                    TensorBlock(
                        values=values,
                        samples=_prepend_system_to_samples(
                            meta.sample_names,
                            meta.sample_values,
                            system_index,
                            device=values.device,
                        ),
                        components=meta.components,
                        properties=meta.properties,
                    )
                )
                key_values.append(
                    torch.cat(
                        [
                            key_tensor,
                            torch.tensor(
                                [ell, sigma],
                                dtype=key_tensor.dtype,
                                device=key_tensor.device,
                            ),
                        ]
                    )
                )

    assert key_names is not None
    tensor = TensorMap(
        Labels(key_names + ["chi_lambda", "chi_sigma"], torch.stack(key_values)),
        blocks,
    )
    if "_" in tensor.keys.names:
        tensor = mts.remove_dimension(tensor, "keys", "_")
    return tensor


def per_system_character_fractions(
    outputs: Dict[str, TensorMap],
    name: str,
    n_systems: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reduce the ``<name>_character_projection`` entry of a
    :py:class:`SymmetrizedModel` output dictionary to per-system fractions of
    the squared norm in each :math:`O(3)` isotypical sector.

    For each system and each ``(chi_lambda, chi_sigma)`` key, the projection is
    summed over all samples, components and properties belonging to the system,
    and normalized by the per-system total squared norm (from the
    ``<name>_norm_squared`` entry). Fractions sum to 1 only when the response is
    resolved by the grid and lies entirely within the included character
    sectors.

    Negative or non-finite squared norms and sector weights are rejected rather
    than repaired: they indicate inconsistent input or an unresolved finite
    quadrature. A zero total norm has no mathematical fractions; this helper
    returns all-zero fractions when every sector weight is also zero, and rejects
    a nonzero sector weight paired with a zero norm.

    :param outputs: dictionary returned by :py:class:`SymmetrizedModel` with
        ``compute_character_projections=True``
    :param name: name of the output to reduce, without the
        ``_character_projection`` suffix (e.g. ``"energy_l0"``)
    :param n_systems: number of systems the output was computed for. If ``None``
        (default), inferred from the largest ``system`` sample index; trailing
        systems that contributed no samples are then silently missing from the
        result, so pass it explicitly whenever systems can be empty.
    :return: ``(sigma_plus, sigma_minus, lambdas)`` where the first two tensors
        have shape ``(n_systems, n_lambda)`` and hold the fractions for
        ``chi_sigma = +1`` and ``chi_sigma = -1`` respectively, and ``lambdas``
        holds the sorted ``chi_lambda`` values
    """
    character_projection = outputs[name + "_character_projection"]
    norm_squared = outputs[name + "_norm_squared"]

    if n_systems is None:
        n_systems = _infer_n_systems([character_projection, norm_squared])

    dtype = character_projection.block(0).values.dtype
    device = character_projection.block(0).values.device

    norm = torch.zeros(n_systems, dtype=dtype, device=device)
    for block in norm_squared.blocks():
        if bool(torch.any((~torch.isfinite(block.values)) | (block.values < 0)).item()):
            raise ValueError("squared norm must be finite and non-negative")
        system = block.samples.column("system").to(dtype=torch.long, device=device)
        per_sample = torch.flatten(block.values, start_dim=1).sum(dim=1)
        norm.index_add_(0, system, per_sample.to(dtype=dtype, device=device))
    if bool(torch.any((~torch.isfinite(norm)) | (norm < 0)).item()):
        raise ValueError("squared norm must be finite and non-negative")

    lambdas = torch.unique(character_projection.keys.column("chi_lambda"))
    lambda_to_index = {int(ell.item()): i for i, ell in enumerate(lambdas)}

    sigma_plus = torch.zeros(n_systems, len(lambdas), dtype=dtype, device=device)
    sigma_minus = torch.zeros(n_systems, len(lambdas), dtype=dtype, device=device)
    for key, block in character_projection.items():
        if bool(torch.any((~torch.isfinite(block.values)) | (block.values < 0)).item()):
            raise ValueError(
                "character-projection weights must be finite and non-negative"
            )
        target = sigma_plus if int(key["chi_sigma"]) == 1 else sigma_minus
        column = lambda_to_index[int(key["chi_lambda"])]
        system = block.samples.column("system").to(dtype=torch.long, device=device)
        per_sample = torch.flatten(block.values, start_dim=1)
        target[:, column].index_add_(0, system, per_sample.sum(dim=1))

    sector_total = sigma_plus.sum(dim=1) + sigma_minus.sum(dim=1)
    zero_norm = norm == 0
    if bool(torch.any(zero_norm & (sector_total != 0)).item()):
        raise ValueError("zero squared norm is inconsistent with nonzero sector weight")
    denominator = torch.where(zero_norm, torch.ones_like(norm), norm)
    return (
        sigma_plus / denominator.unsqueeze(1),
        sigma_minus / denominator.unsqueeze(1),
        lambdas,
    )
