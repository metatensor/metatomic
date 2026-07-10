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


# per-output accumulator for character-projection coefficients: maps a block key
# (as a tuple of integers) to the block metadata and the running quadrature sums
# of D^ell(g) x(g) for each ell
_ProjectionAccumulator = Dict[Tuple[int, ...], Dict[str, object]]


def _wigner_stacks(
    transformations: List[O3Transformation], ell_max: int
) -> Dict[int, torch.Tensor]:
    """Stack the Wigner-D matrices of a batch of transformations per ell."""
    return {
        ell: torch.stack([t.wigner_D_matrix(ell) for t in transformations])
        for ell in range(ell_max + 1)
    }


def _compute_batch_projection_contributions(
    tensor: TensorMap,
    weights: torch.Tensor,
    wigner_matrices: Dict[int, torch.Tensor],
    max_o3_lambda_character: int,
    *,
    storage_device: Optional[torch.device] = None,
) -> _ProjectionAccumulator:
    """Compute one batch's contribution to the projection coefficients.

    For every block and every ``ell`` up to ``max_o3_lambda_character``, this
    computes the weighted partial quadrature sum ``sum_g w_g D^ell_mn(g) x(g)``
    over the batch of group elements ``g``, i.e. the Fourier coefficients of
    the direct (un-back-rotated) output ``x`` seen as a function on the group.
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
            coefficient = torch.einsum("imn,i...->mn...", D, weighted_values)
            if storage_device is not None and coefficient.device != storage_device:
                coefficient = coefficient.to(device=storage_device)
            coefficients[ell] = coefficient

        key_values = key.values.clone()
        sample_values_out = sample_values.clone()
        components = list(block.components)
        properties = block.properties
        if storage_device is not None:
            key_values = key_values.to(device=storage_device)
            sample_values_out = sample_values_out.to(device=storage_device)
            components = [
                component.to(device=storage_device) for component in block.components
            ]
            properties = block.properties.to(device=storage_device)

        block_contributions[key_tuple] = {
            "key_names": list(tensor.keys.names),
            "key_values": key_values,
            "sample_names": sample_names,
            "sample_values": sample_values_out,
            "components": components,
            "properties": properties,
            "coefficients": coefficients,
        }

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
        existing = accumulator[key_tuple]
        existing_coefficients = existing["coefficients"]
        contribution_coefficients = entry["coefficients"]
        assert isinstance(existing_coefficients, dict)
        assert isinstance(contribution_coefficients, dict)
        for ell, tensor in contribution_coefficients.items():
            if ell in existing_coefficients:
                existing_coefficients[ell] = existing_coefficients[ell] + tensor
            else:
                existing_coefficients[ell] = tensor


def _accumulate_batch(
    positive_accumulators: Dict[str, _ProjectionAccumulator],
    negative_accumulators: Dict[str, _ProjectionAccumulator],
    decomposed_direct: Dict[str, TensorMap],
    weights: torch.Tensor,
    wigner_stacks: Dict[int, torch.Tensor],
    max_o3_lambda_character: int,
    inversion: int,
    storage_device: Optional[torch.device],
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
            storage_device=storage_device,
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

    ``positive``/``negative`` hold the quadrature sums of ``D^ell(g) x(g)``
    over the proper rotations and over the rotations composed with the
    inversion, respectively. For each ``ell`` and ``sigma``, the O(3) isotypical
    projection combines them as ``plus + sigma * (-1)^ell * minus``, and its
    squared norm is ``(2 ell + 1) / 4 * sum_mn combined^2``, one value per
    sample. Results are packed into a TensorMap keyed by the original block
    keys extended with ``chi_lambda``/``chi_sigma`` (``None`` if there is
    nothing to finalize).
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

        key_names = list(meta["key_names"])
        key_tensor = meta["key_values"]
        sample_names = meta["sample_names"]
        sample_values = meta["sample_values"]
        components = meta["components"]
        properties = meta["properties"]
        plus_coeffs = plus_entry["coefficients"] if plus_entry is not None else {}
        minus_coeffs = minus_entry["coefficients"] if minus_entry is not None else {}

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
                            sample_names,
                            sample_values,
                            system_index,
                            device=values.device,
                        ),
                        components=components,
                        properties=properties,
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
    ``<name>_norm_squared`` entry). For an output lying entirely within the
    resolved sectors, the fractions sum to 1 over all
    ``(chi_lambda, chi_sigma)``.

    :param outputs: dictionary returned by :py:class:`SymmetrizedModel` with
        ``compute_character_projections=True``
    :param name: name of the output to reduce, without the
        ``_character_projection`` suffix (e.g. ``"energy_l0"``)
    :param n_systems: number of systems the output was computed for. If ``None``
        (default), inferred from the largest ``system`` sample index; trailing
        systems that contributed no samples are then silently missing from the
        result, so pass it explicitly whenever systems can be empty.
    :return: ``(proper, improper, lambdas)`` where ``proper`` and ``improper``
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
        system = block.samples.column("system").to(dtype=torch.long, device=device)
        per_sample = block.values.reshape(block.values.shape[0], -1).sum(dim=1)
        norm.index_add_(0, system, per_sample.to(dtype=dtype, device=device))
    norm = torch.clamp(torch.abs(norm), min=torch.finfo(dtype).tiny)

    lambdas = torch.unique(character_projection.keys.column("chi_lambda"))
    lambda_to_index = {int(ell.item()): i for i, ell in enumerate(lambdas)}

    proper = torch.zeros(n_systems, len(lambdas), dtype=dtype, device=device)
    improper = torch.zeros(n_systems, len(lambdas), dtype=dtype, device=device)
    for key, block in character_projection.items():
        target = proper if int(key["chi_sigma"]) == 1 else improper
        column = lambda_to_index[int(key["chi_lambda"])]
        system = block.samples.column("system").to(dtype=torch.long, device=device)
        per_sample = torch.abs(block.values).reshape(block.values.shape[0], -1)
        target[:, column].index_add_(0, system, per_sample.sum(dim=1))

    return proper / norm.unsqueeze(1), improper / norm.unsqueeze(1), lambdas
