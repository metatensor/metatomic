"""Regression tests for models exported by previous versions of metatomic"""

import glob
import os
import sys

import metatensor.torch as mts
import pytest
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES = sorted(glob.glob(os.path.join(ROOT, "references", "*", "input.json")))
CASE_IDS = [os.path.basename(os.path.dirname(case)) for case in CASES]

sys.path.append(ROOT)

import metatomic_regtest  # noqa E401


def _print_block_mismatches(prefix, block, expected_block, rtol, atol, max_elements):
    """
    Print the metadata and values of the first ``max_elements`` elements that are not
    close between ``block`` and ``expected_block``.
    """
    if block.samples != expected_block.samples:
        print(f"{prefix} samples differ")
        return

    if block.properties != expected_block.properties:
        print(f"{prefix} properties differ")
        return

    if block.components != expected_block.components:
        print(f"{prefix} components differ")
        return

    values = block.values.detach().cpu()
    expected_values = expected_block.values.detach().cpu().to(values.dtype)
    if values.shape != expected_values.shape:
        print(f"{prefix} shape {values.shape} != expected {expected_values.shape}")
        return

    diff = torch.abs(values - expected_values)
    not_close = diff > atol + rtol * torch.abs(expected_values)
    n_failing = int(not_close.sum())
    if n_failing == 0:
        return

    print(
        f"{prefix} {n_failing}/{values.numel()} elements not close, "
        f"max abs diff={float(diff.max()) if diff.numel() > 0 else 0.0:.2g}"
    )

    for index in torch.nonzero(not_close)[:max_elements]:
        index = index.tolist()
        sample = block.samples.entry(index[0]).print()
        components = [
            block.components[c].entry(i).print() for c, i in enumerate(index[1:-1])
        ]
        property = block.properties.entry(index[-1]).print()
        print(
            f"    sample={sample} components=[{', '.join(components)}] "
            f"property={property}: actual={float(values[tuple(index)]):g} "
            f"expected={float(expected_values[tuple(index)]):g} "
            f"diff={float(diff[tuple(index)]):.2g}"
        )


def _print_mismatches(name, actual, expected, rtol, atol, max_elements=5):
    """
    Debug helper: print the environment, and the first ``max_elements`` elements
    that are not close between ``actual`` and ``expected`` for each block and
    gradient.
    """
    print(f"\n===== mismatches for '{name}' (rtol={rtol:1.0e}, atol={atol:1.0e}) =====")
    if actual.keys != expected.keys:
        print(f"keys differ:\n  actual={actual.keys}\n  expected={expected.keys}")
        return

    for key, block in actual.items():
        expected_block = expected.block(key)
        _print_block_mismatches(
            f"key={key.print()}", block, expected_block, rtol, atol, max_elements
        )

        gradients = block.gradients_list()
        if set(gradients) != set(expected_block.gradients_list()):
            print(
                f"key={key.print()} gradients differ: {gradients} != "
                f"{expected_block.gradients_list()}"
            )
            continue

        for parameter in gradients:
            _print_block_mismatches(
                f"key={key.print()} gradient '{parameter}'",
                block.gradient(parameter),
                expected_block.gradient(parameter),
                rtol,
                atol,
                max_elements,
            )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_regression(case):
    input = metatomic_regtest.load_input(case)
    model = metatomic_regtest.load_model(input.model)
    outputs = metatomic_regtest.run_model(model, input)
    expected = metatomic_regtest.load_expected(input)

    for name in input.outputs:
        try:
            mts.allclose_raise(
                outputs[name],
                expected[name],
                rtol=input.outputs[name]["rtol"],
                atol=input.outputs[name]["atol"],
            )
        except Exception as e:
            _print_mismatches(
                name,
                outputs[name],
                expected[name],
                rtol=input.outputs[name]["rtol"],
                atol=input.outputs[name]["atol"],
            )
            raise AssertionError(
                f"Output '{name}' of model '{input.model}' changed in '{case}'"
            ) from e
