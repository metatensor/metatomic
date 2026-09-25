"""Regression tests for models exported by previous versions of metatomic"""

import glob
import os
import sys

import metatensor.torch as mts
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES = sorted(glob.glob(os.path.join(ROOT, "references", "*", "input.json")))
CASE_IDS = [os.path.basename(os.path.dirname(case)) for case in CASES]

sys.path.append(ROOT)

import metatomic_regtest  # noqa E401


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
            raise AssertionError(
                f"Output '{name}' of model '{input.model}' changed in case '{case}'"
            ) from e
