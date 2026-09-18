#!/usr/bin/env python3
"""
Create or update the reference outputs of the regression tests.

    python update-references.py --list
    python update-references.py references/pet-mad-s-v1.5.0
    python update-references.py --all --force
"""

import argparse
import glob
import os
import sys

import metatensor.torch as mts
import torch

import metatomic_regtest as regtest


ROOT = os.path.dirname(os.path.abspath(__file__))


def all_cases():
    cases = glob.glob(os.path.join(ROOT, "references", "*", "input.json"))
    return sorted(os.path.relpath(os.path.dirname(case), ROOT) for case in cases)


def deviation(actual: mts.TensorMap, expected: mts.TensorMap) -> str:
    """Summarize how far ``actual`` is from ``expected``."""
    lines = []
    for key, expected_block in expected.items():
        try:
            actual_block = actual.block(key)
        except Exception:
            return "        metadata changed, can not compare values"

        if actual_block.values.shape != expected_block.values.shape:
            return "        shape changed, can not compare values"

        arrays = [("values", actual_block.values, expected_block.values)]
        for name, gradient in expected_block.gradients():
            try:
                arrays.append(
                    (
                        f"{name} gradients",
                        actual_block.gradient(name).values,
                        gradient.values,
                    )
                )
            except Exception:
                lines.append(f"        {name} gradients: missing in the new output")

        for what, new, old in arrays:
            if new.shape != old.shape:
                lines.append(f"        {what}: shape changed")
                continue

            delta = (new.double() - old.double()).abs()
            scale = old.double().abs().clamp(min=torch.finfo(torch.float64).tiny)
            lines.append(
                f"        {what}: max absolute {delta.max():.3e}, "
                f"max relative {(delta / scale).max():.3e}"
            )

    return "\n".join(lines)


def generate(case: str, force: bool) -> bool:
    directory = os.path.join(ROOT, case)
    input = regtest.load_input(os.path.join(directory, "input.json"))

    existing = {}
    for name, output in input.outputs.items():
        path = os.path.join(directory, output["reference"])
        if os.path.exists(path):
            existing[name] = mts.load(path)

    if existing and not force:
        print(f"{case}: references already exist, use --force to overwrite")
        return False

    print(f"{case}: running {input.model}")
    model = regtest.load_model(input.model)
    outputs = regtest.run_model(model, input)

    for name, output in input.outputs.items():
        path = os.path.join(directory, output["reference"])
        if name in existing:
            print(f"    {name}: deviation from the previous reference")
            print(deviation(outputs[name], existing[name]))
        else:
            print(f"    {name}: new reference")

        mts.save(path, outputs[name])

    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cases", nargs="*", help="cases to regenerate")
    parser.add_argument("--all", action="store_true", help="regenerate every case")
    parser.add_argument("--list", action="store_true", help="list the known cases")
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing references"
    )
    args = parser.parse_args()

    if args.list:
        for case in all_cases():
            print(case)
        return 0

    if args.all:
        cases = all_cases()
    elif args.cases:
        cases = [os.path.relpath(os.path.abspath(case), ROOT) for case in args.cases]
    else:
        parser.error("give at least one case, or --all")

    ok = True
    for case in cases:
        ok &= generate(case, force=args.force)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
