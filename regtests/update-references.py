#!/usr/bin/env python3
"""
Create or update the reference outputs of the regression tests.

    python update-references.py --list
    python update-references.py pet-mad-s-v1.5.0
    python update-references.py --all --force
    python update-references.py --all --relock
    python update-references.py --check
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
VENVS_PATH = os.path.join(ROOT, "build", "venvs", "references")

# torch publishes CPU-only wheels on its own index, and they are much smaller
DEFAULT_EXTRA_INDEX = "https://download.pytorch.org/whl/cpu"

# Maximal size of a reference file, in bytes
MAX_REFERENCE_SIZE = 100 * 1024

# What the engine needs to run a model.
DEPENDENCIES = [
    "metatomic-torch",
    "metatensor-operations",
    "vesin",
    "torch==2.13.*",
    "numpy",
]


def all_cases():
    cases = glob.glob(os.path.join(ROOT, "references", "*", "input.json"))
    return sorted(os.path.basename(os.path.dirname(case)) for case in cases)


def case_directory(case: str) -> str:
    return os.path.join(ROOT, "references", case)


def lock_path(case: str) -> str:
    return os.path.join(case_directory(case), "references.lock")


def read_lock(case: str) -> dict:
    path = lock_path(case)
    if not os.path.exists(path):
        return {}

    with open(path) as fd:
        return json.load(fd)


def file_hash(path: str) -> str:
    with open(path, "rb") as fd:
        return hashlib.sha256(fd.read()).hexdigest()


def input_hash(case: str) -> str:
    """
    Checksum of the ``input.json`` of a case.

    The tolerances are left out of it: how closely an output has to match its reference
    says nothing about whether the reference itself is still the right answer, so
    changing one should not ask for a regeneration. The rest is hashed in a canonical
    form, so that reformatting the file does not either.
    """
    with open(os.path.join(case_directory(case), "input.json")) as fd:
        data = json.load(fd)

    for output in data["outputs"].values():
        output.pop("rtol", None)
        output.pop("atol", None)

    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def reference_hashes(case: str) -> dict:
    """Checksum of every reference ``.mts`` file of a case"""
    directory = case_directory(case)
    with open(os.path.join(directory, "input.json")) as fd:
        outputs = json.load(fd)["outputs"]

    return {
        name: file_hash(os.path.join(directory, name))
        for name in sorted({output["reference"] for output in outputs.values()})
    }


def python_version(python: str) -> str:
    version = subprocess.run(
        [python, "-c", "import platform; print(platform.python_version())"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    return ".".join(version.split(".")[:2])


def venv_python(directory: str) -> str:
    if os.name == "nt":
        return os.path.join(directory, "Scripts", "python.exe")

    return os.path.join(directory, "bin", "python")


def environment_marker(dependencies: list, python: str) -> str:
    """What an environment holds, as the string stored inside it."""
    return json.dumps({"python": python_version(python), "dependencies": dependencies})


def venv_directory(dependencies: list, python: str) -> str:
    """Where the environment holding ``dependencies`` lives."""
    digest = hashlib.sha256(environment_marker(dependencies, python).encode())
    return os.path.join(VENVS_PATH, digest.hexdigest()[:12])


def create_venv(dependencies: list, python: str) -> str:
    """
    Get a virtual environment with exactly ``dependencies`` installed in it, creating it
    if we don't have it already.
    """
    marker = environment_marker(dependencies, python)
    directory = venv_directory(dependencies, python)
    stamp = os.path.join(directory, "regtests-environment.json")

    if os.path.exists(stamp) and open(stamp).read() == marker:
        return directory

    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.makedirs(VENVS_PATH, exist_ok=True)
    print(f"+++++ creating a virtualenv in {directory}")
    subprocess.run([python, "-m", "venv", directory], check=True)

    environment = dict(os.environ)
    environment.setdefault("PIP_EXTRA_INDEX_URL", DEFAULT_EXTRA_INDEX)

    pip = [venv_python(directory), "-m", "pip", "--disable-pip-version-check"]
    subprocess.run(pip + ["install", "--upgrade", "pip"], check=True, env=environment)

    print("+++++ installing dependencies")
    subprocess.run(pip + ["install"] + list(dependencies), check=True, env=environment)

    with open(stamp, "w") as fd:
        fd.write(marker)

    return directory


def settle_venv(directory: str, dependencies: list, python: str) -> str:
    """
    Move an environment to the name its *installed* dependencies give it.

    An environment is built from what the lock asks for, which the first time around is
    the unpinned ``DEPENDENCIES``; the lock then records what pip actually installed.
    Those are different names for the same environment, so without this the one we just
    built is not the one the lock points at: it would be pruned right after being used,
    and built again from scratch on the next run.
    """
    target = venv_directory(dependencies, python)
    if target == directory:
        return directory

    if os.path.exists(target):
        # another case already settled on this exact environment
        shutil.rmtree(directory)
        return target

    os.replace(directory, target)
    with open(os.path.join(target, "regtests-environment.json"), "w") as fd:
        fd.write(environment_marker(dependencies, python))

    return target


def freeze(directory: str) -> list:
    """The exact version of everything installed in the environment."""
    output = subprocess.run(
        [venv_python(directory), "-m", "pip", "freeze", "--disable-pip-version-check"],
        check=True,
        capture_output=True,
        text=True,
    )

    return sorted(line for line in output.stdout.splitlines() if line.strip())


def prune_venvs():
    """
    Delete the environments that no case refers to any more.

    Regenerating a case for the first time moves it from the unpinned dependencies to
    the ones its lock now holds, which is a different environment; without this the
    first one would stay around forever.
    """
    keep = {
        venv_directory(read_lock(case)["dependencies"], sys.executable)
        for case in all_cases()
        if read_lock(case)
    }

    if not os.path.exists(VENVS_PATH):
        return

    for name in sorted(os.listdir(VENVS_PATH)):
        directory = os.path.join(VENVS_PATH, name)
        if directory not in keep:
            print(f"+++++ removing the unused environment {name}")
            shutil.rmtree(directory, ignore_errors=True)


def up_to_date(case: str, locked: dict) -> bool:
    """
    Whether the references of ``case`` are the ones its current ``input.json`` produced.

    This is what the lock's checksums are for: without them we could only tell that the
    reference files exist, not that they still answer the question being asked.
    """
    if "input_sha256" not in locked or "references_sha256" not in locked:
        return False

    if locked["input_sha256"] != input_hash(case):
        return False

    directory = case_directory(case)
    for name, digest in locked["references_sha256"].items():
        path = os.path.join(directory, name)
        if not os.path.exists(path) or file_hash(path) != digest:
            return False

    return True


def regenerate(case: str, force: bool, relock: bool, python: str) -> bool:
    """Generate the references of ``case`` in the environment its lock describes."""
    print(f"+++++ {case}:")

    locked = read_lock(case)
    if locked and not force and not relock and up_to_date(case, locked):
        print("+++++ already generated from the current input.json")
        return True

    if locked and not relock:
        dependencies = locked["dependencies"]
        print(f"+++++ using the {len(dependencies)} dependencies in references.lock")
    else:
        dependencies = DEPENDENCIES
        if locked:
            print("+++++ re-resolving the dependencies (--relock)")

    directory = create_venv(dependencies, python)

    command = [venv_python(directory), os.path.abspath(__file__), "--worker", case]
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        return False

    installed = freeze(directory)
    directory = settle_venv(directory, installed, python)

    with open(lock_path(case), "w") as fd:
        json.dump(
            {
                "input_sha256": input_hash(case),
                "references_sha256": reference_hashes(case),
                "python": python_version(venv_python(directory)),
                "dependencies": installed,
            },
            fd,
            indent=2,
        )
        fd.write("\n")

    return True


def deviation(actual, expected) -> str:
    """Summarize how far ``actual`` is from ``expected``."""
    lines = []
    for key, expected_block in expected.items():
        try:
            actual_block = actual.block(key)
        except Exception:
            return "+++++     metadata changed, can not compare values"

        if actual_block.values.shape != expected_block.values.shape:
            return "+++++     shape changed, can not compare values"

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
                lines.append(f"+++++     {name} gradients: missing in the new output")

        for what, new, old in arrays:
            if new.shape != old.shape:
                lines.append(f"+++++     {what}: shape changed")
                continue

            delta = (new.double() - old.double()).abs()
            scale = old.double().abs().clamp(min=1e-100)
            lines.append(
                f"+++++     {what}: max absolute {delta.max():.3e}, "
                f"max relative {(delta / scale).max():.3e}"
            )

    return "\n".join(lines)


def check(case: str, models: dict) -> list:
    """Check that the reference files match what ``input.json`` asks for"""
    import metatomic_regtest as regtest

    directory = case_directory(case)
    try:
        input = regtest.load_input(os.path.join(directory, "input.json"))
    except Exception as e:
        return [f"input.json can not be read: {e}"]

    problems = []
    if input.model not in models:
        problems.append(f"uses the model '{input.model}', which is not in models.json")

    regenerate_it = f"run `python regtests/update-references.py {case} --force`"
    locked = read_lock(case)
    if not locked:
        problems.append(
            "references.lock is missing, so there is no record of what generated the "
            f"references: {regenerate_it}"
        )
    elif locked["input_sha256"] != input_hash(case):
        problems.append(
            f"input.json has changed since the references were generated: "
            f"{regenerate_it}"
        )

    recorded = locked.get("references_sha256", {})
    referenced = set()
    for name, output in input.outputs.items():
        reference = output["reference"]
        referenced.add(reference)

        path = os.path.join(directory, reference)
        if not os.path.exists(path):
            problems.append(
                f"the reference for '{name}' is missing ({reference}): run "
                f"`python regtests/update-references.py {case}`"
            )
            continue

        if reference in recorded and recorded[reference] != file_hash(path):
            problems.append(
                f"{reference} is not the file that was generated: {regenerate_it}"
            )

        size = os.path.getsize(path)
        if size > MAX_REFERENCE_SIZE:
            problems.append(
                f"{reference} is {size / 1024:.0f} kiB, more than the "
                f"{MAX_REFERENCE_SIZE // 1024} kiB a reference may take: use a smaller "
                "system, or ask for fewer outputs"
            )

    for path in sorted(glob.glob(os.path.join(directory, "*.mts"))):
        name = os.path.basename(path)
        if name not in referenced:
            problems.append(f"{name} is not used by any output of input.json")

    return problems


def generate(case: str) -> bool:
    import metatensor.torch as mts

    import metatomic_regtest as regtest

    directory = case_directory(case)
    input = regtest.load_input(os.path.join(directory, "input.json"))

    existing = {}
    for name, output in input.outputs.items():
        path = os.path.join(directory, output["reference"])
        if os.path.exists(path):
            existing[name] = mts.load(path)

    print(f"+++++ {case}: running {input.model}")
    model = regtest.load_model(input.model)
    outputs = regtest.run_model(model, input)

    for name, output in input.outputs.items():
        path = os.path.join(directory, output["reference"])
        if name in existing:
            print(f"+++++   {name}: deviation from the previous reference")
            print(deviation(outputs[name], existing[name]))
        else:
            print(f"+++++   {name}: new reference")

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
        "--check",
        action="store_true",
        help="check that every case has the reference data input.json describes, "
        "without running any model (every case, unless some are named)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate the references even if they are already up to date",
    )
    parser.add_argument(
        "--relock",
        action="store_true",
        help="re-resolve the dependencies instead of using the locked ones",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="python interpreter used to create the virtualenv",
    )
    parser.add_argument(
        # how `regenerate` runs the generation inside the environment it just prepared
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    # show the messages from this process and any subprocess in the correct order
    sys.stdout.reconfigure(line_buffering=True)

    if args.list:
        for case in all_cases():
            print(case)
        return 0

    if args.all or (args.check and not args.cases):
        cases = all_cases()
    elif args.cases:
        cases = args.cases
    else:
        parser.error("give at least one case, or --all")

    for case in cases:
        if case not in all_cases():
            known = "\n".join(f"    - {name}" for name in all_cases())
            parser.error(f"unknown case '{case}', references/ contains:\n{known}")

    if args.check:
        with open(os.path.join(ROOT, "models.json")) as fd:
            models = json.load(fd)

        failed = False
        for case in cases:
            for message in check(case, models):
                failed = True
                print(f"ERROR: {case}: {message}", file=sys.stderr)

        if failed:
            print("\nsome reference data is missing or out of date", file=sys.stderr)
            return 1

        print(f"+++++ all {len(cases)} cases have their reference data")
        return 0

    if args.worker:
        return 0 if all(generate(case) for case in cases) else 1

    ok = True
    for case in cases:
        ok &= regenerate(case, args.force, args.relock, args.python)

    prune_venvs()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
