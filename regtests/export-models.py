#!/usr/bin/env python3
"""
Build the models used by the regression tests, and lock the environment used to build
them.

    python regtests/export-models.py --list
    python regtests/export-models.py pet-mad-s-v1.5.0
    python regtests/export-models.py --all
    python regtests/export-models.py --all --relock
    python regtests/export-models.py --all --upload
    python regtests/export-models.py --check

``--check`` verifies, without building anything, that every model is built from the
current ``models.json`` and can be downloaded. This is what the test suite runs before
the tests themselves.

``--upload`` publishes the models it built to HuggingFace and records their address in
``models.lock``, so that everyone else can run the tests without building anything. It
needs the ``huggingface_hub`` package and write access to the repository, through
``HF_TOKEN`` or a previous ``hf auth login``.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request


ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(ROOT, "models.json")
LOCK_PATH = os.path.join(ROOT, "models.lock")
# Models are built here, and *not* in the `cache/` the tests download into so both CI
# and local tests use the exact same file.
BUILD_PATH = os.path.join(ROOT, "build")
# one subdirectory per kind of virtualenv, so that each script only ever prunes the
# ones it made: update-references.py keeps its own under build/venvs/references
VENVS_PATH = os.path.join(BUILD_PATH, "venvs", "models")

# torch publishes CPU-only wheels on its own index, and they are much smaller
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# where `--upload` publishes the models
DEFAULT_HF_REPO = "metatensor/metatomic-regtests"


def save_lock(data: dict):
    with open(LOCK_PATH, "w") as fd:
        json.dump(data, fd, indent=2, sort_keys=True)
        fd.write("\n")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fd:
        while chunk := fd.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def venv_python(directory: str) -> str:
    if os.name == "nt":
        return os.path.join(directory, "Scripts", "python.exe")

    return os.path.join(directory, "bin", "python")


def create_venv(name: str, dependencies: list, python: str) -> str:
    """
    Create the virtual environment used to build ``name``, and install
    ``dependencies`` in it.

    The environment is recreated from scratch every time: a leftover environment from a
    previous set of dependencies would silently change what ends up in the model.
    """
    directory = os.path.join(VENVS_PATH, name)
    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.makedirs(VENVS_PATH, exist_ok=True)
    print(f"+++++ creating a virtualenv in {directory}")
    subprocess.run([python, "-m", "venv", directory], check=True)

    environment = dict(os.environ)
    environment.setdefault("PIP_EXTRA_INDEX_URL", TORCH_CPU_INDEX)

    pip = [venv_python(directory), "-m", "pip", "--disable-pip-version-check"]
    subprocess.run(
        pip + ["--quiet", "install", "--upgrade", "pip"], check=True, env=environment
    )

    print("+++++ installing dependencies")
    subprocess.run(pip + ["install"] + list(dependencies), check=True, env=environment)

    return directory


def freeze(directory: str) -> list:
    """The exact version of everything installed in the environment."""
    output = subprocess.run(
        [venv_python(directory), "-m", "pip", "freeze", "--disable-pip-version-check"],
        check=True,
        capture_output=True,
        text=True,
    )

    return sorted(line for line in output.stdout.splitlines() if line.strip())


# Patch the metatomic export to always record the same date, so that the checksum of a
# model does not change every time it is rebuilt.
FROZEN_DATE = "1970-01-01T00:00:00+00:00"

PRELUDE = f"""
import datetime as _datetime
import types as _types

import metatomic.torch.model as _model


_FROZEN = _datetime.datetime.fromisoformat({FROZEN_DATE!r})


class _FrozenDatetime(_datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return _FROZEN if tz is None else _FROZEN.astimezone(tz)


# `metatomic.torch.model` does `import datetime` and then `datetime.datetime.now(...)`,
# so replacing the module it sees is enough to freeze the date it records.
_model.datetime = _types.SimpleNamespace(
    datetime=_FrozenDatetime, timezone=_datetime.timezone
)
"""


def build_model(directory: str, source: str) -> str:
    """
    Run the ``source`` code of a model inside the given environment, and return the path
    of the file it produced.

    The code runs in an empty directory, so that the model it writes can be picked up
    without every project having to agree on a file name.
    """
    workdir = tempfile.mkdtemp(prefix="metatomic-regtests-")
    try:
        script = os.path.join(workdir, "build_model.py")
        with open(script, "w") as fd:
            fd.write(PRELUDE)
            fd.write("\n")
            fd.write(source)

        print("+++++ running the model export")
        environment = dict(os.environ)
        # keep TorchScript from emitting class attributes in a random order
        environment["PYTHONHASHSEED"] = "0"
        # make sure the current virtualenv is available in PATH for packages that spawn
        # subprocesses
        environment["PATH"] = (
            os.path.dirname(venv_python(directory)) + os.pathsep + environment["PATH"]
        )
        subprocess.run(
            [venv_python(directory), "build_model.py"],
            check=True,
            cwd=workdir,
            env=environment,
        )

        produced = [f for f in os.listdir(workdir) if f.endswith(".pt")]
        if len(produced) != 1:
            raise RuntimeError(
                f"expected the source to write exactly one '.pt' file, got {produced}. "
                f"Make the 'source' in models.json write a single model file."
            )

        # move it out of the temporary directory before it is deleted
        path = os.path.join(BUILD_PATH, "building.pt")
        os.makedirs(BUILD_PATH, exist_ok=True)
        shutil.move(os.path.join(workdir, produced[0]), path)
        return path
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def export(
    name: str, models_json: dict, locked: dict, relock: bool, python: str
) -> dict:
    print(f"+++++ {name}:")

    if locked and not relock and locked.get("spec") == models_json:
        # nothing changed and we can reuse the previous build
        built = os.path.join(BUILD_PATH, locked["sha256"] + ".pt")
        if os.path.exists(built) and sha256(built) == locked["sha256"]:
            print("+++++ already built from the current models.json")
            return locked

    if locked and not relock and "spec" in locked:
        # rebuild exactly what the lock recorded
        dependencies = locked["dependencies"]
        spec = locked["spec"]
        print(f"+++++ using the {len(dependencies)} dependencies locked in models.lock")

        if spec != models_json:
            print("+++++ warning: models.json has changed since this model was built,")
            print(
                "+++++          pass --relock to build what models.json now describes"
            )
    else:
        # build the model from the current models.json, and record what dependencies
        # were used
        spec = models_json
        dependencies = spec["dependencies"]
        if locked and not relock:
            print(
                "+++++ models.lock predates the 'spec' field, re-resolving dependencies"
            )
        elif locked:
            print("+++++ re-resolving the dependencies (--relock)")

    directory = create_venv(name, dependencies, python)
    installed = freeze(directory)
    path = build_model(directory, "\n".join(spec["source"]))

    digest = sha256(path)
    final = os.path.join(BUILD_PATH, f"{digest}.pt")
    os.replace(path, final)

    # a URL describes the model that was uploaded to it, so it is only worth keeping
    # while the model is unchanged. Keeping it next to a different checksum would make
    # the lock describe something that does not exist: downloading from that URL then
    # fails the checksum check.
    unchanged = bool(locked) and locked["sha256"] == digest

    entry = {
        "sha256": digest,
        "size": os.path.getsize(final),
        "url": locked.get("url") if unchanged else None,
        "python": subprocess.run(
            [
                venv_python(directory),
                "-c",
                "import platform; print(platform.python_version())",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        # the models.json entry this was built from, so that we can tell whether
        # models.json has moved on since
        "spec": spec,
        # what `pip freeze` reported, unlike the requirements in `spec`
        "dependencies": installed,
    }

    if locked and locked["sha256"] != digest:
        print(
            f"+++++ the model changed: {locked['sha256'][:12]}… -> {digest[:12]}…",
            file=sys.stderr,
        )
        if locked["dependencies"] == installed:
            print(
                "+++++   the dependencies are unchanged, so either the model does\n"
                "+++++   not serialize deterministically, or an upstream package\n"
                "+++++   changed without its version changing",
                file=sys.stderr,
            )
        else:
            added = set(installed) - set(locked["dependencies"])
            removed = set(locked["dependencies"]) - set(installed)
            for dependency in sorted(removed):
                print(f"+++++   - {dependency}")
            for dependency in sorted(added):
                print(f"+++++   + {dependency}")

        if locked.get("url") is not None:
            print(
                "+++++   dropping the 'url' in models.lock, it pointed at the\n"
                "+++++   previous model: re-run with --upload to publish this one",
                file=sys.stderr,
            )

    print(f"+++++ wrote {os.path.relpath(final, ROOT)} ({entry['size'] / 1e6:.1f} MB)")

    return entry


def upload(name: str, path: str, repo: str) -> str:
    """
    Publish a model on HuggingFace, and return the URL to download it from.

    The URL points at the commit this upload creates rather than at a branch, so that it
    keeps referring to this exact model even once a newer one is uploaded over it.
    """
    try:
        from huggingface_hub import upload_file
    except ImportError:
        raise RuntimeError(
            "--upload requires the `huggingface_hub` package, install it with "
            "`pip install huggingface_hub`"
        ) from None

    path_in_repo = f"{name}.pt"
    print(f"+++++ uploading to {repo}/{path_in_repo}")

    commit = upload_file(
        path_or_fileobj=path,
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type="model",
        commit_message=f"{name} ({sha256(path)[:12]})",
    )

    return f"https://huggingface.co/{repo}/resolve/{commit.oid}/{path_in_repo}"


def downloadable(url: str, size: int):
    """Check that ``url`` can be downloaded, without downloading it."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return f"can not be downloaded from {url}: {e}"

    if length is not None and int(length) != size:
        return (
            f"the file at {url} is {length} bytes, but models.lock says it should be "
            f"{size}: the URL points at a different model"
        )

    return None


def check(name: str, models_json: dict, locked: dict) -> list:
    """
    Everything that stops the tests from using ``name``, as ``(fatal, message)`` pairs.

    This only reads ``models.lock`` and asks whether the model can be downloaded; it
    never builds anything.
    """
    if locked is None:
        return [
            (
                True,
                f"has never been built: run `python regtests/export-models.py {name}`",
            )
        ]

    problems = []
    if "spec" not in locked:
        problems.append(
            (
                True,
                "models.lock predates the 'spec' field, so there is no way to tell "
                "what it was built from: "
                f"run `python regtests/export-models.py {name} --relock`",
            )
        )
    elif locked["spec"] != models_json:
        problems.append(
            (
                True,
                "models.json has changed since it was built: run "
                f"`python regtests/export-models.py {name} --relock`",
            )
        )

    if locked.get("url") is None:
        problems.append(
            (
                True,
                "has not been published, so nothing can download it: run "
                f"`python regtests/export-models.py {name} --upload`",
            )
        )
    else:
        error = downloadable(locked["url"], locked["size"])
        if error is not None:
            problems.append((True, error))

    return problems


def prune_lock(locked: dict, models: dict):
    """remove entries from models.lock that are no in models.json"""
    for name in sorted(set(locked) - set(models)):
        print(f"+++++ removing {name} from models.lock since it is not in models.json")
        del locked[name]


def prune_builds(locked: dict):
    """
    Delete what the lock no longer refers to: models left behind by a rebuild that
    changed one, and the virtualenvs of models that are gone.
    """
    keep = {entry["sha256"] + ".pt" for entry in locked.values()}
    for name in sorted(os.listdir(BUILD_PATH)) if os.path.exists(BUILD_PATH) else []:
        if not name.endswith(".pt") or name in keep:
            continue

        os.unlink(os.path.join(BUILD_PATH, name))

    for name in sorted(os.listdir(VENVS_PATH)) if os.path.exists(VENVS_PATH) else []:
        if name not in locked:
            shutil.rmtree(os.path.join(VENVS_PATH, name), ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("models", nargs="*", help="models to build")
    parser.add_argument("--all", action="store_true", help="build every model")
    parser.add_argument("--list", action="store_true", help="list the known models")
    parser.add_argument(
        "--check",
        action="store_true",
        help="check that the models are built from the current models.json and can be "
        "downloaded, without building anything (every model, unless some are named)",
    )
    parser.add_argument(
        "--relock",
        action="store_true",
        help="re-resolve the dependencies instead of using the locked ones",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="python interpreter used to create the virtualenvs",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="publish the models on HuggingFace and record their URL in models.lock",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_HF_REPO,
        help=f"HuggingFace repository to upload to (default: {DEFAULT_HF_REPO})",
    )
    args = parser.parse_args()

    # show the messages from this process and any subprocess in the correct order
    sys.stdout.reconfigure(line_buffering=True)

    with open(MODELS_PATH) as fd:
        models = json.load(fd)

    if not os.path.exists(LOCK_PATH):
        locked = {}
    else:
        with open(LOCK_PATH) as fd:
            locked = json.load(fd)

    if args.list:
        for name in sorted(models):
            print(name)
        return 0

    if args.all or (args.check and not args.models):
        names = sorted(models)
    elif args.models:
        names = args.models
    else:
        parser.error("give at least one model, or --all")

    for name in names:
        if name not in models:
            known = "\n".join(f"    - {key}" for key in sorted(models))
            parser.error(f"unknown model '{name}', models.json contains:\n{known}")

    if args.check:
        failed = False
        for name in names:
            for fatal, message in check(name, models[name], locked.get(name)):
                failed = failed or fatal
                label = "ERROR" if fatal else "warning"
                print(f"{label}: {name}: {message}", file=sys.stderr)

        if failed:
            print(
                "\nsome models are not usable, see above",
                file=sys.stderr,
            )
            return 1

        print(f"+++++ all {len(names)} models are up to date and available")
        return 0

    for name in names:
        entry = locked.get(name)
        locked[name] = export(name, models[name], entry, args.relock, args.python)

        if args.upload:
            built = locked[name]
            already = entry is not None and entry.get("url") is not None
            if already and entry["sha256"] == built["sha256"]:
                print("+++++ the model is unchanged, keeping the URL it already has")
            else:
                built["url"] = upload(
                    name, os.path.join(BUILD_PATH, built["sha256"] + ".pt"), args.repo
                )

        save_lock(locked)

    prune_lock(locked, models)
    save_lock(locked)
    prune_builds(locked)

    print(f"\n+++++ updated {os.path.relpath(LOCK_PATH, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
