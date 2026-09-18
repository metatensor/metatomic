"""
The models used by the regression tests, and the code to fetch and cache them locally.

``models.json`` says how to build each model and ``models.lock`` records the exact
environment it was built in; both are handled by ``export-models.py``. This module only
consumes the result: it looks a model up in the lock, and gets its file from the cache,
downloading it if the lock says where it is published.
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(ROOT, "models.json")
LOCK_PATH = os.path.join(ROOT, "models.lock")
CACHE_PATH = os.path.join(ROOT, "cache")

# how many times we retry a download before giving up
N_RETRIES = 3
# read this many bytes at a time when downloading
CHUNK_SIZE = 1024 * 1024


class ModelNotCached(Exception):
    """The model is not in the local cache, and there is nowhere to get it from."""


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fd:
        while True:
            chunk = fd.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def _download(url: str, expected_sha256: str, path: str):
    """
    Download ``url`` to ``path``, checking that its content matches ``expected_sha256``.
    """
    tmp = f"{path}.tmp.{os.getpid()}"

    last_error = None
    for attempt in range(N_RETRIES):
        if attempt != 0:
            # exponential backoff, to absorb transient CI network failures
            time.sleep(2**attempt)

        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                with open(tmp, "wb") as fd:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        digest.update(chunk)
                        fd.write(chunk)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
            if os.path.exists(tmp):
                os.unlink(tmp)
            continue

        actual = digest.hexdigest()
        if actual != expected_sha256:
            # this is not a transient error, don't retry it
            os.unlink(tmp)
            raise RuntimeError(
                f"checksum mismatch for {url}: expected sha256 {expected_sha256}, "
                f"got {actual}. Either the registry is out of date, or the artifact "
                f"was modified in place (it should be pinned to an immutable URL)"
            )

        os.replace(tmp, path)
        return

    raise RuntimeError(f"failed to download {url} after {N_RETRIES} tries") from (
        last_error
    )


def _fetch(url: str, sha256: str, suffix: str) -> str:
    """Download the artifact with the given ``sha256`` into the cache."""
    path = os.path.join(CACHE_PATH, sha256 + suffix)
    os.makedirs(CACHE_PATH, exist_ok=True)

    _download(url, sha256, path)
    return path


def model_path(name: str) -> str:
    """
    Get the local path of the model called ``name``, downloading it if required.
    """
    with open(MODELS_PATH) as fd:
        models = json.load(fd)

    if not os.path.exists(LOCK_PATH):
        locked = {}
    else:
        with open(LOCK_PATH) as fd:
            locked = json.load(fd)

    if name not in locked:
        if name in models:
            raise ModelNotCached(
                f"'{name}' has not been built yet: run `python export-models.py "
                f"{name} --upload` to build and publish it"
            )

        known = "\n".join(f"    - {key}" for key in sorted(models))
        raise KeyError(f"unknown model '{name}', models.json contains:\n{known}")

    entry = locked[name]
    path = os.path.join(CACHE_PATH, entry["sha256"] + ".pt")
    if os.path.exists(path):
        actual = _sha256(path)
        if actual != entry["sha256"]:
            raise RuntimeError(
                f"the cached file {path} does not match its own name: it hashes to "
                f"{actual}. Delete it and run the tests again."
            )

        return path

    if entry.get("url") is None:
        raise ModelNotCached(
            f"'{name}' is not in {CACHE_PATH} and models.lock does not say where it is "
            f"published: run `python export-models.py {name} --upload`"
        )

    return _fetch(entry["url"], entry["sha256"], suffix=".pt")
