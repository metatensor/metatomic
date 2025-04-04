import os

from setuptools import setup


ROOT = os.path.realpath(os.path.dirname(__file__))
METATOMIC_TORCH = os.path.join(ROOT, "python", "metatomic_torch")


if __name__ == "__main__":
    extras_require = {}

    # when packaging a sdist for release, we should never use local dependencies
    METATOMIC_NO_LOCAL_DEPS = os.environ.get("METATOMIC_NO_LOCAL_DEPS", "0") == "1"

    if not METATOMIC_NO_LOCAL_DEPS and os.path.exists(METATOMIC_TORCH):
        # we are building from a git checkout
        extras_require["torch"] = f"metatomic-torch @ file://{METATOMIC_TORCH}"
    else:
        # we are building from a sdist/installing from a wheel
        extras_require["torch"] = "metatomic-torch"

    setup(
        author=", ".join(open(os.path.join(ROOT, "AUTHORS")).read().splitlines()),
        extras_require=extras_require,
    )
