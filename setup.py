import os

from setuptools import setup


ROOT = os.path.realpath(os.path.dirname(__file__))
METATOMIC_CORE = os.path.join(ROOT, "python", "metatomic_core")
METATOMIC_TORCH = os.path.join(ROOT, "python", "metatomic_torch")
METATOMIC_ASE = os.path.join(ROOT, "python", "metatomic_ase")
METATOMIC_TORCHSIM = os.path.join(ROOT, "python", "metatomic_torchsim")


if __name__ == "__main__":
    extras_require = {}
    install_requires = []

    # when packaging a sdist for release, we should never use local dependencies
    METATOMIC_NO_LOCAL_DEPS = os.environ.get("METATOMIC_NO_LOCAL_DEPS", "0") == "1"

    if not METATOMIC_NO_LOCAL_DEPS and os.path.exists(METATOMIC_CORE):
        assert os.path.exists(METATOMIC_TORCH)
        assert os.path.exists(METATOMIC_ASE)
        assert os.path.exists(METATOMIC_TORCHSIM)

        # we are building from a git checkout
        install_requires.append(f"metatomic-core @ file://{METATOMIC_CORE}")
        extras_require["torch"] = f"metatomic-torch @ file://{METATOMIC_TORCH}"
        extras_require["ase"] = f"metatomic-ase @ file://{METATOMIC_ASE}"
        extras_require["torchsim"] = f"metatomic-torchsim @ file://{METATOMIC_TORCHSIM}"
    else:
        # we are building from a sdist/installing from a wheel
        install_requires.append("metatomic-core")

        extras_require["torch"] = "metatomic-torch"
        extras_require["ase"] = "metatomic-ase"
        extras_require["torchsim"] = "metatomic-torchsim"

    setup(
        author=", ".join(open(os.path.join(ROOT, "AUTHORS")).read().splitlines()),
        install_requires=install_requires,
        extras_require=extras_require,
    )
