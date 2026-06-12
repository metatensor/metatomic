import os
import pathlib

from setuptools import setup


ROOT = pathlib.Path(__file__).parent.resolve()
METATOMIC_CORE = (ROOT / "python" / "metatomic_core").resolve()
METATOMIC_TORCH = (ROOT / "python" / "metatomic_torch").resolve()
METATOMIC_ASE = (ROOT / "python" / "metatomic_ase").resolve()
METATOMIC_TORCHSIM = (ROOT / "python" / "metatomic_torchsim").resolve()


if __name__ == "__main__":
    extras_require = {}
    install_requires = []

    # when packaging a sdist for release, we should never use local dependencies
    METATOMIC_NO_LOCAL_DEPS = os.environ.get("METATOMIC_NO_LOCAL_DEPS", "0") == "1"

    if not METATOMIC_NO_LOCAL_DEPS and METATOMIC_CORE.exists():
        assert METATOMIC_TORCH.exists()
        assert METATOMIC_ASE.exists()
        assert METATOMIC_TORCHSIM.exists()

        # we are building from a git checkout
        install_requires.append(f"metatomic-core @ {METATOMIC_CORE.as_uri()}")
        extras_require["torch"] = f"metatomic-torch @ {METATOMIC_TORCH.as_uri()}"
        extras_require["ase"] = f"metatomic-ase @ {METATOMIC_ASE.as_uri()}"
        extras_require["torchsim"] = (
            f"metatomic-torchsim @ {METATOMIC_TORCHSIM.as_uri()}"
        )
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
