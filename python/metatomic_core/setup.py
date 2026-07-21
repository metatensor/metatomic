import glob
import os
import pathlib
import subprocess
import sys

import packaging.version
from setuptools import Extension, setup
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist


ROOT = pathlib.Path(__file__).parent.resolve()

METATOMIC_CORE_VERSION = "0.1.0"

METATOMIC_BUILD_TYPE = os.environ.get("METATOMIC_BUILD_TYPE", "release")
if METATOMIC_BUILD_TYPE not in ["debug", "release"]:
    raise Exception(
        f"invalid build type passed: '{METATOMIC_BUILD_TYPE}', "
        "expected 'debug' or 'release'"
    )

# the root path to the _native_ source code of metatomic (Rust source, but built with
# cmake)
METATOMIC_CORE_SRC = os.path.join(ROOT, "..", "..", "metatomic-core")


class universal_wheel(bdist_wheel):
    # When building the wheel, the `wheel` package assumes that if we have a
    # binary extension then we are linking to `libpython.so`; and thus the wheel
    # is only usable with a single python version. This is not the case for
    # here, and the wheel will be compatible with any Python >=3. This is
    # tracked in https://github.com/pypa/wheel/issues/185, but until then we
    # manually override the wheel tag.
    def get_tag(self):
        tag = bdist_wheel.get_tag(self)
        # tag[2:] contains the os/arch tags, we want to keep them
        return ("py3", "none") + tag[2:]


class cmake_ext(build_ext):
    """
    Build the native library using cmake
    """

    def finalize_options(self):
        if self.editable_mode:
            raise RuntimeError(
                "metatensor-core does not support editable installation yet"
            )
        return super().finalize_options()

    def run(self):
        import metatensor

        source_dir = ROOT
        build_dir = os.path.join(ROOT, "build", "cmake-build")
        install_dir = os.path.join(os.path.realpath(self.build_lib), "metatomic")

        os.makedirs(build_dir, exist_ok=True)

        use_external_lib = os.environ.get(
            "METATOMIC_CORE_PYTHON_USE_EXTERNAL_LIB", "OFF"
        )

        cmake_options = [
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DMETATOMIC_CORE_SOURCE_DIR={METATOMIC_CORE_SRC}",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            f"-DCMAKE_BUILD_TYPE={METATOMIC_BUILD_TYPE}",
            f"-DMETATOMIC_CORE_PYTHON_USE_EXTERNAL_LIB={use_external_lib}",
            f"-DCMAKE_PREFIX_PATH={metatensor.utils.cmake_prefix_path}",
        ]

        if "CARGO" in os.environ:
            cmake_options.append(f"-DCARGO_EXE={os.environ['CARGO']}")

        # Handle cross-compilation by detecting cibuildwheels environnement
        # variables
        if sys.platform.startswith("darwin"):
            # ARCHFLAGS is set by cibuildwheels
            ARCHFLAGS = os.environ.get("ARCHFLAGS")
            if ARCHFLAGS is not None:
                archs = filter(
                    lambda u: bool(u),
                    ARCHFLAGS.strip().split("-arch "),
                )
                archs = list(archs)
                assert len(archs) == 1
                arch = archs[0].strip()

                if arch == "x86_64":
                    cmake_options.append("-DRUST_BUILD_TARGET=x86_64-apple-darwin")
                elif arch == "arm64":
                    cmake_options.append("-DRUST_BUILD_TARGET=aarch64-apple-darwin")
                else:
                    raise ValueError(f"unknown arch: {arch}")

        elif sys.platform.startswith("linux"):
            # we set RUST_BUILD_TARGET in our custom docker image
            RUST_BUILD_TARGET = os.environ.get("RUST_BUILD_TARGET")
            if RUST_BUILD_TARGET is not None:
                cmake_options.append(f"-DRUST_BUILD_TARGET={RUST_BUILD_TARGET}")

        elif sys.platform.startswith("win32"):
            # CARGO_BUILD_TARGET is set by cibuildwheels
            CARGO_BUILD_TARGET = os.environ.get("CARGO_BUILD_TARGET")
            if CARGO_BUILD_TARGET is not None:
                cmake_options.append(f"-DRUST_BUILD_TARGET={CARGO_BUILD_TARGET}")

        else:
            raise ValueError(f"unknown platform: {sys.platform}")

        subprocess.run(
            ["cmake", source_dir, *cmake_options],
            cwd=build_dir,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", build_dir, "--parallel", "--target", "install"],
            check=True,
        )


class bdist_egg_disabled(bdist_egg):
    """Disabled version of bdist_egg

    Prevents setup.py install performing setuptools' default easy_install,
    which it should never ever do.
    """

    def run(self):
        sys.exit(
            "Aborting implicit building of eggs.\nUse `pip install .` or "
            "`python -m build --wheel . && pip install dist/metatomic_torch-*.whl` "
            "to install from source."
        )


class sdist_generate_data(sdist):
    """
    Create a sdist with an additional generated files:
        - `git_version_info`
    """

    def run(self):
        n_commits, git_hash = git_version_info()
        with open("git_version_info", "w") as fd:
            fd.write(f"{n_commits}\n{git_hash}\n")

        generate_cxx_tar()

        # run original sdist
        super().run()

        os.unlink("git_version_info")
        for path in glob.glob("metatomic-core-cxx-*.tar.gz"):
            os.unlink(path)


def generate_cxx_tar():
    script = os.path.join(ROOT, "..", "..", "scripts", "package-core.sh")
    assert os.path.exists(script)

    try:
        output = subprocess.run(
            ["bash", "--version"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding="utf8",
        )
    except Exception as e:
        raise RuntimeError("could not run `bash`, is it installed?") from e

    output = subprocess.run(
        ["bash", script, os.getcwd()],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        encoding="utf8",
    )
    if output.returncode != 0:
        stderr = output.stderr
        stdout = output.stdout
        raise RuntimeError(
            "failed to collect C++ sources for Python sdist\n"
            f"stdout:\n {stdout}\n\nstderr:\n {stderr}"
        )


def git_version_info():
    """
    If git is available and we are building from a checkout, get the number of commits
    since the last tag & full hash of the code. Otherwise, this always returns (0, "").
    """
    TAG_PREFIX = "metatomic-v"

    if (ROOT / "git_version_info").exists():
        # we are building from a sdist, without git available, but the git
        # version was recorded in the `git_version_info` file
        with open(ROOT / "git_version_info") as fd:
            n_commits = int(fd.readline().strip())
            git_hash = fd.readline().strip()
    else:
        script = (ROOT / ".." / ".." / "scripts" / "git-version-info.py").resolve()
        assert script.exists()

        output = subprocess.run(
            [sys.executable, script, TAG_PREFIX],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding="utf8",
        )

        if output.returncode != 0:
            raise Exception(
                "failed to get git version info.\n"
                f"stdout: {output.stdout}\n"
                f"stderr: {output.stderr}\n"
            )
        elif output.stderr:
            print(output.stderr, file=sys.stderr)
            n_commits = 0
            git_hash = ""
        else:
            lines = output.stdout.splitlines()
            n_commits = int(lines[0].strip())
            git_hash = lines[1].strip()

    return n_commits, git_hash


def create_version_number(version):
    version = packaging.version.parse(version)

    n_commits, git_hash = git_version_info()

    if n_commits != 0:
        # if we have commits since the last tag, this mean we are in a pre-release of
        # the next version. So we increase either the minor version number or the
        # release candidate number (if we are closing up on a release)
        if version.pre is not None:
            assert version.pre[0] == "rc"
            pre = ("rc", version.pre[1] + 1)
            release = version.release
        else:
            major, minor, _ = version.release
            release = (major, minor + 1, 0)
            pre = None

        version = version.__replace__(
            release=release,
            pre=pre,
            dev=n_commits,
            local=git_hash,
        )

    return str(version)


if __name__ == "__main__":
    if not os.path.exists(METATOMIC_CORE_SRC):
        # we are building from a sdist, which should include metatomic-core Rust
        # sources as a tarball
        tarballs = glob.glob(os.path.join(ROOT, "metatomic-core-*.tar.gz"))

        if not len(tarballs) == 1:
            raise RuntimeError(
                "expected a single 'metatomic-core-*.tar.gz' file containing "
                "metatomic-core Rust sources. remove all files and re-run "
                "scripts/package-core.sh"
            )

        METATOMIC_CORE_SRC = os.path.realpath(tarballs[0])
        subprocess.run(
            ["cmake", "-E", "tar", "xf", METATOMIC_CORE_SRC],
            cwd=ROOT,
            check=True,
        )

        METATOMIC_CORE_SRC = ".".join(METATOMIC_CORE_SRC.split(".")[:-2])

    with open(ROOT / "AUTHORS") as fd:
        authors = fd.read().splitlines()

    if authors[0].startswith(".."):
        # handle "raw" symlink files (on Windows or from full repo tarball)
        with open(ROOT / authors[0]) as fd:
            authors = fd.read().splitlines()

    install_requires = [
        "metatensor-core >=0.2.2,<0.3",
    ]

    setup(
        version=create_version_number(METATOMIC_CORE_VERSION),
        author=", ".join(authors),
        ext_modules=[Extension(name="metatomic", sources=[])],
        install_requires=install_requires,
        cmdclass={
            "build_ext": cmake_ext,
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
            "bdist_wheel": universal_wheel,
            "sdist": sdist_generate_data,
        },
        package_data={
            "metatomic-core": [
                "metatomic/lib/*",
                "metatomic/include/*",
                "metatomic/include/metatomic/*",
            ]
        },
    )
