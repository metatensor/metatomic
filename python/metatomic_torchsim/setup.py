import os
import subprocess
import sys

import packaging.version
from setuptools import setup
from setuptools.command.sdist import sdist


ROOT = os.path.realpath(os.path.dirname(__file__))
METATOMIC_TORCH = os.path.realpath(os.path.join(ROOT, "..", "metatomic_torch"))

METATOMIC_TORCHSIM_VERSION = "0.1.0"


class sdist_generate_data(sdist):
    """
    Create a sdist with an additional generated files:
        - `git_version_info`
        - `metatomic-torch-cxx-*.tar.gz`
    """

    def run(self):
        n_commits, git_hash = git_version_info()
        with open("git_version_info", "w") as fd:
            fd.write(f"{n_commits}\n{git_hash}\n")

        # run original sdist
        super().run()

        os.unlink("git_version_info")


def git_version_info():
    """
    If git is available and we are building from a checkout, get the number of commits
    since the last tag & full hash of the code. Otherwise, this always returns (0, "").
    """
    TAG_PREFIX = "metatomic-torchsim-v"

    if os.path.exists("git_version_info"):
        # we are building from a sdist, without git available, but the git
        # version was recorded in the `git_version_info` file
        with open("git_version_info") as fd:
            n_commits = int(fd.readline().strip())
            git_hash = fd.readline().strip()
    else:
        script = os.path.join(ROOT, "..", "..", "scripts", "git-version-info.py")
        assert os.path.exists(script)

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
            major, minor, patch = version.release
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
    with open(os.path.join(ROOT, "AUTHORS")) as fd:
        authors = fd.read().splitlines()

    install_requires = ["torch-sim-atomistic >=0.5,<0.6"]

    # when packaging a sdist for release, we should never use local dependencies
    METATOMIC_NO_LOCAL_DEPS = os.environ.get("METATOMIC_NO_LOCAL_DEPS", "0") == "1"

    if not METATOMIC_NO_LOCAL_DEPS and os.path.exists(METATOMIC_TORCH):
        # we are building from a git checkout or full repo archive
        install_requires.append(f"metatomic-torch @ file://{METATOMIC_TORCH}")
    else:
        # we are building from a sdist/installing from a wheel
        install_requires.append("metatomic-torch >=0.1.11,<0.2.0")

    setup(
        version=create_version_number(METATOMIC_TORCHSIM_VERSION),
        author=", ".join(authors),
        cmdclass={
            "sdist": sdist_generate_data,
        },
    )
