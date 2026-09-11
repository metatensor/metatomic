# This is a custom Python build backend wrapping setuptool's to only depend on
# torch/metatensor-torch when building the wheel and not the sdist
import os
import pathlib

from setuptools import build_meta


ROOT = pathlib.Path(__file__).parent.resolve()

METATOMIC_CORE = (ROOT / ".." / ".." / "metatomic_core").resolve()
METATOMIC_NO_LOCAL_DEPS = os.environ.get("METATOMIC_NO_LOCAL_DEPS", "0") == "1"


if not METATOMIC_NO_LOCAL_DEPS and METATOMIC_CORE.exists():
    # we are building from a git checkout
    METATOMIC_CORE_DEP = f"metatomic-core @ {METATOMIC_CORE.as_uri()}"
else:
    # we are building from a sdist
    METATOMIC_CORE_DEP = "metatomic-core >=0.1.0,<0.2"


FORCED_TORCH_VERSION = os.environ.get("METATOMIC_TORCH_BUILD_WITH_TORCH_VERSION")
if FORCED_TORCH_VERSION is not None:
    TORCH_DEP = f"torch =={FORCED_TORCH_VERSION}"
else:
    TORCH_DEP = "torch >=2.3"

# ==================================================================================== #
#                   Build backend functions definition                                 #
# ==================================================================================== #

# Use the default version of these
prepare_metadata_for_build_wheel = build_meta.prepare_metadata_for_build_wheel
get_requires_for_build_sdist = build_meta.get_requires_for_build_sdist
build_wheel = build_meta.build_wheel
build_sdist = build_meta.build_sdist


# Special dependencies to build the wheels
def get_requires_for_build_wheel(config_settings=None):
    defaults = build_meta.get_requires_for_build_wheel(config_settings)
    return defaults + [TORCH_DEP, METATOMIC_CORE_DEP]


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    raise RuntimeError("metatomic-torch does not support editable installation yet")
