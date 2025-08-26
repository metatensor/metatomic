# This is a custom Python build backend wrapping setuptool's to only depend on
# torch/metatensor-torch when building the wheel and not the sdist
import os

from setuptools import build_meta


ROOT = os.path.realpath(os.path.dirname(__file__))

FORCED_TORCH_VERSION = os.environ.get("METATOMIC_TORCH_BUILD_WITH_TORCH_VERSION")
if FORCED_TORCH_VERSION is not None:
    TORCH_DEP = f"torch =={FORCED_TORCH_VERSION}"
else:
    TORCH_DEP = "torch >=2.1"

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
    return defaults + [
        TORCH_DEP,
        "metatensor-torch >=0.8.0,<0.9.0",
    ]
