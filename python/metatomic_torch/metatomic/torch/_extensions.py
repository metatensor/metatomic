import glob
import hashlib
import os
import shutil
import site
import sys
import warnings

import metatensor.torch
import torch

from . import _c_lib


METATOMIC_TORCH_LIB_PATH = _c_lib._lib_path()
METATENSOR_TORCH_LIB_PATH = metatensor.torch._c_lib._lib_path()


def _find_delocate_deps(module, lib_name: str, optional=False):
    """
    Find a shared library named `lib_name` that was inserted by delocate inside a wheel
    on macOS.

    :param module: module corresponding to the wheel
    :param lib_name: name of the library to find
    :param optional: should we warn if the library is not found?
    """
    assert sys.platform == "darwin"
    # delocate puts the dependencies in <wheel>/.dylibs/
    search_dir = os.path.join(os.path.dirname(module.__file__), ".dylibs")

    libs_list = glob.glob(os.path.join(search_dir, f"{lib_name}.*"))
    if len(libs_list) == 0 and not optional:
        warnings.warn(
            f"No {lib_name} shared library found in '{search_dir}'. "
            "This may cause issues when loading and running the model.",
            stacklevel=2,
        )
    elif len(libs_list) > 1:
        raise RuntimeError(
            f"Multiple {lib_name} shared libraries found in '{search_dir}': "
            f"{libs_list}. Try to re-install in a fresh environment."
        )
    else:  # len(libs_list) == 1
        return libs_list[0]


def _find_auditwheel_deps(wheel: str, lib_name: str, optional=False):
    """
    Find a shared library named `lib_name` that was inserted by auditwheel inside a
    wheel on Linux.

    :param wheel: name of the wheel/distribution
    :param lib_name: name of the library to find
    :param optional: should we warn if the library is not found?
    """
    assert isinstance(wheel, str)
    assert sys.platform.startswith("linux")
    # auditwheel puts the dependencies in <wheel>.libs/
    search_dir = f"{wheel}.libs/"
    libs_list = []

    for prefix in site.getsitepackages():
        libs_dir = os.path.join(prefix, search_dir)
        if os.path.exists(libs_dir):
            libs_list = glob.glob(os.path.join(libs_dir, lib_name + "-*.so*"))
            if len(libs_list) != 0:
                # found it!
                break

    if len(libs_list) == 0 and not optional:
        warnings.warn(
            f"No {lib_name} shared library found in '{search_dir}'. "
            "This may cause issues when loading and running the model.",
            stacklevel=2,
        )
    elif len(libs_list) > 1:
        raise RuntimeError(
            f"Multiple {lib_name} shared libraries found in '{search_dir}': "
            f"{libs_list}. Try to re-install in a fresh environment."
        )
    else:  # len(libs_list) == 1
        return libs_list[0]


def _featomic_deps_path():
    import featomic

    deps_path = []
    if sys.platform.startswith("linux"):
        libgomp_path = _find_auditwheel_deps("featomic_torch", "libgomp")
        if libgomp_path is not None:
            deps_path.append(libgomp_path)

    deps_path.append(featomic._c_lib._lib_path())

    return deps_path


def _sphericart_deps_path():
    import sphericart.torch

    deps_path = []

    if sys.platform.startswith("linux"):
        libgomp_path = _find_auditwheel_deps("sphericart_torch", "libgomp")
        if libgomp_path is not None:
            deps_path.append(libgomp_path)

        # sphericart uses a separate library to get the CUDA stream corresponding to a
        # tensor, see https://github.com/lab-cosmo/sphericart/pull/164
        sphericart_torch_path = sphericart.torch._lib_path()
        lib_dir = os.path.dirname(sphericart_torch_path)

        cuda_stream_lib = os.path.join(lib_dir, "libsphericart_torch_cuda_stream.so")
        if os.path.exists(cuda_stream_lib):
            deps_path.append(cuda_stream_lib)

    return deps_path


def _deepmd_deps_path():
    import deepmd
    import deepmd.lib

    deps_path = []

    if sys.platform == "darwin":
        libmpi_path = _find_delocate_deps(deepmd, "libmpi")
        if libmpi_path is not None:
            deps_path.append(libmpi_path)

        libpmpi_path = _find_delocate_deps(deepmd, "libpmpi")
        if libpmpi_path is not None:
            deps_path.append(libpmpi_path)

    elif sys.platform.startswith("linux"):
        libgomp_path = _find_auditwheel_deps("deepmd_kit", "libgomp")
        if libgomp_path is not None:
            deps_path.append(libgomp_path)

    libs_dir = os.path.dirname(deepmd.lib.__file__)
    # libdeepmd.so/deepmd.dll/libdeepmd.dylib
    deps_path += list(glob.glob(os.path.join(libs_dir, "*deepmd.*")))

    if sys.platform.startswith("linux"):
        deps_path += list(glob.glob(os.path.join(libs_dir, "libdeepmd_op_cuda.so")))
        deps_path += list(glob.glob(os.path.join(libs_dir, "libdeepmd_dyn_cudart.so")))
        # there is also a dependency on libmpi, but it is not distributed in the wheel

    return deps_path


# Manual definition of which TorchScript extensions have their own dependencies. The
# dependencies should be returned in the order they need to be loaded.
EXTENSIONS_WITH_DEPENDENCIES = {
    "featomic_torch": _featomic_deps_path,
    "sphericart_torch": _sphericart_deps_path,
    "deepmd_op_pt": _deepmd_deps_path,
}


def _collect_extensions(extensions_path):
    """
    Record the list of loaded TorchScript extensions (and their dependencies), to check
    that they are also loaded when executing the model.
    """
    if extensions_path is not None:
        if os.path.exists(extensions_path):
            shutil.rmtree(extensions_path)
        os.makedirs(extensions_path)
        # TODO: the extensions are currently collected in a separate directory,
        # should we store the files directly inside the model file? This would makes
        # the model platform-specific but much more convenient (since the end user
        # does not have to move a model around)

    extensions = []
    extensions_deps = []
    for library in torch.ops.loaded_libraries:
        assert os.path.exists(library)

        # these should be provided by the simulation engine
        if os.path.samefile(library, METATENSOR_TORCH_LIB_PATH):
            continue
        if os.path.samefile(library, METATOMIC_TORCH_LIB_PATH):
            continue

        path = _copy_extension(library, extensions_path)
        info = _extension_info(library)
        info["path"] = path
        extensions.append(info)

        for extra in EXTENSIONS_WITH_DEPENDENCIES.get(info["name"], lambda: [])():
            path = _copy_extension(extra, extensions_path)
            info = _extension_info(extra)
            info["path"] = path
            extensions_deps.append(info)

    return extensions, extensions_deps


def _copy_extension(full_path, extensions_dir):
    full_path = os.path.realpath(full_path)

    prefixes = site.getsitepackages()
    if site.ENABLE_USER_SITE:
        prefixes.append(site.getusersitepackages())

    # this also takes care of extensions installed directly in the same prefix as
    # Python, for example when installing a standalone libmetatensor_torch with conda.
    prefixes.append(sys.prefix)

    path = full_path
    for prefix in prefixes:
        prefix = os.path.realpath(prefix)
        assert os.path.isabs(prefix)

        # Remove any local prefix
        if path.startswith(prefix):
            path = os.path.relpath(path, prefix)
            break

    if extensions_dir is not None:
        collect_path = os.path.realpath(os.path.join(extensions_dir, path))
        if collect_path == path:
            raise RuntimeError(
                f"extensions directory '{extensions_dir}' would overwrite files, "
                "you should set it to a local path instead"
            )

        if os.path.exists(collect_path):
            raise RuntimeError(
                f"more than one extension would be collected at {collect_path}"
            )

        os.makedirs(os.path.dirname(collect_path), exist_ok=True)
        shutil.copyfile(full_path, collect_path)

    return path


def _extension_info(path):
    # get the name of the library, excluding any shared object prefix/suffix
    name = os.path.basename(path)
    if name.startswith("lib"):
        name = name[3:]

    if name.endswith(".so"):
        name = name[:-3]

    if name.endswith(".dll"):
        name = name[:-4]

    if name.endswith(".dylib"):
        name = name[:-6]

    # Collect the hash of the extension shared library. We don't currently use
    # this, but it would allow for binary-level reproducibility later.
    with open(path, "rb") as fd:
        sha256 = hashlib.sha256(fd.read()).hexdigest()

    return {"name": name, "sha256": sha256}
