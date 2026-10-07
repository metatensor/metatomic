# This script is embedded inside the python plugin, and executed with the user's Python
# interpreter to find the information required to start the same Python inside the
# current process: path to the shared libpython, PYTHONHOME, and code to set up
# `sys.path` (including virtual environments and `.pth` files).
#
# The output is made of multiple lines:
#   1. "metatomic-find-python-success" or "metatomic-find-python-error: <message>"
#   2. path to the Python executable
#   3. path to the shared libpython
#   4. value for PYTHONHOME 5+. Python code to execute in the embedded interpreter to
#   setup sys.path

import ctypes
import ctypes.util
import os
import site
import sys
import sysconfig


IS_WINDOWS = os.name == "nt"
IS_APPLE = sys.platform == "darwin"


def _linked_libpython():
    """Find the libpython used by the current interpreter, if any"""
    if IS_WINDOWS:
        # `ctypes.pythonapi` is a handle to `pythonXY.dll`
        buffer = ctypes.create_unicode_buffer(32768)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        length = kernel32.GetModuleFileNameW(
            ctypes.c_void_p(ctypes.pythonapi._handle), buffer, len(buffer)
        )
        if length == 0:
            return None
        return buffer.value

    class Dl_info(ctypes.Structure):
        _fields_ = [
            ("dli_fname", ctypes.c_char_p),
            ("dli_fbase", ctypes.c_void_p),
            ("dli_sname", ctypes.c_char_p),
            ("dli_saddr", ctypes.c_void_p),
        ]

    try:
        libdl = ctypes.CDLL(None)
        dladdr = libdl.dladdr
    except (OSError, AttributeError):
        libdl = ctypes.CDLL(ctypes.util.find_library("dl"))
        dladdr = libdl.dladdr

    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(Dl_info)]
    dladdr.restype = ctypes.c_int

    info = Dl_info()
    address = ctypes.cast(ctypes.pythonapi.Py_GetVersion, ctypes.c_void_p)
    if dladdr(address, ctypes.byref(info)) == 0 or info.dli_fname is None:
        return None

    path = os.path.realpath(os.fsdecode(info.dli_fname))
    if path == os.path.realpath(sys.executable):
        # Python is statically linked in the executable
        return None

    return path


def _candidate_paths():
    """Iterate over possible paths for libpython"""
    yield _linked_libpython()

    if IS_WINDOWS:
        suffix = ".dll"
    elif IS_APPLE:
        suffix = ".dylib"
    else:
        suffix = sysconfig.get_config_var("SHLIB_SUFFIX") or ".so"

    names = []
    for var in ["INSTSONAME", "LDLIBRARY"]:
        value = sysconfig.get_config_var(var)
        if value and not value.endswith(".a"):
            names.append(value)

    version = sysconfig.get_config_var("VERSION") or "{}.{}".format(
        *sys.version_info[:2]
    )
    abiflags = sysconfig.get_config_var("ABIFLAGS") or ""
    prefix = "" if IS_WINDOWS else "lib"
    for stem in [f"python{version}{abiflags}", f"python{version}"]:
        names.append(prefix + stem + suffix)

    directories = []
    for var in ["LIBPL", "LIBDIR"]:
        value = sysconfig.get_config_var(var)
        if value:
            directories.append(value)
            multiarch = sysconfig.get_config_var("MULTIARCH")
            if multiarch:
                directories.append(os.path.join(value, multiarch))

    framework_prefix = sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX")
    if framework_prefix:
        framework = sysconfig.get_config_var("PYTHONFRAMEWORK") or "Python"
        yield os.path.join(
            framework_prefix, f"{framework}.framework", "Versions", version, framework
        )

    directories.append(sys.base_exec_prefix)
    directories.append(os.path.join(sys.base_exec_prefix, "lib"))
    directories.append(os.path.dirname(sys.executable))

    for directory in directories:
        for name in names:
            yield os.path.join(directory, name)


def find_libpython():
    for path in _candidate_paths():
        if path and os.path.isabs(path) and os.path.isfile(path):
            return os.path.realpath(path)
    return None


def python_home():
    if IS_WINDOWS or sys.base_prefix == sys.base_exec_prefix:
        return sys.base_prefix
    return sys.base_prefix + os.pathsep + sys.base_exec_prefix


def setup_code():
    # sys.path[0] is the current directory when running with `python -c`
    path = sys.path[1:]

    site_dirs = list(site.getsitepackages())
    if site.ENABLE_USER_SITE:
        site_dirs.append(site.getusersitepackages())
    site_dirs = [d for d in site_dirs if os.path.isdir(d)]

    return "\n".join(
        [
            "import sys, site",
            f"sys.path[:] = {path!r}",
            f"sys.prefix = {sys.prefix!r}",
            f"sys.exec_prefix = {sys.exec_prefix!r}",
            f"sys.executable = {sys.executable!r}",
            # re-process .pth files, executing `import` lines (used for example
            # by editable installs). Paths already in sys.path are not added again
            f"for d in {site_dirs!r}:",
            "    site.addsitedir(d)",
        ]
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if sys.version_info < (3, 10):
        version = "{}.{}".format(*sys.version_info[:2])
        print(
            "metatomic-find-python-error: Python >= 3.10 is required, "
            f"{sys.executable} is v{version}"
        )
        return

    libpython = find_libpython()
    if libpython is None:
        print(
            "metatomic-find-python-error: could not find a shared libpython "
            f"for {sys.executable}, "
            "this Python might have been built without --enable-shared"
        )
        return

    print("metatomic-find-python-success")
    print(sys.executable)
    print(libpython)
    print(python_home())
    print(setup_code())


if __name__ == "__main__":
    main()
