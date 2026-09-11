#!/usr/bin/env python
"""
Unified declaration generator for metatomic C API bindings.

Usage:
    ./scripts/update-declarations.py            # generate all language bindings
    ./scripts/update-declarations.py python     # generate python only
"""

import os
import sys

from pycparser import c_ast, parse_file


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
FAKE_INCLUDES = [
    os.path.join(ROOT, "python", "scripts", "include"),
    os.path.join(ROOT, "scripts", "include"),
]
METATOMIC_HEADER = os.path.relpath(
    os.path.join(ROOT, "metatomic-core", "include", "metatomic.h")
)


# ============================================================================ #
# Shared AST parsing
# ============================================================================ #


class Function:
    def __init__(self, name, restype):
        self.name = name
        self.restype = restype
        self.args = []

    def add_arg(self, name, type):
        self.args.append((name, type))


class Struct:
    def __init__(self, name):
        self.name = name
        self.members = {}

    def add_member(self, name, type):
        self.members[name] = type


class Enum:
    def __init__(self, name):
        self.name = name
        self.values = {}

    def add_value(self, name, value):
        self.values[name] = value


class AstVisitor(c_ast.NodeVisitor):
    def __init__(self, *, include_dlpack=True):
        self.functions = []
        self.enums = []
        self.structs = []
        self.types = {}
        self.defines = {}

    def visit_Decl(self, node):
        node_name = node.name
        if node_name is None:
            node_name = node.type.name

        if not node_name.startswith("mta_"):
            return

        if isinstance(node.type, c_ast.Enum):
            enum = Enum(node_name)
            for enumerator in node.type.values.enumerators:
                # Strip C unsigned/long suffixes (e.g. 0U, 1UL)
                value = enumerator.value.value.rstrip("UuLl")
                enum.add_value(enumerator.name, value)
            self.enums.append(enum)
        elif isinstance(node.type, c_ast.FuncDecl):
            function = Function(node.name, node.type.type)
            for parameter in node.type.args.params:
                function.add_arg(parameter.name, parameter.type)
            self.functions.append(function)
        else:
            raise RuntimeError(f"Unknown declaration type for {node_name}")

    def visit_Typedef(self, node):
        # Extract metatomic stuff only
        if not node.name.startswith("mta_"):
            return

        if isinstance(node.type.type, c_ast.Enum):
            enum = Enum(node.name)
            for enumerator in node.type.type.values.enumerators:
                # Strip C unsigned/long suffixes (e.g. 0U, 1UL)
                value = enumerator.value.value.rstrip("UuLl")
                enum.add_value(enumerator.name, value)
            self.enums.append(enum)

        elif isinstance(node.type.type, c_ast.Struct):
            if node.name.startswith("DLPackExchangeAPI"):
                return

            struct = Struct(node.name)
            for _, member in node.type.type.children():
                struct.add_member(member.name, member.type)
            self.structs.append(struct)

        else:
            # keep `node.type` (not `node.type.type`) so that pointer typedefs
            # such as `typedef mta_opaque_string_t* mta_string_t` are translated
            # to a ctypes pointer instead of the pointed-to type
            self.types[node.name] = node.type


def _typedecl_name(type):
    assert isinstance(type, c_ast.TypeDecl)
    if isinstance(type.type, c_ast.Struct):
        return type.type.name
    elif isinstance(type.type, c_ast.Enum):
        return type.type.name
    else:
        assert len(type.type.names) == 1
        return type.type.names[0]


def parse_header(file):
    cpp_args = ["-E"]
    for path in FAKE_INCLUDES:
        cpp_args += ["-I", path]
    ast = parse_file(file, use_cpp=True, cpp_path="gcc", cpp_args=cpp_args)

    visitor = AstVisitor()
    visitor.visit(ast)

    # `#define` without a value associated
    no_value_define = ["METATOMIC_H", "MTA_EXTERN_C"]

    with open(file) as fd:
        for line in fd:
            if "#define" in line:
                split = line.split()

                name = split[1]
                if name in no_value_define:
                    continue
                value = split[2]
                visitor.defines[name] = value
    return visitor


# ==================================================================================== #
#                                 Python backend                                       #
# ==================================================================================== #


def _py_type_name(name):
    if name.startswith("mta_") or name.startswith("mts_") or name.startswith("DL"):
        return name
    elif name == "uintptr_t":
        return "c_uintptr_t"
    elif name == "void":
        return "None"
    elif name == "int8_t":
        return "ctypes.c_int8"
    elif name == "uint8_t":
        return "ctypes.c_uint8"
    elif name == "int16_t":
        return "ctypes.c_int16"
    elif name == "uint16_t":
        return "ctypes.c_uint16"
    elif name == "int32_t":
        return "ctypes.c_int32"
    elif name == "uint32_t":
        return "ctypes.c_uint32"
    elif name == "int64_t":
        return "ctypes.c_int64"
    elif name == "uint64_t":
        return "ctypes.c_uint64"
    else:
        return "ctypes.c_" + name


def _py_funcdecl(type):
    restype = _py_type(type.type)
    args = [_py_type(t.type) for t in type.args.params]
    return f"CFUNCTYPE({restype}, {', '.join(args)})"


def _py_type(type):
    if isinstance(type, c_ast.PtrDecl):
        if isinstance(type.type, c_ast.PtrDecl):
            if isinstance(type.type.type, c_ast.TypeDecl):
                name = _typedecl_name(type.type.type)
                if name == "char":
                    return "POINTER(ctypes.c_char_p)"
                elif name == "uint8_t":
                    return "POINTER(ctypes.c_char_p)"
                name = _py_type_name(name)
                return f"POINTER(POINTER({name}))"
            elif isinstance(type.type.type, c_ast.PtrDecl):
                assert isinstance(type.type.type.type, c_ast.TypeDecl)
                assert _typedecl_name(type.type.type.type) == "char"
                return "POINTER(POINTER(ctypes.c_char_p))"
        elif isinstance(type.type, c_ast.TypeDecl):
            name = _typedecl_name(type.type)
            if name == "void":
                return "ctypes.c_void_p"
            elif name == "char":
                return "ctypes.c_char_p"
            elif name == "uint8_t":
                return "ctypes.c_char_p"
            else:
                return f"POINTER({_py_type_name(name)})"
        elif isinstance(type.type, c_ast.FuncDecl):
            return _py_funcdecl(type.type)
    else:
        if isinstance(type, c_ast.TypeDecl):
            return _py_type_name(_typedecl_name(type))
        elif isinstance(type, c_ast.IdentifierType):
            return _py_type_name(type.names[0])
        elif isinstance(type, c_ast.ArrayDecl):
            if isinstance(type.dim, c_ast.Constant):
                size = type.dim.value
            else:
                raise Exception("dynamically sized arrays are not supported")
            return f"{_py_type(type.type)} * {size}"
        elif isinstance(type, c_ast.FuncDecl):
            return _py_funcdecl(type)
    raise Exception("Unknown type")


def generate_python(data):
    outpath = os.path.join(
        ROOT, "python", "metatomic_core", "src", "metatomic", "_c_api.py"
    )
    with open(outpath, "w") as f:
        f.write(
            """# fmt: off
# flake8: noqa
\"\"\"
This file declares the C-API corresponding to metatomic.h, in a way compatible
with the ctypes Python module.

This file is automatically generated by `scripts/update-declarations.py`,
do not edit it manually!
\"\"\"

import ctypes
import platform
from ctypes import CFUNCTYPE, POINTER

from ctypes_dlpack import DLDataType, DLDevice, DLManagedTensorVersioned, DLPackVersion
from metatensor._c_api import (
    mts_labels_t,
    mts_block_t,
    mts_tensormap_t,
    mts_realloc_buffer_t,
    mts_create_array_callback_t,
)


class _EnumType(type(ctypes.c_int32)):
    def __new__(metacls, name, bases, namespace):
        if "_members_" not in namespace:
            members = {}
            for key, value in namespace.items():
                if not key.startswith("_"):
                    members[key] = value
            namespace["_members_"] = members
        else:
            members = namespace["_members_"]

        namespace["_reverse_map_"] = {v: k for k, v in members.items()}
        return type(ctypes.c_int32).__new__(metacls, name, bases, namespace)

    def __repr__(self):
        return f"<Enum {self.__name__}>"


class _Enum(ctypes.c_int32, metaclass=_EnumType):
    _members_ = {}

    def __repr__(self):
        value_name = self._reverse_map_.get(self.value, str(self.value))
        return f"{self.__class__.__name__}.{value_name}"

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        if type(self) is type(other):
            return self.value == other.value
        return NotImplemented

    def __hash__(self):
        return hash(self.value)


arch = platform.architecture()[0]
if arch == "32bit":
    c_uintptr_t = ctypes.c_uint32
elif arch == "64bit":
    c_uintptr_t = ctypes.c_uint64

"""
        )

        # Enums
        for enum in data.enums:
            f.write(f"\n\nclass {enum.name}(_Enum):\n")
            for name, value in enum.values.items():
                f.write(f"    {name} = {value}\n")

        # structs declartions, without fields
        for struct in data.structs:
            f.write(f"\n\nclass {struct.name}(ctypes.Structure):\n")
            f.write("    pass\n")

        # typedefs
        f.write("\n\n")
        for name, c_type in data.types.items():
            if name == "mta_status_t":
                # this is already defined as an enum
                continue
            f.write(f"{name} = {_py_type(c_type)}\n")

        # structs fields definitions
        f.write("\n")
        for struct in data.structs:
            if len(struct.members) == 0:
                continue
            f.write(f"\n{struct.name}._fields_ = [\n")
            for name, type in struct.members.items():
                f.write(f'    ("{name}", {_py_type(type)}),\n')
            f.write("]\n")

        # Functions
        f.write("\n\ndef setup_functions(lib):\n")
        f.write("    from ._status import check_status\n")
        for function in data.functions:
            f.write(f"\n    lib.{function.name}.argtypes = [")
            args = [_py_type(arg[1]) for arg in function.args]
            if args == ["None"]:
                args = []
            for arg in args:
                f.write(f"\n        {arg},")
            f.write("\n    ]\n")
            restype = _py_type(function.restype)
            if restype == "mta_status_t" and function.name != "mta_last_error":
                restype = "check_status"
            f.write(f"    lib.{function.name}.restype = {restype}\n")


# ==================================================================================== #
#                                       main                                           #
# ==================================================================================== #


def main():
    data = parse_header(METATOMIC_HEADER)

    targets = sys.argv[1:] if len(sys.argv) > 1 else ["python"]

    for target in targets:
        if target == "python":
            generate_python(data)
        else:
            print(f"Unknown target: {target}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
