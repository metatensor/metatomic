#!/usr/bin/env python
"""
A small script checking that all the C API functions are documented
"""

import os
import sys

from pycparser import c_ast, parse_file


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
C_API_DOCS = os.path.join(ROOT, "docs", "src", "core", "reference", "c")
FAKE_INCLUDES = [os.path.join(ROOT, "scripts", "include")]
METATOMIC_HEADER = os.path.relpath(
    os.path.join(ROOT, "metatomic-core", "include", "metatomic.h")
)


ERRORS = 0


def error(message):
    global ERRORS
    ERRORS += 1
    print(message)


def documented_functions():
    functions = []

    for root, _, paths in os.walk(C_API_DOCS):
        for path in paths:
            with open(os.path.join(root, path), encoding="utf8") as fd:
                for line in fd:
                    if line.startswith(".. doxygenfunction::"):
                        name = line.split()[2]
                        functions.append(name)

    return functions


def functions_in_outline():
    # function from the "miscellaneous" section of the docs don't require an outline
    # (since they are not related to a specific struct type)
    functions = [
        "mta_version",
        "mta_last_error",
        "mta_set_last_error",
        "mta_string_create",
        "mta_string_free",
        "mta_string_view",
        "mta_format_metadata",
        "mta_unit_conversion_factor",
    ]

    for root, _, paths in os.walk(C_API_DOCS):
        for path in paths:
            with open(os.path.join(root, path), encoding="utf8") as fd:
                for line in fd:
                    if ":c:func:" in line:
                        name = line.split("`")[1]
                        functions.append(name)
    return functions


def all_functions():
    cpp_args = ["-E"]
    for path in FAKE_INCLUDES:
        cpp_args += ["-I", path]
    ast = parse_file(METATOMIC_HEADER, use_cpp=True, cpp_path="gcc", cpp_args=cpp_args)

    functions = []

    class AstVisitor(c_ast.NodeVisitor):
        def visit_Decl(self, node):
            if not isinstance(node.type, c_ast.FuncDecl):
                return

            if not node.name.startswith("mta_"):
                return

            functions.append(node.name)

    visitor = AstVisitor()
    visitor.visit(ast)

    return functions


if __name__ == "__main__":
    docs = documented_functions()
    outline = functions_in_outline()
    for function in all_functions():
        if function not in docs:
            error("Missing documentation for {}".format(function))
        if function not in outline:
            error("Missing outline for {}".format(function))

    if ERRORS != 0:
        sys.exit(1)
