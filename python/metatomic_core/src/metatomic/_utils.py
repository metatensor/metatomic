def _string_from_mta(printed) -> str:
    """Read a ``mta_string_t`` as Python ``str`` and free the C string."""
    from ._c_lib import _get_library

    lib = _get_library()
    try:
        return lib.mta_string_view(printed).decode("utf8")
    finally:
        lib.mta_string_free(printed)
