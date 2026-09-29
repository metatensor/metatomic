import os

import pytest

from metatomic_torch._extensions import _copy_extension


def _write(path, content: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fd:
        fd.write(content)


def test_collect_extension_outside_any_prefix(tmp_path):
    # A JIT-compiled extension living outside every known prefix (e.g. in
    # torch's own extension build cache) used to make `_copy_extension` raise
    # "extensions directory ... would overwrite files", because joining an
    # absolute path onto extensions_dir silently discards extensions_dir.
    library = tmp_path / "torch_extensions_cache" / "my_ext.so"
    _write(library, b"fake shared library bytes")

    extensions_dir = tmp_path / "collected_extensions"
    path = _copy_extension(str(library), str(extensions_dir))

    assert os.path.isfile(extensions_dir / path)


def test_collect_same_extension_twice_is_a_no_op(tmp_path):
    # Two exports sharing the same extensions_dir (e.g. a parallel sweep on
    # a shared filesystem) both collecting the identical extension should
    # not conflict with each other.
    library = tmp_path / "torch_extensions_cache" / "my_ext.so"
    _write(library, b"fake shared library bytes")

    extensions_dir = tmp_path / "collected_extensions"
    first = _copy_extension(str(library), str(extensions_dir))
    second = _copy_extension(str(library), str(extensions_dir))

    assert first == second


def test_collect_different_extensions_same_basename_do_not_collide(tmp_path):
    library_a = tmp_path / "cache_a" / "my_ext.so"
    library_b = tmp_path / "cache_b" / "my_ext.so"
    _write(library_a, b"content A")
    _write(library_b, b"content B, and it is longer too")

    extensions_dir = tmp_path / "collected_extensions"
    path_a = _copy_extension(str(library_a), str(extensions_dir))
    path_b = _copy_extension(str(library_b), str(extensions_dir))

    assert path_a != path_b


def test_collect_extension_genuine_conflict_still_raises(tmp_path, monkeypatch):
    # When the collected path *is* derived from a known prefix (site-packages
    # here), a real name collision between different content must still be
    # an error.
    site_packages = tmp_path / "site-packages"
    library = site_packages / "pkg" / "lib_a.so"
    _write(library, b"content A")

    monkeypatch.setattr("site.getsitepackages", lambda: [str(site_packages)])

    extensions_dir = tmp_path / "collected_extensions"
    path = _copy_extension(str(library), str(extensions_dir))

    # Tamper with the already-collected file so it disagrees with the source.
    _write(extensions_dir / path, b"tampered, different content")

    with pytest.raises(RuntimeError, match="would be collected"):
        _copy_extension(str(library), str(extensions_dir))
