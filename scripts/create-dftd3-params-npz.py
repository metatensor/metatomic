#!/usr/bin/env python3
"""Create the packaged DFT-D3 reference-table npz.

By default this downloads the official Grimme DFT-D3 reference archive,
verifies its checksum, parses ``dftd3.f`` and ``pars.f``, and writes the
packaged ``npz`` used by ``metatomic.torch.dftd3``.

Alternatively, ``--input`` can read an existing torch checkpoint in the
nvalchemiops/Grimme cache layout:

* ``rcov``: ``(Z,)`` in Bohr
* ``r4r2``: ``(Z,)`` in Bohr
* ``c6ab``: ``(Z, Z, M, M)`` in Hartree * Bohr^6
* ``cn_ref``: ``(Z, Z, M, M)``, with the per-element CN reference grid

The output is the layout used by ``metatomic.torch.dftd3.DFTD3``:

* ``rcov``: ``(Z,)``
* ``r4r2``: ``(Z,)``
* ``c6``: ``(Z, Z, M, M)``
* ``cn_ref``: ``(Z, M)``
"""

from __future__ import annotations

import argparse
import io
import re
import tarfile
import urllib.error
import urllib.request
from hashlib import md5
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DFTD3_TGZ_URL = "https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/dftd3.tgz"
DFTD3_TGZ_MD5 = "a76c752e587422c239c99109547516d2"
DEFAULT_OUTPUT = (
    ROOT
    / "python"
    / "metatomic_torch"
    / "metatomic"
    / "torch"
    / "data"
    / "dftd3_parameters.npz"
)


def _md5_hexdigest(content: bytes) -> str:
    try:
        hasher = md5(usedforsecurity=False)
    except TypeError:
        hasher = md5()
    hasher.update(content)
    return hasher.hexdigest()


def _download_reference_archive(url: str) -> dict[str, str]:
    with urllib.request.urlopen(url, timeout=30) as response:
        content = response.read()

    checksum = _md5_hexdigest(content)
    if checksum != DFTD3_TGZ_MD5:
        raise ValueError(
            "DFT-D3 archive checksum mismatch: expected "
            + DFTD3_TGZ_MD5
            + ", got "
            + checksum
        )

    files = {}
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name not in ("dftd3.f", "pars.f"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            files[name] = handle.read().decode("utf-8", errors="ignore")

    missing = {"dftd3.f", "pars.f"} - set(files)
    if missing:
        raise ValueError("DFT-D3 archive is missing " + ", ".join(sorted(missing)))
    return files


def _read_reference_source(source_dir: Path) -> dict[str, str]:
    files = {}
    for name in ("dftd3.f", "pars.f"):
        path = source_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        files[name] = path.read_text(encoding="utf-8")
    return files


def _find_fortran_array(content: str, var_name: str) -> np.ndarray:
    lines = content.splitlines()
    in_data_block = False
    data_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("!") or stripped.lower().startswith("c "):
            continue

        if not in_data_block:
            if re.match(rf"^\s*data\s+{var_name}\s*/\s*", line, re.IGNORECASE):
                in_data_block = True
                data_lines.append(line)
        else:
            data_lines.append(line)
            if "/" in line and not stripped.startswith("!"):
                break

    if not data_lines:
        raise ValueError(f"variable '{var_name}' not found in Fortran source")

    data_str = " ".join(data_lines)
    match = re.search(
        rf"data\s+{var_name}\s*/\s*(.*?)\s*/",
        data_str,
        re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"failed to parse '{var_name}'")

    numbers = re.findall(r"[-+]?\d+\.\d+(?:_wp)?", match.group(1))
    return np.asarray([float(n.replace("_wp", "")) for n in numbers], dtype=np.float64)


def _parse_pars_array(content: str) -> np.ndarray:
    values = []
    in_data_section = False

    for line in content.splitlines():
        if "real*8" in line.lower() and "pars" in line.lower():
            continue
        if "pars(" in line.lower() and "=(" in line:
            in_data_section = True
        if not in_data_section:
            continue
        if "/)" in line:
            in_data_section = False

        if "!" in line:
            line = line[: line.index("!")]
        line = line.replace("pars(", " ").replace("=(/", " ")
        line = line.replace("/)", " ").replace(":", " ")

        numbers = re.findall(r"[-+]?\d+\.\d+[eEdD][-+]?\d+", line)
        values.extend(
            float(number.replace("D", "e").replace("d", "e")) for number in numbers
        )

    values = np.asarray(values, dtype=np.float64)
    n_records = len(values) // 5
    if len(values) % 5 != 0:
        values = values[: n_records * 5]
    return values.reshape(n_records, 5)


def _decode_element(encoded: int) -> tuple[int, int]:
    atom = encoded
    cn_idx = 1
    while atom > 100:
        atom -= 100
        cn_idx += 1
    return atom, cn_idx


def _build_c6_and_cn_ref(pars_records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c6 = np.zeros((95, 95, 5, 5), dtype=np.float32)
    cn_ref_pair = np.full((95, 95, 5, 5), -1.0, dtype=np.float32)
    cn_values = {element: {} for element in range(95)}

    for c6_value, z_i_encoded, z_j_encoded, cn_i, cn_j in pars_records:
        z_i, cn_i_index = _decode_element(int(z_i_encoded))
        z_j, cn_j_index = _decode_element(int(z_j_encoded))
        if z_i < 1 or z_i > 94 or z_j < 1 or z_j > 94:
            continue
        if cn_i_index < 1 or cn_i_index > 5 or cn_j_index < 1 or cn_j_index > 5:
            continue

        i = cn_i_index - 1
        j = cn_j_index - 1
        c6[z_i, z_j, i, j] = c6_value
        c6[z_j, z_i, j, i] = c6_value

        if i not in cn_values[z_i]:
            cn_values[z_i][i] = cn_i
        if j not in cn_values[z_j]:
            cn_values[z_j][j] = cn_j

    for element in range(1, 95):
        for partner in range(1, 95):
            for cn_index, cn_value in cn_values[element].items():
                cn_ref_pair[element, partner, cn_index, :] = cn_value

    return c6, cn_ref_pair


def _extract_from_reference_sources(files: dict[str, str]) -> dict[str, np.ndarray]:
    r4r2_raw = _find_fortran_array(files["dftd3.f"], "r2r4")
    rcov_raw = _find_fortran_array(files["dftd3.f"], "rcov")
    pars_records = _parse_pars_array(files["pars.f"])

    rcov = np.zeros(95, dtype=np.float32)
    r4r2 = np.zeros(95, dtype=np.float32)
    rcov[1:] = rcov_raw.astype(np.float32)
    r4r2[1:] = r4r2_raw.astype(np.float32)

    c6, cn_ref_pair = _build_c6_and_cn_ref(pars_records)
    return {
        "rcov": rcov,
        "r4r2": r4r2,
        "c6": c6,
        "cn_ref": _element_cn_ref(torch.from_numpy(cn_ref_pair)).numpy(),
    }


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _get(raw: Any, name: str) -> torch.Tensor:
    if isinstance(raw, dict):
        return raw[name]
    return getattr(raw, name)


def _element_cn_ref(cn_ref: torch.Tensor) -> torch.Tensor:
    if cn_ref.ndim == 2:
        return cn_ref
    if cn_ref.ndim != 4:
        raise ValueError("'cn_ref' must be either 2D or 4D, got " + str(cn_ref.shape))
    if cn_ref.shape[1] < 2:
        raise ValueError("4D 'cn_ref' must contain at least one real partner axis")
    return cn_ref[:, 1, :, 0]


def _as_float32_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(dtype=torch.float32).numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input",
        type=Path,
        help=(
            "input torch checkpoint containing rcov, r4r2, c6/c6ab and cn_ref "
            "tables in atomic units"
        ),
    )
    source.add_argument(
        "--source-dir",
        type=Path,
        help="directory containing Grimme reference dftd3.f and pars.f files",
    )
    source.add_argument(
        "--url",
        default=DFTD3_TGZ_URL,
        help="DFT-D3 reference archive URL; used when no local source is provided",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output npz file to create",
    )
    args = parser.parse_args()

    output_path = args.output.expanduser().resolve()

    if args.input is not None:
        input_path = args.input.expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        raw = _torch_load(input_path)
        c6 = (
            _get(raw, "c6")
            if (isinstance(raw, dict) and "c6" in raw)
            else _get(raw, "c6ab")
        )
        params = {
            "rcov": _as_float32_numpy(_get(raw, "rcov")),
            "r4r2": _as_float32_numpy(_get(raw, "r4r2")),
            "c6": _as_float32_numpy(c6),
            "cn_ref": _as_float32_numpy(_element_cn_ref(_get(raw, "cn_ref"))),
        }
        source_description = str(input_path)
    elif args.source_dir is not None:
        source_dir = args.source_dir.expanduser().resolve()
        params = _extract_from_reference_sources(_read_reference_source(source_dir))
        source_description = str(source_dir)
    else:
        try:
            params = _extract_from_reference_sources(_download_reference_archive(args.url))
        except (OSError, urllib.error.URLError) as e:
            raise RuntimeError("failed to download DFT-D3 reference archive") from e
        source_description = args.url

    if params["c6"].ndim != 4:
        raise ValueError("'c6' must be 4D, got " + str(params["c6"].shape))
    if params["cn_ref"].shape != (params["c6"].shape[0], params["c6"].shape[2]):
        raise ValueError(
            "unexpected converted 'cn_ref' shape "
            + str(params["cn_ref"].shape)
            + " for c6 shape "
            + str(params["c6"].shape)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **params,
        source=np.array("Grimme DFT-D3 reference tables, atomic units"),
        layout=np.array("metatomic.torch.dftd3 v1"),
    )

    print(f"read {source_description}")
    print(f"wrote {output_path}")
    for name, value in params.items():
        print(f"  {name}: shape={value.shape} dtype={value.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
