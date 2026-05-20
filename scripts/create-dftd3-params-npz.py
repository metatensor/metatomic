#!/usr/bin/env python3
"""Create the packaged DFT-D3 reference-table npz.

By default this downloads the reference tables from ``simple-dftd3`` and the
matching covalent radii from ``mctc-lib``, and writes the packaged ``npz`` used
by ``metatomic.torch.dftd3``. This currently provides DFT-D3 C6 references up
to element 103.

The output is the layout used by ``metatomic.torch.dftd3.DFTD3``:

* ``rcov``: ``(Z,)``
* ``r4r2``: ``(Z,)``
* ``c6``: ``(Z, Z, M, M)``
* ``cn_ref``: ``(Z, M)``
"""

from __future__ import annotations

import argparse
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SIMPLE_DFTD3_REFERENCE_URL = (
    "https://raw.githubusercontent.com/dftd3/simple-dftd3/refs/heads/main/"
    "src/dftd3/reference.f90"
)
SIMPLE_DFTD3_R4R2_URL = (
    "https://raw.githubusercontent.com/dftd3/simple-dftd3/refs/heads/main/"
    "src/dftd3/data/r4r2.f90"
)
MCTC_COVRAD_URL = (
    "https://raw.githubusercontent.com/grimme-lab/mctc-lib/refs/heads/main/"
    "src/mctc/data/covrad.f90"
)
ANGSTROM_TO_BOHR = 1.0 / 0.5291772105448199
DEFAULT_OUTPUT = (
    ROOT
    / "python"
    / "metatomic_torch"
    / "metatomic"
    / "torch"
    / "data"
    / "dftd3_parameters.npz"
)
_FLOAT_RE = r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eEdD][-+]?\d+)?(?:_wp)?"


def _extract_from_simple_dftd3_sources(files: dict[str, str]) -> dict[str, np.ndarray]:
    def find_integer_parameter(content: str, var_name: str) -> int:
        match = re.search(
            rf"integer\s*,\s*parameter\s*::\s*{var_name}\s*=\s*(\d+)",
            content,
            re.IGNORECASE,
        )
        if match is None:
            raise ValueError(f"integer parameter '{var_name}' not found")
        return int(match.group(1))

    def find_bracket_array(content: str, var_name: str) -> str:
        match = re.search(
            rf"{var_name}\s*(?:\([^)]*\))?\s*=\s*[^[]*\[",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            raise ValueError(f"array '{var_name}' not found")

        start = match.end() - 1
        depth = 0
        for index in range(start, len(content)):
            char = content[index]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return content[start + 1 : index]

        raise ValueError(f"failed to find end of array '{var_name}'")

    def fortran_numbers(content: str) -> np.ndarray:
        without_comments = []
        for line in content.splitlines():
            if "!" in line:
                line = line[: line.index("!")]
            without_comments.append(line)

        values = []
        for number in re.findall(_FLOAT_RE, "\n".join(without_comments)):
            values.append(
                float(number.replace("_wp", "").replace("D", "e").replace("d", "e"))
            )
        return np.asarray(values, dtype=np.float64)

    def parse_c6(content: str, max_elem: int, max_ref: int) -> np.ndarray:
        n_pairs = max_elem * (max_elem + 1) // 2
        flat = np.zeros(max_ref * max_ref * n_pairs, dtype=np.float64)

        for match in re.finditer(
            r"c6ab_view\(\s*(\d+)\s*:\s*(\d+)\s*\)\s*=\s*\[",
            content,
            re.IGNORECASE,
        ):
            start = int(match.group(1)) - 1
            stop = int(match.group(2))
            depth = 0
            data_start = match.end() - 1
            data_stop = None
            for index in range(data_start, len(content)):
                char = content[index]
                if char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                    if depth == 0:
                        data_stop = index
                        break

            if data_stop is None:
                raise ValueError("failed to find end of c6ab_view assignment")

            values = fortran_numbers(content[data_start + 1 : data_stop])
            if values.shape[0] != stop - start:
                raise ValueError(
                    f"c6ab_view assignment {start + 1}:{stop} contains "
                    f"{values.shape[0]} values"
                )
            flat[start:stop] = values

        triangular = flat.reshape((max_ref, max_ref, n_pairs), order="F")
        c6 = np.zeros((max_elem + 1, max_elem + 1, max_ref, max_ref), dtype=np.float32)

        for z_i in range(1, max_elem + 1):
            for z_j in range(1, max_elem + 1):
                if z_i > z_j:
                    pair_index = z_j + z_i * (z_i - 1) // 2
                    c6[z_i, z_j] = triangular[:, :, pair_index - 1]
                else:
                    pair_index = z_i + z_j * (z_j - 1) // 2
                    c6[z_i, z_j] = triangular[:, :, pair_index - 1].T

        return c6

    reference = files["reference.f90"]
    max_elem = find_integer_parameter(reference, "max_elem")
    max_ref = find_integer_parameter(reference, "max_ref")

    reference_cn_values = fortran_numbers(find_bracket_array(reference, "reference_cn"))
    expected_cn_values = max_elem * max_ref
    if reference_cn_values.shape[0] != expected_cn_values:
        raise ValueError(
            f"reference_cn contains {reference_cn_values.shape[0]} values, "
            f"expected {expected_cn_values}"
        )
    cn_ref = np.full((max_elem + 1, max_ref), -1.0, dtype=np.float32)
    cn_ref[1:] = reference_cn_values.reshape((max_elem, max_ref)).astype(np.float32)

    r4_over_r2 = fortran_numbers(find_bracket_array(files["r4r2.f90"], "r4_over_r2"))
    if r4_over_r2.shape[0] < max_elem:
        raise ValueError(
            f"r4_over_r2 covers only {r4_over_r2.shape[0]} elements, "
            f"but reference.f90 covers {max_elem}"
        )
    atomic_numbers = np.arange(1, max_elem + 1, dtype=np.float64)
    r4r2 = np.zeros(max_elem + 1, dtype=np.float32)
    r4r2[1:] = np.sqrt(0.5 * r4_over_r2[:max_elem] * np.sqrt(atomic_numbers)).astype(
        np.float32
    )

    covalent_radii = fortran_numbers(
        find_bracket_array(files["covrad.f90"], "covalent_rad_2009")
    )
    if covalent_radii.shape[0] < max_elem:
        raise ValueError(
            f"covalent radii cover only {covalent_radii.shape[0]} elements, "
            f"but reference.f90 covers {max_elem}"
        )
    # Pre-multiply by 4/3 and convert from Angstrom to Bohr
    # See https://github.com/tad-mctc/tad-mctc/blob/0d3bb31018520fb8a85bc79c000d4aae01f51235/src/tad_mctc/data/radii.py#L133
    rcov = np.zeros(max_elem + 1, dtype=np.float32)
    rcov[1:] = (
        4.0 / 3.0 * covalent_radii[:max_elem] * ANGSTROM_TO_BOHR
    ).astype(np.float32)

    return {
        "rcov": rcov,
        "r4r2": r4r2,
        "c6": parse_c6(reference, max_elem, max_ref),
        "cn_ref": cn_ref,
    }


def main() -> int:
    def download_sources(
        reference_url: str,
        r4r2_url: str,
        covrad_url: str,
    ) -> dict[str, str]:
        def download_url(url: str) -> bytes:
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read()

        return {
            "reference.f90": download_url(reference_url).decode(
                "utf-8", errors="ignore"
            ),
            "r4r2.f90": download_url(r4r2_url).decode("utf-8", errors="ignore"),
            "covrad.f90": download_url(covrad_url).decode("utf-8", errors="ignore"),
        }

    def read_sources(source_dir: Path) -> dict[str, str]:
        source_dir = source_dir.expanduser().resolve()
        return {
            "reference.f90": (source_dir / "reference.f90").read_text(
                encoding="utf-8"
            ),
            "r4r2.f90": (source_dir / "r4r2.f90").read_text(encoding="utf-8"),
            "covrad.f90": (source_dir / "covrad.f90").read_text(encoding="utf-8"),
        }

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="directory containing reference.f90, r4r2.f90 and covrad.f90 files",
    )
    parser.add_argument(
        "--simple-reference-url",
        default=SIMPLE_DFTD3_REFERENCE_URL,
        help="simple-dftd3 reference.f90 URL",
    )
    parser.add_argument(
        "--simple-r4r2-url",
        default=SIMPLE_DFTD3_R4R2_URL,
        help="simple-dftd3 r4r2.f90 URL",
    )
    parser.add_argument(
        "--covrad-url",
        default=MCTC_COVRAD_URL,
        help="mctc-lib covrad.f90 URL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output npz file to create",
    )
    args = parser.parse_args()

    output_path = args.output.expanduser().resolve()

    if args.source_dir is not None:
        source_dir = args.source_dir.expanduser().resolve()
        params = _extract_from_simple_dftd3_sources(read_sources(source_dir))
        source_description = str(source_dir)
    else:
        try:
            params = _extract_from_simple_dftd3_sources(
                download_sources(
                    args.simple_reference_url,
                    args.simple_r4r2_url,
                    args.covrad_url,
                )
            )
        except (OSError, urllib.error.URLError) as e:
            raise RuntimeError("failed to download simple-dftd3 reference tables") from e
        source_description = (
            f"{args.simple_reference_url}, {args.simple_r4r2_url}, "
            f"and {args.covrad_url}"
        )

    if params["c6"].ndim != 4:
        raise ValueError(f"'c6' must be 4D, got {params['c6'].shape}")
    if params["cn_ref"].shape != (params["c6"].shape[0], params["c6"].shape[2]):
        raise ValueError(
            f"unexpected converted 'cn_ref' shape {params['cn_ref'].shape} "
            f"for c6 shape {params['c6'].shape}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **params,
        source=np.array(source_description),
        layout=np.array("metatomic.torch.dftd3 v1"),
    )

    print(f"read {source_description}")
    print(f"wrote {output_path}")
    for name, value in params.items():
        print(f"  {name}: shape={value.shape} dtype={value.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
