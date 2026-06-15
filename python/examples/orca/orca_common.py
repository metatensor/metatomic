"""Shared helpers for Metatomic ORCA external-tool scripts."""

from __future__ import annotations

import os
import warnings
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

import ase.units
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from metatomic_ase import MetatomicCalculator


@dataclass(frozen=True)
class ExtInpData:
    """Parsed contents of an ORCA ``*.extinp.tmp`` file."""

    xyz_path: Path
    charge: int
    multiplicity: int
    ncores: int
    do_gradient: bool
    pointcharges_path: Path | None = None


@dataclass(frozen=True)
class MetatomicOrcaSettings:
    model: Path
    extensions_directory: Path | None = None
    device: str | None = None


@dataclass(frozen=True)
class OrcaPreparedJob:
    """Parsed ORCA external-tool job ready for evaluation."""

    input_path: Path
    extinp: ExtInpData
    settings: MetatomicOrcaSettings
    atoms: Atoms


def _first_field(line: str) -> str:
    """Return the first whitespace-separated token, ignoring ``#`` comments."""
    stripped = line.split("#", 1)[0].strip()
    if not stripped:
        return ""
    return stripped.split()[0]


def read_extinp(inputfile: str | Path) -> ExtInpData:
    """Parse an ORCA external-tool input file."""
    input_path = Path(inputfile)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    fields: list[str] = []
    with input_path.open() as handle:
        for line in handle:
            token = _first_field(line)
            if token:
                fields.append(token)

    if len(fields) < 5:
        raise ValueError(f"ORCA extinp file has too few fields: {input_path}")

    xyz_name = fields[0]
    charge = int(fields[1])
    multiplicity = int(fields[2])
    ncores = int(fields[3])
    do_gradient_flag = int(fields[4])

    if do_gradient_flag not in (0, 1):
        raise ValueError("do_gradient from ORCA input must be 0 or 1.")
    if multiplicity < 1:
        raise ValueError("Multiplicity must be a positive integer.")
    if ncores < 1:
        raise ValueError("NCores must be a positive integer.")

    xyz_path = Path(xyz_name)
    if not xyz_path.is_absolute():
        xyz_path = (input_path.parent / xyz_path).resolve()

    pointcharges_path = None
    if len(fields) >= 6:
        pc_name = fields[5]
        pointcharges_path = Path(pc_name)
        if not pointcharges_path.is_absolute():
            pointcharges_path = (input_path.parent / pointcharges_path).resolve()

    return ExtInpData(
        xyz_path=xyz_path,
        charge=charge,
        multiplicity=multiplicity,
        ncores=ncores,
        do_gradient=bool(do_gradient_flag),
        pointcharges_path=pointcharges_path,
    )


def read_xyz(
    xyz_file: str | Path,
) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Read element symbols and Cartesian coordinates (Angstrom) from XYZ."""
    symbols: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    xyz_path = Path(xyz_file)
    with xyz_path.open() as handle:
        natoms = int(handle.readline().strip())
        handle.readline()  # comment line
        for _ in range(natoms):
            line = handle.readline()
            if not line:
                break
            parts = line.split()
            symbols.append(parts[0])
            coordinates.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return symbols, coordinates


def write_engrad(
    filename: str | Path,
    natoms: int,
    energy_hartree: float,
    gradient_hartree_bohr: list[float] | None = None,
) -> None:
    """Write ORCA ``*.engrad`` output (energy in Eh, gradient in Eh/bohr)."""
    output_path = Path(filename)
    lines = [
        "#",
        "# Number of atoms",
        "#",
        f"{natoms}",
        "#",
        "# Total energy [Eh]",
        "#",
        f"{energy_hartree:.12e}",
    ]
    if gradient_hartree_bohr:
        lines.extend(
            [
                "#",
                "# Gradient [Eh/Bohr] A1X, A1Y, A1Z, A2X, ...",
                "#",
            ]
        )
        lines.extend(f"{value: .12e}" for value in gradient_hartree_bohr)

    try:
        output_path.write_text("\n".join(lines) + "\n")
    except OSError as exc:
        raise RuntimeError(f"Failed to write ORCA output file {output_path}: {exc}") from exc


_CALCULATOR_CACHE: dict[tuple[str, str | None, str | None], MetatomicCalculator] = {}


def settings_cache_key(settings: MetatomicOrcaSettings) -> tuple[str, str | None, str | None]:
    extensions = (
        str(settings.extensions_directory.resolve())
        if settings.extensions_directory is not None
        else None
    )
    return (str(settings.model.resolve()), extensions, settings.device)


def get_calculator(settings: MetatomicOrcaSettings) -> MetatomicCalculator:
    """Return a cached Metatomic calculator for the given settings."""
    key = settings_cache_key(settings)
    cached = _CALCULATOR_CACHE.get(key)
    if cached is not None:
        return cached

    extensions = key[1]
    calculator = MetatomicCalculator(
        str(settings.model),
        extensions_directory=extensions,
        device=settings.device,
    )
    _CALCULATOR_CACHE[key] = calculator
    return calculator


def clear_calculator_cache() -> None:
    """Drop cached calculators (useful in tests)."""
    _CALCULATOR_CACHE.clear()


def atoms_from_xyz(xyz_file: str | Path) -> Atoms:
    """Build an ASE ``Atoms`` object from an ORCA XYZ file."""
    symbols, coordinates = read_xyz(xyz_file)
    numbers = [atomic_numbers[symbol.capitalize()] for symbol in symbols]
    return Atoms(numbers=numbers, positions=np.asarray(coordinates, dtype=float))


def atoms_from_extinp(extinp: ExtInpData) -> Atoms:
    """Build an ASE ``Atoms`` object from parsed ORCA external-tool input."""
    atoms = atoms_from_xyz(extinp.xyz_path)
    atoms.info["charge"] = extinp.charge
    atoms.info["spin_multiplicity"] = extinp.multiplicity
    return atoms


def forces_to_orca_gradient(forces_ev_angstrom: np.ndarray) -> np.ndarray:
    """Convert ASE forces (eV/Å) to ORCA gradients (Eh/bohr)."""
    gradient_ev_angstrom = -np.asarray(forces_ev_angstrom, dtype=float)
    return (gradient_ev_angstrom * ase.units.Bohr / ase.units.Hartree).reshape(-1)


def evaluate_structure(
    atoms: Atoms,
    calculator: MetatomicCalculator,
    *,
    do_gradient: bool,
) -> tuple[float, list[float]]:
    """Evaluate energy (Eh) and optional gradient (Eh/bohr) for ``atoms``."""
    atoms.calc = calculator
    energy_hartree = float(atoms.get_potential_energy() / ase.units.Hartree)

    gradient: list[float] = []
    if do_gradient:
        forces = atoms.get_forces()
        gradient = forces_to_orca_gradient(forces).tolist()

    return energy_hartree, gradient


def _warn_unsupported_extinp_fields(extinp: ExtInpData) -> None:
    if extinp.pointcharges_path is not None:
        warnings.warn(
            "Metatomic ORCA external tool does not incorporate ORCA point charges.",
            UserWarning,
            stacklevel=2,
        )


def add_model_arguments(parser: ArgumentParser) -> None:
    """Register model-related flags on ``parser``."""
    parser.add_argument(
        "--model",
        default=os.environ.get("METATOMIC_MODEL"),
        help="Path to an exported metatomic model (.pt). Can also be set with METATOMIC_MODEL.",
    )
    parser.add_argument(
        "--extensions-directory",
        default=os.environ.get("METATOMIC_EXTENSIONS"),
        help=(
            "Directory containing compiled model extensions. "
            "Can also be set with METATOMIC_EXTENSIONS."
        ),
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("METATOMIC_DEVICE"),
        help="Torch device for model evaluation (e.g. cpu, cuda).",
    )


def settings_from_namespace(
    args: Namespace,
    *,
    default_settings: MetatomicOrcaSettings | None = None,
) -> MetatomicOrcaSettings:
    """Build settings from parsed CLI args, optionally inheriting server defaults."""
    model_value = args.model
    if not model_value and default_settings is not None:
        model_value = str(default_settings.model)

    if not model_value:
        raise ValueError(
            "Missing model path. Pass --model PATH, set METATOMIC_MODEL, "
            "or start the server with a default model."
        )

    model = Path(model_value).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"Model file not found: {model}")

    extensions_value = args.extensions_directory
    if not extensions_value and default_settings is not None:
        extensions_value = (
            str(default_settings.extensions_directory)
            if default_settings.extensions_directory is not None
            else None
        )

    extensions_directory = None
    if extensions_value:
        extensions_directory = Path(extensions_value).expanduser().resolve()
        if not extensions_directory.is_dir():
            raise FileNotFoundError(
                f"Extensions directory not found: {extensions_directory}"
            )

    device = args.device
    if device is None and default_settings is not None:
        device = default_settings.device

    return MetatomicOrcaSettings(
        model=model,
        extensions_directory=extensions_directory,
        device=device,
    )


def build_runner_parser(*, prog: str, description: str) -> ArgumentParser:
    """CLI argument parser for standalone ORCA external-tool runs."""
    parser = ArgumentParser(prog=prog, description=description)
    parser.add_argument("inputfile", help="ORCA *.extinp.tmp file")
    add_model_arguments(parser)
    return parser


def parse_runner_arguments(
    arguments: list[str] | None = None,
    *,
    default_settings: MetatomicOrcaSettings | None = None,
) -> tuple[Path, MetatomicOrcaSettings]:
    """Parse ORCA/client argument vectors into an input path and settings."""
    parser = build_runner_parser(
        prog="metatomic-orca-external",
        description="Metatomic ML potential wrapper for ORCA's external-tool interface.",
    )
    args = parser.parse_args(arguments)
    settings = settings_from_namespace(args, default_settings=default_settings)
    return Path(args.inputfile), settings


def prepare_orca_job(
    inputfile: str | Path,
    *,
    settings: MetatomicOrcaSettings,
) -> OrcaPreparedJob:
    """Parse an ORCA external-tool input file into a prepared job."""
    input_path = Path(inputfile).resolve()
    extinp = read_extinp(input_path)
    _warn_unsupported_extinp_fields(extinp)

    if not extinp.xyz_path.is_file():
        raise FileNotFoundError(f"XYZ file not found: {extinp.xyz_path}")

    return OrcaPreparedJob(
        input_path=input_path,
        extinp=extinp,
        settings=settings,
        atoms=atoms_from_extinp(extinp),
    )


def prepare_orca_job_from_arguments(
    arguments: list[str],
    directory: str,
    *,
    default_settings: MetatomicOrcaSettings | None = None,
) -> OrcaPreparedJob:
    """Parse an ORCA/client argument vector into a prepared job."""
    working_dir = Path(directory).resolve()
    if not working_dir.is_dir():
        raise ValueError(f"Invalid directory: {working_dir}")

    inputfile, settings = parse_runner_arguments(
        arguments,
        default_settings=default_settings,
    )
    return prepare_orca_job(working_dir / inputfile, settings=settings)


def run_prepared_jobs(jobs: list[OrcaPreparedJob]) -> list[Path]:
    """Evaluate one or more prepared ORCA jobs."""
    engrad_paths: list[Path] = []
    for job in jobs:
        calculator = get_calculator(job.settings)
        energy, gradient = evaluate_structure(
            job.atoms,
            calculator,
            do_gradient=job.extinp.do_gradient,
        )
        basename = job.extinp.xyz_path.name.removesuffix(".xyz")
        engrad_path = job.input_path.parent / f"{basename}.engrad"
        write_engrad(
            engrad_path,
            natoms=len(job.atoms),
            energy_hartree=energy,
            gradient_hartree_bohr=gradient or None,
        )
        engrad_paths.append(engrad_path)
    return engrad_paths


def run_orca_job(inputfile: str | Path, settings: MetatomicOrcaSettings) -> Path:
    """Parse ``inputfile``, evaluate the structure, and write ``*.engrad``."""
    job = prepare_orca_job(inputfile, settings=settings)
    return run_prepared_jobs([job])[0]
