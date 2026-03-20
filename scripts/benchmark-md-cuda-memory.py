#!/usr/bin/env python3
"""Benchmark CUDA memory usage during i-PI/metatomic MD.

This is based on ``MD/heat_flux.py`` but removes the hardcoded path map and adds
CUDA-memory monitoring around the MD phases. By default it uses the local
``MD/i-pi.xml`` and ``MD/water-350K.xyz`` files.

For ``ffdirect`` + ``pes=metatomic`` the force evaluations happen in-process, so
``torch.cuda`` statistics should reflect the relevant allocations. If you move force
evaluation out of process, prefer ``--sample-nvidia-smi`` as well.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MD_DIR = REPO_ROOT / "MD"
CHECKOUT_PACKAGE_ROOT = REPO_ROOT / "python" / "metatomic_torch"


def _clear_metatomic_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "metatomic" or module_name.startswith("metatomic."):
            del sys.modules[module_name]


def _bootstrap_metatomic():
    import_mode = os.environ.get("METATOMIC_BENCHMARK_IMPORT", "auto").lower()
    if import_mode not in {"auto", "checkout", "installed"}:
        raise ValueError(
            "METATOMIC_BENCHMARK_IMPORT must be one of: auto, checkout, installed"
        )

    candidates = ["checkout", "installed"] if import_mode == "auto" else [import_mode]
    errors: list[tuple[str, Exception]] = []

    for candidate in candidates:
        inserted_checkout_path = False
        try:
            if candidate == "checkout":
                sys.path.insert(0, str(CHECKOUT_PACKAGE_ROOT))
                inserted_checkout_path = True

            import metatomic.torch as mta  # noqa: E402

            return mta, candidate
        except Exception as exc:
            errors.append((candidate, exc))
            _clear_metatomic_modules()
            if inserted_checkout_path:
                sys.path.pop(0)

    details = "\n".join(
        f"  - {candidate}: {type(exc).__name__}: {exc}"
        for candidate, exc in errors
    )
    raise RuntimeError(
        "Could not import metatomic.torch for benchmarking.\n"
        "Tried the following import modes:\n"
        f"{details}\n"
        "If you want to benchmark the checkout version, build/install "
        "`python/metatomic_torch` for the active PyTorch version first."
    )


MTA, IMPORT_SOURCE = _bootstrap_metatomic()


DEFAULT_XML_TEMPLATE = MD_DIR / "i-pi.xml"
DEFAULT_STRUCTURE = MD_DIR / "water-350K.xyz"


@dataclass(frozen=True)
class MemorySample:
    phase: str
    label: str
    elapsed_s: float
    allocated_mb: float | None
    reserved_mb: float | None
    max_allocated_mb: float | None
    max_reserved_mb: float | None
    free_mb: float | None
    total_mb: float | None
    nvidia_smi_used_mb: float | None
    nvidia_smi_total_mb: float | None


@dataclass(frozen=True)
class PhaseSummary:
    phase: str
    steps: int
    elapsed_s: float
    peak_torch_allocated_mb: float | None
    peak_torch_reserved_mb: float | None
    peak_sampled_allocated_mb: float | None
    peak_sampled_reserved_mb: float | None
    peak_nvidia_smi_used_mb: float | None


def bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return value / (1024.0 * 1024.0)


def format_optional_mb(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path.cwd() / f"md-cuda-memory-{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Metatomic model used by the i-PI force field.",
    )
    parser.add_argument(
        "--property-model",
        type=Path,
        help=(
            "Optional model used for the custom heat_flux property. Defaults to "
            "--model."
        ),
    )
    parser.add_argument(
        "--extensions-directory",
        type=Path,
        help="Directory containing TorchScript extensions for the model(s).",
    )
    parser.add_argument(
        "--structure",
        type=Path,
        default=DEFAULT_STRUCTURE,
        help="Reference structure/topology used for the MD initialization.",
    )
    parser.add_argument(
        "--xml-template",
        type=Path,
        default=DEFAULT_XML_TEMPLATE,
        help="i-PI XML template with the placeholders from MD/i-pi.xml.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch/CUDA device used by both the force field and the property model.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.5,
        help="MD timestep in femtoseconds.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=350.0,
        help="Simulation temperature in kelvin.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Stride used for the i-PI property output.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to i-PI.",
    )
    parser.add_argument(
        "--pre-equil-steps",
        type=int,
        default=0,
        help="Number of pre-equilibration steps. Set to 0 to skip this phase.",
    )
    parser.add_argument(
        "--production-steps",
        type=int,
        default=1000,
        help="Number of production steps.",
    )
    parser.add_argument(
        "--pre-equil-tau",
        type=float,
        default=2.0,
        help="SVR thermostat tau for the pre-equilibration phase in femtoseconds.",
    )
    parser.add_argument(
        "--production-tau",
        type=float,
        default=50.0,
        help="SVR thermostat tau for the production phase in femtoseconds.",
    )
    parser.add_argument(
        "--skip-heat-flux-property",
        action="store_true",
        help="Do not register the custom heat_flux property calculator.",
    )
    parser.add_argument(
        "--property-output-key",
        default="heat_flux",
        help="Output key requested from the property model.",
    )
    parser.add_argument(
        "--monitor-interval-s",
        type=float,
        default=0.2,
        help="Polling interval for the background CUDA-memory monitor in seconds.",
    )
    parser.add_argument(
        "--sample-nvidia-smi",
        action="store_true",
        help="Also poll nvidia-smi for whole-GPU memory usage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory where i-PI outputs and memory traces are written.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="CSV file for the sampled CUDA-memory timeline. Defaults inside output-dir.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="JSON file for the phase summaries and configuration. Defaults inside output-dir.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.exists():
        raise FileNotFoundError(f"model does not exist: {args.model}")
    if args.property_model is not None and not args.property_model.exists():
        raise FileNotFoundError(f"property model does not exist: {args.property_model}")
    if not args.structure.exists():
        raise FileNotFoundError(f"structure does not exist: {args.structure}")
    if not args.xml_template.exists():
        raise FileNotFoundError(f"XML template does not exist: {args.xml_template}")
    if args.extensions_directory is not None and not args.extensions_directory.exists():
        raise FileNotFoundError(
            f"extensions directory does not exist: {args.extensions_directory}"
        )
    if args.pre_equil_steps < 0:
        raise ValueError("--pre-equil-steps must be non-negative")
    if args.production_steps <= 0:
        raise ValueError("--production-steps must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.dt <= 0:
        raise ValueError("--dt must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.monitor_interval_s < 0:
        raise ValueError("--monitor-interval-s must be non-negative")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(
            "This benchmark is intended for CUDA memory tracking; pass a CUDA device "
            f"instead of '{args.device}'."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")


def resolved_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    csv_path = args.csv if args.csv is not None else args.output_dir / "cuda-memory.csv"
    json_path = (
        args.json if args.json is not None else args.output_dir / "cuda-memory.json"
    )
    return csv_path, json_path


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def format_model_parameter(model_path: Path, extensions_directory: Path | None) -> str:
    value = str(model_path.resolve())
    if extensions_directory is not None:
        value += f",extensions:{extensions_directory.resolve()}"
    return value


def render_phase_xml(
    template_text: str,
    *,
    structure_path: Path,
    model_parameter: str,
    device: str,
    timestep_fs: float,
    temperature_k: float,
    stride: int,
    seed: int,
    init_structure: str,
    init_mode: str,
    tau_fs: float,
    prefix: str,
    init_velocities_xml: str,
    include_heat_flux_property: bool,
    property_output_key: str,
) -> str:
    xml = template_text
    replacements = {
        "__TEMPLATE__": str(structure_path.resolve()),
        "__MODEL__": model_parameter,
        "__TIMESTEP__": str(timestep_fs),
        "__TEMPERATURE__": str(temperature_k),
        "__STRIDE__": str(stride),
        "__SEED__": str(seed),
        "__INIT_STRUCTURE__": init_structure,
        "__TAU__": str(tau_fs),
        "__MODE__": init_mode,
        "__PREFIX__": prefix,
        "__INIT_VELOCITIES__": init_velocities_xml,
    }
    for source, target in replacements.items():
        xml = xml.replace(source, target)

    if "device:cuda" in xml:
        xml = xml.replace("device:cuda", f"device:{device}")
    elif "__DEVICE__" in xml:
        xml = xml.replace("__DEVICE__", device)

    if include_heat_flux_property:
        if property_output_key != "heat_flux":
            xml = xml.replace("heat_flux", property_output_key)
    else:
        xml = xml.replace(", heat_flux", "")
        xml = xml.replace("heat_flux, ", "")
        xml = xml.replace(" heat_flux ]", " ]")

    return xml


class HeatFluxPropertyCalculator:
    def __init__(
        self,
        *,
        model_path: Path,
        extensions_directory: Path | None,
        ref_topology: Path,
        device: str,
        output_key: str,
    ):
        try:
            import sphericart.torch  # noqa: F401
        except ImportError:
            pass

        from vesin.metatomic import compute_requested_neighbors

        self._compute_requested_neighbors = compute_requested_neighbors
        self.output_key = output_key
        self.device = device

        atoms = ase.io.read(ref_topology)
        atom_types = torch.as_tensor(atoms.get_atomic_numbers())
        cell = torch.as_tensor(atoms.get_cell())
        pbc = torch.as_tensor(atoms.get_pbc())
        self.masses = torch.as_tensor(atoms.get_masses())
        self.dummy_system = lambda positions: MTA.System(
            types=atom_types,
            positions=positions,
            cell=cell,
            pbc=pbc,
        )

        self.model = MTA.load_atomistic_model(
            str(model_path.resolve()),
            str(extensions_directory.resolve()) if extensions_directory else None,
        ).to(device=device)

        outputs = self.model.capabilities().outputs
        if output_key not in outputs:
            available = ", ".join(sorted(outputs.keys()))
            raise KeyError(
                f"property model does not expose '{output_key}'. "
                f"Available outputs: {available}"
            )

        self.options = MTA.ModelEvaluationOptions(
            length_unit="A",
            outputs={output_key: outputs[output_key]},
        )

    def _create_tensormap(self, values: torch.Tensor, quantity: str) -> TensorMap:
        values = values[..., None]
        components = []
        if values.shape[1] != 1:
            components.append(
                Labels(
                    ["xyz"],
                    torch.arange(values.shape[1], dtype=torch.int64).reshape(-1, 1),
                )
            )

        block = TensorBlock(
            values=values,
            samples=Labels(
                ["system", "atom"],
                torch.vstack(
                    [
                        torch.zeros(values.shape[0], dtype=torch.int64),
                        torch.arange(values.shape[0], dtype=torch.int64),
                    ]
                ).T,
            ),
            components=components,
            properties=Labels([quantity], torch.tensor([[0]], dtype=torch.int64)),
        )
        return TensorMap(Labels(["_"], torch.tensor([[0]], dtype=torch.int64)), [block])

    def __call__(self, properties) -> np.ndarray:
        positions = torch.as_tensor(properties.beads.q * 0.52917721).reshape(-1, 3)
        momenta = torch.as_tensor(properties.beads.p).reshape(-1, 3) * 0.12217864
        velocities = momenta / self.masses[:, None] * 0.098226948

        system = self.dummy_system(positions=positions)
        system.add_data("velocities", self._create_tensormap(velocities, "velocity"))
        system.add_data("masses", self._create_tensormap(self.masses, "mass"))
        system = system.to(self.model._model_dtype, self.device)

        self._compute_requested_neighbors(system, "A", model=self.model)
        result = self.model([system], self.options, check_consistency=False)
        return (
            result[self.output_key]
            .block()
            .values.detach()
            .clone()
            .cpu()
            .numpy()
            .reshape(-1)
        )


class CudaMemoryMonitor:
    def __init__(
        self,
        *,
        device: str,
        interval_s: float,
        sample_nvidia_smi: bool,
    ):
        self.device = torch.device(device)
        self.device_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        self.interval_s = interval_s
        self.sample_nvidia_smi = (
            sample_nvidia_smi and shutil.which("nvidia-smi") is not None
        )
        self.samples: list[MemorySample] = []
        self._phase = "idle"
        self._start_time: float | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _query_nvidia_smi(self) -> tuple[float | None, float | None]:
        if not self.sample_nvidia_smi:
            return None, None

        command = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            str(self.device_index),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
        except Exception:
            return None, None

        if completed.returncode != 0:
            return None, None

        try:
            used_mb, total_mb = [
                float(part.strip()) for part in completed.stdout.strip().split(",")
            ]
        except Exception:
            return None, None

        return used_mb, total_mb

    def _sample(self, label: str) -> MemorySample:
        elapsed_s = (
            0.0 if self._start_time is None else time.perf_counter() - self._start_time
        )
        allocated = bytes_to_mb(torch.cuda.memory_allocated(self.device))
        reserved = bytes_to_mb(torch.cuda.memory_reserved(self.device))
        max_allocated = bytes_to_mb(torch.cuda.max_memory_allocated(self.device))
        max_reserved = bytes_to_mb(torch.cuda.max_memory_reserved(self.device))
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        nvidia_used, nvidia_total = self._query_nvidia_smi()
        return MemorySample(
            phase=self._phase,
            label=label,
            elapsed_s=elapsed_s,
            allocated_mb=allocated,
            reserved_mb=reserved,
            max_allocated_mb=max_allocated,
            max_reserved_mb=max_reserved,
            free_mb=bytes_to_mb(free_bytes),
            total_mb=bytes_to_mb(total_bytes),
            nvidia_smi_used_mb=nvidia_used,
            nvidia_smi_total_mb=nvidia_total,
        )

    def _append_sample(self, label: str) -> None:
        with self._lock:
            self.samples.append(self._sample(label))

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self._append_sample("poll")

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._append_sample("monitor_start")
        if self.interval_s > 0:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase
            self.samples.append(self._sample("phase_start"))

    def snapshot(self, label: str) -> None:
        self._append_sample(label)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_s * 2.0, 1.0))
        self._append_sample("monitor_stop")


def summarize_phase(
    samples: list[MemorySample],
    *,
    phase: str,
    steps: int,
    elapsed_s: float,
    peak_torch_allocated_mb: float | None,
    peak_torch_reserved_mb: float | None,
) -> PhaseSummary:
    phase_samples = [sample for sample in samples if sample.phase == phase]
    sampled_allocated = [
        sample.allocated_mb
        for sample in phase_samples
        if sample.allocated_mb is not None
    ]
    sampled_reserved = [
        sample.reserved_mb for sample in phase_samples if sample.reserved_mb is not None
    ]
    sampled_nvidia = [
        sample.nvidia_smi_used_mb
        for sample in phase_samples
        if sample.nvidia_smi_used_mb is not None
    ]
    return PhaseSummary(
        phase=phase,
        steps=steps,
        elapsed_s=elapsed_s,
        peak_torch_allocated_mb=peak_torch_allocated_mb,
        peak_torch_reserved_mb=peak_torch_reserved_mb,
        peak_sampled_allocated_mb=max(sampled_allocated) if sampled_allocated else None,
        peak_sampled_reserved_mb=max(sampled_reserved) if sampled_reserved else None,
        peak_nvidia_smi_used_mb=max(sampled_nvidia) if sampled_nvidia else None,
    )


def write_samples_csv(path: Path, samples: list[MemorySample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(samples[0]).keys()))
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def build_custom_properties(
    *,
    skip_heat_flux_property: bool,
    property_model_path: Path,
    extensions_directory: Path | None,
    structure_path: Path,
    device: str,
    output_key: str,
) -> dict[str, dict[str, Any]]:
    if skip_heat_flux_property:
        return {}

    calculator = HeatFluxPropertyCalculator(
        model_path=property_model_path,
        extensions_directory=extensions_directory,
        ref_topology=structure_path,
        device=device,
        output_key=output_key,
    )
    property_function = lambda properties: calculator(properties)
    return {
        output_key: {
            "func": property_function,
            "dimension": "undefined",
            "size": 3,
            "help": f"{output_key} computed with a custom metatomic property model",
        }
    }


def run_phase(
    *,
    xml_text: str,
    phase_name: str,
    steps: int,
    monitor: CudaMemoryMonitor,
    output_dir: Path,
    custom_properties: dict[str, dict[str, Any]],
    device: str,
) -> PhaseSummary:
    from ipi.utils.scripting import InteractiveSimulation

    monitor.set_phase(phase_name)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    monitor.snapshot(f"{phase_name}_before_simulation")

    start = time.perf_counter()
    with working_directory(output_dir):
        simulation = InteractiveSimulation(xml_text, custom_properties=custom_properties)
        monitor.snapshot(f"{phase_name}_simulation_created")
        simulation.run(steps)
    torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - start
    monitor.snapshot(f"{phase_name}_after_run")

    return summarize_phase(
        monitor.samples,
        phase=phase_name,
        steps=steps,
        elapsed_s=elapsed_s,
        peak_torch_allocated_mb=bytes_to_mb(torch.cuda.max_memory_allocated(device)),
        peak_torch_reserved_mb=bytes_to_mb(torch.cuda.max_memory_reserved(device)),
    )


def main() -> int:
    args = parse_args()
    validate_args(args)

    property_model_path = args.property_model or args.model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = resolved_output_paths(args)

    template_text = args.xml_template.read_text(encoding="utf-8")
    model_parameter = format_model_parameter(args.model, args.extensions_directory)
    structure_path = args.structure.resolve()

    custom_properties = build_custom_properties(
        skip_heat_flux_property=args.skip_heat_flux_property,
        property_model_path=property_model_path.resolve(),
        extensions_directory=(
            args.extensions_directory.resolve()
            if args.extensions_directory is not None
            else None
        ),
        structure_path=structure_path,
        device=args.device,
        output_key=args.property_output_key,
    )

    pre_equil_xml = None
    if args.pre_equil_steps > 0:
        pre_equil_xml = render_phase_xml(
            template_text,
            structure_path=structure_path,
            model_parameter=model_parameter,
            device=args.device,
            timestep_fs=args.dt,
            temperature_k=args.temperature,
            stride=args.stride,
            seed=args.seed,
            init_structure=str(structure_path),
            init_mode="ase",
            tau_fs=args.pre_equil_tau,
            prefix="pre-equil",
            init_velocities_xml=(
                f"<velocities mode='thermal' units='ase'> {args.temperature} </velocities>"
            ),
            include_heat_flux_property=not args.skip_heat_flux_property,
            property_output_key=args.property_output_key,
        )

    production_init_structure = (
        "pre-equil.ckpt" if args.pre_equil_steps > 0 else str(structure_path)
    )
    production_init_mode = "chk" if args.pre_equil_steps > 0 else "ase"
    production_init_velocities = "" if args.pre_equil_steps > 0 else (
        f"<velocities mode='thermal' units='ase'> {args.temperature} </velocities>"
    )
    production_xml = render_phase_xml(
        template_text,
        structure_path=structure_path,
        model_parameter=model_parameter,
        device=args.device,
        timestep_fs=args.dt,
        temperature_k=args.temperature,
        stride=args.stride,
        seed=args.seed,
        init_structure=production_init_structure,
        init_mode=production_init_mode,
        tau_fs=args.production_tau,
        prefix="production",
        init_velocities_xml=production_init_velocities,
        include_heat_flux_property=not args.skip_heat_flux_property,
        property_output_key=args.property_output_key,
    )

    print(f"Metatomic import source: {IMPORT_SOURCE}")
    print(f"Output directory: {args.output_dir}")
    print(f"Force-field model: {args.model.resolve()}")
    print(f"Property model: {property_model_path.resolve()}")
    print(f"Structure: {structure_path}")
    print(f"XML template: {args.xml_template.resolve()}")
    print(f"Device: {args.device}")
    print(f"Property output key: {args.property_output_key}")
    print(f"Monitoring interval: {args.monitor_interval_s} s")
    print(f"Polling nvidia-smi: {args.sample_nvidia_smi}")
    print()

    monitor = CudaMemoryMonitor(
        device=args.device,
        interval_s=args.monitor_interval_s,
        sample_nvidia_smi=args.sample_nvidia_smi,
    )
    monitor.start()

    phase_summaries: list[PhaseSummary] = []
    try:
        if pre_equil_xml is not None:
            phase_summaries.append(
                run_phase(
                    xml_text=pre_equil_xml,
                    phase_name="pre_equil",
                    steps=args.pre_equil_steps,
                    monitor=monitor,
                    output_dir=args.output_dir,
                    custom_properties=custom_properties,
                    device=args.device,
                )
            )
        phase_summaries.append(
            run_phase(
                xml_text=production_xml,
                phase_name="production",
                steps=args.production_steps,
                monitor=monitor,
                output_dir=args.output_dir,
                custom_properties=custom_properties,
                device=args.device,
            )
        )
    finally:
        monitor.stop()

    if monitor.samples:
        write_samples_csv(csv_path, monitor.samples)

    payload = {
        "model": str(args.model.resolve()),
        "property_model": str(property_model_path.resolve()),
        "extensions_directory": (
            None
            if args.extensions_directory is None
            else str(args.extensions_directory.resolve())
        ),
        "structure": str(structure_path),
        "xml_template": str(args.xml_template.resolve()),
        "device": args.device,
        "dt_fs": args.dt,
        "temperature_k": args.temperature,
        "stride": args.stride,
        "seed": args.seed,
        "pre_equil_steps": args.pre_equil_steps,
        "production_steps": args.production_steps,
        "pre_equil_tau_fs": args.pre_equil_tau,
        "production_tau_fs": args.production_tau,
        "skip_heat_flux_property": args.skip_heat_flux_property,
        "property_output_key": args.property_output_key,
        "monitor_interval_s": args.monitor_interval_s,
        "sample_nvidia_smi": args.sample_nvidia_smi,
        "metatomic_import_source": IMPORT_SOURCE,
        "csv": str(csv_path),
        "json": str(json_path),
        "phase_summaries": [asdict(summary) for summary in phase_summaries],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for summary in phase_summaries:
        print(f"{summary.phase}:")
        print(f"  steps: {summary.steps}")
        print(f"  elapsed: {summary.elapsed_s:.2f} s")
        print(
            "  torch peak allocated/reserved [MB]: "
            f"{format_optional_mb(summary.peak_torch_allocated_mb)} / "
            f"{format_optional_mb(summary.peak_torch_reserved_mb)}"
        )
        print(
            "  sampled peak allocated/reserved [MB]: "
            f"{format_optional_mb(summary.peak_sampled_allocated_mb)} / "
            f"{format_optional_mb(summary.peak_sampled_reserved_mb)}"
        )
        print(
            "  sampled peak nvidia-smi used [MB]: "
            f"{format_optional_mb(summary.peak_nvidia_smi_used_mb)}"
        )

    print()
    print(f"Wrote CUDA-memory timeline to {csv_path}")
    print(f"Wrote phase summary to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
