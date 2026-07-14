"""Smoke tests for the ORCA external-tool file protocol (no ORCA required)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest.mock import MagicMock

import ase.units
import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

EXAMPLE_DIR = Path(__file__).resolve().parent
COMMON_MODULE = EXAMPLE_DIR / "orca_common.py"
SERVER_MODULE = EXAMPLE_DIR / "metatomic-orca-server"
CLIENT_MODULE = EXAMPLE_DIR / "metatomic-orca-client"


def load_module(path: Path, name: str):
    mock_metatomic_ase = MagicMock()
    sys.modules.setdefault("metatomic_ase", mock_metatomic_ase)

    loader = SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


@pytest.fixture
def orca():
    module = load_module(COMMON_MODULE, "orca_common")
    module.clear_calculator_cache()
    yield module
    module.clear_calculator_cache()


@pytest.fixture
def orca_server(orca):
    module = load_module(SERVER_MODULE, "metatomic_orca_server")
    return module


@pytest.fixture
def orca_client():
    return load_module(CLIENT_MODULE, "metatomic_orca_client")


class FixedEnergyCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, energy_ev: float, forces_ev_angstrom: np.ndarray):
        super().__init__()
        self.energy_ev = energy_ev
        self.forces_ev_angstrom = np.asarray(forces_ev_angstrom, dtype=float)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        self.results = {
            "energy": self.energy_ev,
            "forces": self.forces_ev_angstrom.copy(),
        }


def _write_water_job(tmp_path: Path) -> Path:
    xyz_path = tmp_path / "water.xyz"
    xyz_path.write_text(
        "3\n"
        "water\n"
        "O 0.0 0.0 0.1173\n"
        "H 0.0 0.7572 -0.4692\n"
        "H 0.0 -0.7572 -0.4692\n"
    )
    extinp_path = tmp_path / "water_EXT.extinp.tmp"
    extinp_path.write_text(
        f"{xyz_path.name}\n"
        "0\n"
        "1\n"
        "1\n"
        "1\n"
    )
    return extinp_path


def test_read_extinp_and_xyz(tmp_path, orca):
    xyz_path = tmp_path / "water.xyz"
    xyz_path.write_text(
        "3\n"
        "water\n"
        "O 0.0 0.0 0.1173\n"
        "H 0.0 0.7572 -0.4692\n"
        "H 0.0 -0.7572 -0.4692\n"
    )
    extinp_path = tmp_path / "job_EXT.extinp.tmp"
    extinp_path.write_text(
        f"{xyz_path.name}\n"
        "0\n"
        "1\n"
        "1\n"
        "1\n"
    )

    extinp = orca.read_extinp(extinp_path)
    assert extinp.charge == 0
    assert extinp.multiplicity == 1
    assert extinp.do_gradient is True
    assert extinp.xyz_path == xyz_path.resolve()

    symbols, coords = orca.read_xyz(xyz_path)
    assert symbols == ["O", "H", "H"]
    assert len(coords) == 3


def test_write_engrad_roundtrip(tmp_path, orca):
    gradient = [0.0, 0.0, 0.1, -0.05, 0.0, 0.0, -0.05, 0.0, 0.0]
    engrad_path = tmp_path / "job.engrad"
    orca.write_engrad(engrad_path, natoms=3, energy_hartree=-0.5, gradient_hartree_bohr=gradient)

    text = engrad_path.read_text()
    assert "3" in text
    assert "-5.000000000000e-01" in text
    assert "1.000000000000e-01" in text


def test_forces_to_orca_gradient(orca):
    forces = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
    gradient = orca.forces_to_orca_gradient(forces)
    expected = -forces * ase.units.Bohr / ase.units.Hartree
    np.testing.assert_allclose(gradient, expected.reshape(-1))


def test_configure_cpu_threading(orca, monkeypatch):
    monkeypatch.delenv("METATOMIC_DISABLE_THREADING_CONFIG", raising=False)
    orca.reset_threading_config()

    mock_torch = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    configured = orca.configure_cpu_threading(4)
    assert configured == 4
    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert os.environ["MKL_NUM_THREADS"] == "4"
    mock_torch.set_num_threads.assert_called_once_with(4)
    mock_torch.set_num_interop_threads.assert_called_once_with(2)

    mock_torch.reset_mock()
    orca.configure_cpu_threading(4)
    mock_torch.set_num_threads.assert_not_called()

    monkeypatch.setenv("METATOMIC_DISABLE_THREADING_CONFIG", "1")
    orca.reset_threading_config()
    mock_torch.reset_mock()
    orca.configure_cpu_threading(8)
    mock_torch.set_num_threads.assert_not_called()


def test_run_orca_job_configures_threads(tmp_path, orca, monkeypatch):
    extinp_path = _write_water_job(tmp_path)

    energy_ev = -27.2
    forces = np.zeros((3, 3))
    forces[0, 2] = 0.01
    fake_calc = FixedEnergyCalculator(energy_ev, forces)

    settings = orca.MetatomicOrcaSettings(model=tmp_path / "fake.pt")
    settings.model.write_text("placeholder")

    monkeypatch.setattr(orca, "get_calculator", lambda _settings: fake_calc)
    configured: list[int] = []
    monkeypatch.setattr(
        orca,
        "configure_cpu_threading",
        lambda ncores: configured.append(ncores) or ncores,
    )

    engrad_path = orca.run_orca_job(extinp_path, settings)
    assert engrad_path == tmp_path / "water.engrad"
    assert engrad_path.is_file()
    assert configured == [1]

    extinp = orca.read_extinp(extinp_path)
    atoms = orca.atoms_from_extinp(extinp)
    assert atoms.info["charge"] == 0
    assert atoms.info["spin_multiplicity"] == 1

    expected_energy = energy_ev / ase.units.Hartree
    expected_gradient = orca.forces_to_orca_gradient(forces).tolist()
    content = engrad_path.read_text()
    assert f"{expected_energy:.12e}" in content
    for value in expected_gradient:
        assert f"{value: .12e}" in content


def test_server_handles_job(tmp_path, orca, orca_server, monkeypatch):
    extinp_path = _write_water_job(tmp_path)
    fake_model = tmp_path / "fake.pt"
    fake_model.write_text("placeholder")
    settings = orca.MetatomicOrcaSettings(model=fake_model)

    energy_ev = -13.6
    fake_calc = FixedEnergyCalculator(energy_ev, np.zeros((3, 3)))
    monkeypatch.setattr(orca, "get_calculator", lambda _settings: fake_calc)

    server = orca_server.MetatomicOrcaServer(default_settings=settings)
    try:
        result = server.handle(
            [extinp_path.name],
            str(tmp_path),
        )
        assert result["status"] == "Success"
        assert (tmp_path / "water.engrad").is_file()
    finally:
        server.shutdown()


def test_client_server_http_roundtrip(tmp_path, orca, orca_server, orca_client, monkeypatch):
    extinp_path = _write_water_job(tmp_path)
    fake_model = tmp_path / "fake.pt"
    fake_model.write_text("placeholder")
    settings = orca.MetatomicOrcaSettings(model=fake_model)

    fake_calc = FixedEnergyCalculator(-10.0, np.zeros((3, 3)))
    monkeypatch.setattr(orca, "get_calculator", lambda _settings: fake_calc)

    server = orca_server.MetatomicOrcaServer(default_settings=settings)
    handler = orca_server.create_handler(server)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    host, port = httpd.server_address
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        orca_client.send_to_server(
            f"{host}:{port}",
            [extinp_path.name],
            working_directory=str(tmp_path),
        )
        assert (tmp_path / "water.engrad").is_file()
    finally:
        server.shutdown()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2.0)


def test_healthz_endpoint(tmp_path, orca, orca_server, monkeypatch):
    fake_model = tmp_path / "fake.pt"
    fake_model.write_text("placeholder")
    settings = orca.MetatomicOrcaSettings(model=fake_model)
    monkeypatch.setattr(orca, "get_calculator", lambda _settings: FixedEnergyCalculator(0.0, np.zeros((1, 3))))

    server = orca_server.MetatomicOrcaServer(default_settings=settings)
    handler = orca_server.create_handler(server)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    host, port = httpd.server_address
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload == {"status": "OK"}
    finally:
        server.shutdown()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2.0)
