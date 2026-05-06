import os
from typing import Dict, List, Optional

import ase.build
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (  # noqa: E402
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
    load_atomistic_model,
)
from metatomic.torch.dftd3 import DFTD3  # noqa: E402
from metatomic_ase import MetatomicCalculator  # noqa: E402


ATOMIC_NUMBER = 18
D3_CUTOFF = 5.0

# 7-point reference grid matching the standard Grimme tables.
_REF_GRID = 7

_D3_REFERENCE = {
    "default": {
        "energy": -1.0188759583539975,
        "forces": np.array(
            [
                [2.08403908e-04, -3.46786279e-05, -2.63223818e-04],
                [-1.72788710e-04, 3.01841281e-04, 9.78776285e-05],
                [-3.33674221e-05, -3.24432173e-04, -2.60770562e-04],
                [6.23597137e-05, 5.41437350e-05, 2.18544906e-04],
                [-2.91615772e-04, 1.02696409e-04, -2.01794046e-04],
                [1.30506097e-04, -1.86153986e-04, -3.25049482e-04],
                [3.89915664e-04, -1.36489447e-04, 4.59001042e-04],
                [-3.21920461e-04, 5.73377875e-04, 3.96908778e-04],
                [-1.15714385e-04, -1.26793766e-04, 4.37032591e-04],
                [-9.37916284e-05, -9.72796981e-05, -1.92435425e-04],
                [4.03060695e-06, -7.55351969e-05, -2.25671238e-04],
                [-1.77231950e-04, 2.43575675e-04, 2.60718078e-04],
                [-2.72118120e-04, -2.33851266e-04, -2.03280240e-04],
                [-2.16354432e-04, 6.47130672e-06, 1.25777408e-04],
                [3.96260221e-04, -2.79879436e-04, -3.80584805e-04],
                [2.97503826e-04, 1.24675354e-04, -1.85197483e-04],
                [-1.60543039e-04, 1.96368661e-04, -3.58047169e-05],
                [1.23076663e-04, -2.10697536e-05, 1.73215877e-04],
                [-2.33161754e-04, -3.12371651e-04, 3.98150404e-05],
                [3.01534069e-04, 3.23234971e-04, 1.33585664e-04],
                [-1.52001461e-04, -3.99394661e-04, -4.00761453e-04],
                [3.50316113e-05, 3.01698044e-04, 1.88839661e-04],
                [1.99086175e-04, -3.82573963e-05, 1.46570918e-04],
                [1.62738458e-04, 3.90324285e-04, 1.99447471e-04],
                [2.39549563e-04, -3.24303351e-04, -9.65760137e-05],
                [3.06621171e-05, -3.38540987e-05, 4.79922632e-04],
                [-1.71011196e-04, -4.65083627e-05, 2.93343523e-04],
                [-1.36808935e-04, 5.64044382e-06, -3.00317611e-04],
                [3.41097575e-05, -2.41333389e-04, -3.54509295e-04],
                [2.06783603e-04, 1.35638811e-04, 1.94544205e-04],
                [-2.20175801e-04, 1.12757427e-04, -2.41015018e-04],
                [-5.29469872e-05, 3.97419812e-05, -1.78154215e-04],
            ]
        ),
        "stress": np.array(
            [
                [1.54735837e-04, -2.87309432e-08, 2.38046931e-08],
                [-2.87309432e-08, 1.54742106e-04, 1.87303074e-07],
                [2.38046931e-08, 1.87303074e-07, 1.54910224e-04],
            ]
        ),
    },
    "doubled": {
        "energy": -6.9266754355466515,
    },
}


@pytest.fixture
def model_with_extension():
    return AtomisticModel(
        ZeroEnergyModel().eval(),
        ModelMetadata(),
        ModelCapabilities(
            outputs={
                "energy": ModelOutput(
                    sample_kind="system", unit="eV", description="D3-corrected energy"
                ),
                "energy/doubled": ModelOutput(
                    sample_kind="system",
                    unit="eV",
                    description="D3-corrected energy for doubled head",
                ),
            },
            atomic_types=[ATOMIC_NUMBER],
            interaction_range=0.0,
            length_unit="Angstrom",
            supported_devices=["cpu", "cuda"],
            dtype="float64",
        ),
    )


class ZeroEnergyModel(torch.nn.Module):
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        results = torch.jit.annotate(Dict[str, TensorMap], {})

        values = torch.jit.annotate(List[torch.Tensor], [])
        for system in systems:
            values.append((system.positions.sum() + system.cell.sum()) * 0.0)

        if len(values) == 0:
            raise ValueError("ZeroEnergyModel requires at least one system")

        device = values[0].device
        system_labels = Labels(
            "system",
            torch.arange(len(values), dtype=torch.int64, device=device).reshape(-1, 1),
        )
        keys = Labels("_", torch.tensor([[0]], dtype=torch.int64, device=device))
        base_values = torch.stack(values, dim=0).reshape(-1, 1)
        for name in outputs:
            output_values = base_values.clone()
            if name == "energy/doubled":
                output_values = 0.1 + output_values

            properties = Labels(
                "energy", torch.tensor([[0]], dtype=torch.int64, device=device)
            )
            block = TensorBlock(
                values=output_values,
                samples=system_labels,
                components=torch.jit.annotate(List[Labels], []),
                properties=properties,
            )
            blocks = torch.jit.annotate(List[TensorBlock], [block])
            results[name] = TensorMap(keys, blocks)

        return results

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return torch.jit.annotate(List[NeighborListOptions], [])


@pytest.fixture
def atoms():
    rng = np.random.default_rng(0xDEADBEEF)
    system = ase.build.bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))
    system.positions += 0.15 * rng.random(system.positions.shape)
    return system


def _d3_params():
    """Synthetic D3 reference tables matching the values used to generate the
    ground-truth snapshot.

    The new (Z, M) / (Z, Z, M, M) layout collapses to the same effective
    physics as the old (Z, Z, 5, 5) test fixture: a single Ar-Ar C6 of 100,
    one valid CN reference at 1.0, and rcov / r4r2 of 1.5 / 2.0.
    """
    size = ATOMIC_NUMBER + 1
    rcov = torch.zeros(size, dtype=torch.float64)
    rcov[1:] = 1.0
    rcov[ATOMIC_NUMBER] = 1.5

    r4r2 = torch.zeros(size, dtype=torch.float64)
    r4r2[1:] = 1.0
    r4r2[ATOMIC_NUMBER] = 2.0

    c6 = torch.zeros((size, size, _REF_GRID, _REF_GRID), dtype=torch.float64)
    c6[ATOMIC_NUMBER, ATOMIC_NUMBER] = 100.0

    cn_ref = torch.full((size, _REF_GRID), -1.0, dtype=torch.float64)
    # All 7 reference points share the same CN value, so the weights are
    # uniform and the effective C6 matches the per-pair (5, 5) test fixture.
    cn_ref[ATOMIC_NUMBER, :] = 1.0

    return {"rcov": rcov, "r4r2": r4r2, "c6": c6, "cn_ref": cn_ref}


def _damping(**overrides):
    p = {"a1": 0.4, "a2": 4.0, "s8": 1.0}
    p.update(overrides)
    return p


def _eval(model, atoms, outputs, check_consistency=True):
    calc = MetatomicCalculator(
        model,
        check_consistency=check_consistency,
        uncertainty_threshold=None,
    )
    return calc.run_model(atoms, outputs)


def test_dftd3_default_cutoffs_use_grimme(model_with_extension):
    """When the caller does not specify cutoffs, the wrapper must default to
    the Grimme values (50 / 25 Bohr) converted into the model's length unit
    (here Angstrom)."""
    wrapper = DFTD3.wrap(
        model_with_extension,
        d3_params=_d3_params(),
        damping_params={"energy": _damping()},
    )
    BOHR_IN_ANGSTROM = 0.5291772105448199
    nls = wrapper.requested_neighbor_lists()
    assert len(nls) == 1
    assert nls[0].requestors() == ["DFTD3"]
    assert nls[0].cutoff == pytest.approx(50.0 * BOHR_IN_ANGSTROM)


def test_dftd3_energy_correction_matches_reference(atoms, model_with_extension):
    wrapped = DFTD3.wrap(
        model_with_extension,
        d3_params=_d3_params(),
        damping_params={"energy": _damping()},
        cutoff=D3_CUTOFF,
        cn_cutoff=D3_CUTOFF,
    )

    corrected_energy = float(
        _eval(wrapped, atoms, {"energy": ModelOutput(sample_kind="system")})["energy"]
        .block()
        .values.item()
    )

    expected_d3 = _D3_REFERENCE["default"]["energy"]
    np.testing.assert_allclose(corrected_energy, expected_d3, rtol=1e-10, atol=1e-12)


def test_dftd3_multiple_variants_use_independent_damping(atoms, model_with_extension):
    damping_default = _damping(a1=0.4, a2=4.0, s8=1.0)
    damping_doubled = _damping(a1=0.3, a2=3.0, s8=2.0)
    wrapped = DFTD3.wrap(
        model_with_extension,
        d3_params=_d3_params(),
        damping_params={"energy": damping_default, "energy/doubled": damping_doubled},
        cutoff=D3_CUTOFF,
        cn_cutoff=D3_CUTOFF,
    )

    results = _eval(
        wrapped,
        atoms,
        {
            "energy": ModelOutput(sample_kind="system"),
            "energy/doubled": ModelOutput(sample_kind="system"),
        },
    )
    corrected_default = float(results["energy"].block().values.item())
    corrected_doubled = float(results["energy/doubled"].block().values.item())

    expected_default = _D3_REFERENCE["default"]["energy"]
    expected_doubled = _D3_REFERENCE["doubled"]["energy"]
    np.testing.assert_allclose(
        corrected_default, expected_default, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        corrected_doubled, expected_doubled, rtol=1e-10, atol=1e-12
    )
    assert not np.isclose(expected_default, expected_doubled)


def test_dftd3_rejects_per_atom_corrected_energy(model_with_extension, atoms):
    wrapped = DFTD3.wrap(
        model_with_extension,
        d3_params=_d3_params(),
        damping_params={"energy": _damping()},
        cutoff=D3_CUTOFF,
        cn_cutoff=D3_CUTOFF,
    )
    with pytest.raises(
        Exception, match="this model can not compute 'energy' per atom, only globally"
    ):
        _eval(wrapped, atoms, {"energy": ModelOutput(sample_kind="atom")})


def test_dftd3_save_and_reload(tmp_path, model_with_extension, atoms):
    wrapped = DFTD3.wrap(
        model_with_extension,
        d3_params=_d3_params(),
        damping_params={"energy": _damping()},
        cutoff=D3_CUTOFF,
        cn_cutoff=D3_CUTOFF,
    )
    original_energy = float(
        _eval(wrapped, atoms, {"energy": ModelOutput(sample_kind="system")})["energy"]
        .block()
        .values.item()
    )

    path = os.path.join(tmp_path, "dftd3.pt")
    wrapped.save(path)
    reloaded = load_atomistic_model(path)
    reloaded_energy = float(
        _eval(reloaded, atoms, {"energy": ModelOutput(sample_kind="system")})["energy"]
        .block()
        .values.item()
    )
    assert np.isclose(reloaded_energy, original_energy, atol=1e-5, rtol=1e-5)


def test_dftd3_rejects_unknown_variant(model_with_extension):
    with pytest.raises(ValueError, match="wrapped model does not expose"):
        DFTD3.wrap(
            model_with_extension,
            d3_params=_d3_params(),
            damping_params={"energy/does_not_exist": _damping()},
            cutoff=D3_CUTOFF,
            cn_cutoff=D3_CUTOFF,
        )


def test_dftd3_rejects_malformed_damping_key(model_with_extension):
    with pytest.raises(ValueError, match="must be 'energy' or 'energy/"):
        DFTD3.wrap(
            model_with_extension,
            d3_params=_d3_params(),
            damping_params={"not_an_energy_key": _damping()},
            cutoff=D3_CUTOFF,
            cn_cutoff=D3_CUTOFF,
        )


def test_dftd3_autograd_forces_match_d3_reference(atoms, model_with_extension):
    """The pure-PyTorch corrected energy is naturally differentiable through
    the neighbor-list distances. Verify that the autograd path
    ``MetatomicCalculator(..., non_conservative=False).get_forces(atoms)``
    yields conservative forces matching the frozen D3 smoke reference.
    """
    import copy

    wrapped = DFTD3.wrap(
        model_with_extension,
        d3_params=_d3_params(),
        damping_params={"energy": _damping()},
        cutoff=D3_CUTOFF,
        cn_cutoff=D3_CUTOFF,
    )

    wrapped_atoms = copy.deepcopy(atoms)
    wrapped_atoms.calc = MetatomicCalculator(
        wrapped, check_consistency=False, uncertainty_threshold=None
    )
    wrapped_forces = wrapped_atoms.get_forces()

    d3_forces = _D3_REFERENCE["default"]["forces"]

    np.testing.assert_allclose(wrapped_forces, d3_forces, atol=1e-10, rtol=1e-8)


def test_dftd3_autograd_stress_match_d3_reference(atoms, model_with_extension):
    """Same check for stress, via the strain-trick autograd path."""
    import copy

    wrapped = DFTD3.wrap(
        model_with_extension,
        d3_params=_d3_params(),
        damping_params={"energy": _damping()},
        cutoff=D3_CUTOFF,
        cn_cutoff=D3_CUTOFF,
    )

    def _voigt_to_full(voigt):
        return np.array(
            [
                [voigt[0], voigt[5], voigt[4]],
                [voigt[5], voigt[1], voigt[3]],
                [voigt[4], voigt[3], voigt[2]],
            ]
        )

    wrapped_atoms = copy.deepcopy(atoms)
    wrapped_atoms.calc = MetatomicCalculator(
        wrapped, check_consistency=False, uncertainty_threshold=None
    )
    wrapped_stress = _voigt_to_full(wrapped_atoms.get_stress())

    d3_stress = _D3_REFERENCE["default"]["stress"]

    np.testing.assert_allclose(wrapped_stress, d3_stress, atol=1e-12, rtol=1e-8)
