import os

import ase.build
import numpy as np
import pytest
import torch

import metatomic_lj_test  # noqa: E402
from metatomic.torch import (  # noqa: E402
    ModelOutput,
    load_atomistic_model,
)
from metatomic.torch.dftd3 import DFTD3  # noqa: E402
from metatomic_ase import MetatomicCalculator  # noqa: E402


ATOMIC_NUMBER = 18
LJ_CUTOFF = 7.0
D3_CUTOFF = 5.0
SIGMA = 3.405
EPSILON = 0.01032

# 7-point reference grid matching the standard Grimme tables.
_REF_GRID = 7

_D3_REFERENCE = {
    "default": {
        "energy": -1.0188759583539984,
        "forces": np.fromstring(
            """
            0.0002084039080024712 -3.467862794892182e-05 -0.000263223817825068
            -0.00017278871048103522 0.0003018412811340715 9.787762845236189e-05
            -3.3367422078741174e-05 -0.00032443217321935156 -0.0002607705624672538
            6.235971370244925e-05 5.414373502515965e-05 0.00021854490647336793
            -0.0002916157721216603 0.00010269640923196016 -0.00020179404593121883
            0.00013050609721404222 -0.0001861539857158246 -0.0003250494818327482
            0.00038991566408667563 -0.00013648944685717371 0.0004590010423048546
            -0.0003219204607732458 0.0005733778748687468 0.00039690877761913365
            -0.00011571438515570187 -0.00012679376556628142 0.00043703259093319513
            -9.379162844260196e-05 -9.727969807999965e-05 -0.00019243542456385057
            4.030606948923331e-06 -7.553519686181228e-05 -0.0002256712383325067
            -0.0001772319495738292 0.00024357567548714126 0.0002607180778666125
            -0.00027211811956500895 -0.0002338512662259616 -0.00020328023978406252
            -0.0002163544324985875 6.471306719024392e-06 0.0001257774076759196
            0.00039626022128839283 -0.00027987943644729174 -0.00038058480508726053
            0.0002975038256188972 0.00012467535379086757 -0.0001851974831943732
            -0.0001605430394938604 0.00019636866056801407 -3.5804716868354566e-05
            0.0001230766630100459 -2.1069753646196527e-05 0.0001732158770854017
            -0.00023316175357063207 -0.00031237165124216315 3.981504042780467e-05
            0.00030153406888305465 0.0003232349711854779 0.00013358566358198654
            -0.00015200146132537864 -0.00039939466068612123 -0.0004007614530016808
            3.5031611281731976e-05 0.00030169804383896673 0.0001888396611455928
            0.0001990861754647688 -3.825739632562757e-05 0.00014657091780395648
            0.00016273845848624804 0.0003903242846735669 0.00019944747067831767
            0.0002395495631279873 -0.00032430335140988184 -9.657601367764508e-05
            3.066211708143058e-05 -3.385409872327201e-05 0.00047992263155839654
            -0.00017101119639451653 -4.6508362727685354e-05 0.00029334352274510214
            -0.0001368089350143164 5.640443821079179e-06 -0.00030031761103211513
            3.410975750702981e-05 -0.00024133338861993467 -0.00035450929543904414
            0.00020678360306963968 0.0001356388114147171 0.00019454420532006786
            -0.00022017580111641233 0.00011275742738500044 -0.00024101501778844153
            -5.2946987168257624e-05 3.974198115970105e-05 -0.00017815421484645294
            """,
            sep=" ",
        ).reshape(-1, 3),
        "stress": np.fromstring(
            """
            0.0001547358372809402 -2.873094319922506e-08 2.3804693080061202e-08
            -2.873094319922506e-08 0.0001547421061951057 1.8730307424106115e-07
            2.3804693080061202e-08 1.8730307424106115e-07 0.00015491022360242685
            """,
            sep=" ",
        ).reshape(3, 3),
    },
    "doubled": {
        "energy": -7.0266754355466565,
    },
}


@pytest.fixture
def model_with_extension():
    return metatomic_lj_test.lennard_jones_model(
        atomic_type=ATOMIC_NUMBER,
        cutoff=LJ_CUTOFF,
        sigma=SIGMA,
        epsilon=EPSILON,
        length_unit="Angstrom",
        energy_unit="eV",
        with_extension=True,
    )


@pytest.fixture
def model_pure():
    return metatomic_lj_test.lennard_jones_model(
        atomic_type=ATOMIC_NUMBER,
        cutoff=LJ_CUTOFF,
        sigma=SIGMA,
        epsilon=EPSILON,
        length_unit="Angstrom",
        energy_unit="eV",
        with_extension=False,
    )


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


def _sorted_forces(block):
    """Return forces sorted by (system, atom) index. LJ pure shuffles samples."""
    samples = block.samples
    order = np.lexsort(
        (
            samples.column("atom").cpu().numpy(),
            samples.column("system").cpu().numpy(),
        )
    )
    return block.values.squeeze(-1).detach().cpu().numpy()[order]


def test_dftd3_default_cutoffs_use_grimme(model_with_extension):
    """When the caller does not specify cutoffs, the wrapper must default to
    the Grimme values (50 / 25 Bohr) converted into the model's length unit
    (here Angstrom)."""
    wrapper = DFTD3(
        model_with_extension,
        d3_params=_d3_params(),
        damping_params={"energy": _damping()},
    )
    BOHR_IN_ANGSTROM = 0.5291772105448199
    nls = wrapper.requested_neighbor_lists()
    assert len(nls) == 2
    assert nls[1].requestors() == ["DFTD3"]
    assert nls[1].cutoff == pytest.approx(50.0 * BOHR_IN_ANGSTROM)


def test_dftd3_energy_correction_matches_reference(atoms, model_with_extension):
    wrapped = DFTD3.wrap(
        model_with_extension,
        d3_params=_d3_params(),
        damping_params={"energy": _damping()},
        cutoff=D3_CUTOFF,
        cn_cutoff=D3_CUTOFF,
    )

    base_energy = float(
        _eval(
            model_with_extension, atoms, {"energy": ModelOutput(sample_kind="system")}
        )["energy"]
        .block()
        .values.item()
    )
    corrected_energy = float(
        _eval(wrapped, atoms, {"energy": ModelOutput(sample_kind="system")})["energy"]
        .block()
        .values.item()
    )

    expected_d3 = _D3_REFERENCE["default"]["energy"]
    np.testing.assert_allclose(
        corrected_energy - base_energy, expected_d3, rtol=1e-10, atol=1e-12
    )


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

    # LJ extension only computes 'energy/doubled' when 'energy' is also requested
    base_results = _eval(
        model_with_extension,
        atoms,
        {
            "energy": ModelOutput(sample_kind="system"),
            "energy/doubled": ModelOutput(sample_kind="system"),
        },
    )
    base_default = float(base_results["energy"].block().values.item())
    base_doubled = float(base_results["energy/doubled"].block().values.item())

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
        corrected_default - base_default, expected_default, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        corrected_doubled - base_doubled, expected_doubled, rtol=1e-10, atol=1e-12
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
    with pytest.raises(Exception, match="per-atom corrected energies"):
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
        DFTD3(
            model_with_extension,
            d3_params=_d3_params(),
            damping_params={"energy/does_not_exist": _damping()},
            cutoff=D3_CUTOFF,
            cn_cutoff=D3_CUTOFF,
        )


def test_dftd3_rejects_malformed_damping_key(model_with_extension):
    with pytest.raises(ValueError, match="must be 'energy' or 'energy/"):
        DFTD3(
            model_with_extension,
            d3_params=_d3_params(),
            damping_params={"not_an_energy_key": _damping()},
            cutoff=D3_CUTOFF,
            cn_cutoff=D3_CUTOFF,
        )


def test_dftd3_autograd_forces_match_conservative_plus_d3(atoms, model_with_extension):
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

    base_atoms = copy.deepcopy(atoms)
    base_atoms.calc = MetatomicCalculator(
        model_with_extension, check_consistency=False, uncertainty_threshold=None
    )
    base_forces = base_atoms.get_forces()

    d3_forces = _D3_REFERENCE["default"]["forces"]

    np.testing.assert_allclose(
        wrapped_forces, base_forces + d3_forces, atol=1e-10, rtol=1e-8
    )


def test_dftd3_autograd_stress_match_conservative_plus_d3(atoms, model_with_extension):
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

    base_atoms = copy.deepcopy(atoms)
    base_atoms.calc = MetatomicCalculator(
        model_with_extension, check_consistency=False, uncertainty_threshold=None
    )
    base_stress = _voigt_to_full(base_atoms.get_stress())

    d3_stress = _D3_REFERENCE["default"]["stress"]

    np.testing.assert_allclose(
        wrapped_stress, base_stress + d3_stress, atol=1e-12, rtol=1e-8
    )
