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
        "energy": -4.557688784512186,
        "forces": np.array(
            [
                [-0.00726508070174851, 0.00421613323677529, 0.00713047350689955],
                [0.00444419132651295, -0.0138453480050549, -0.00191253781443601],
                [-0.00308607387021654, 0.01232699781381951, 0.00900597104085257],
                [-0.00610167736545015, 0.00276299380227561, -0.00353193039472426],
                [0.01260083353435382, -0.00614033346641161, 0.00477616628028993],
                [-0.00043465811545061, 0.00566025704614823, 0.00934931094503733],
                [-0.00772758165588948, 0.00030335892908554, -0.01598145511929504],
                [0.00775120111007042, -0.02001380204825312, -0.00881644834538752],
                [0.0040023123129326, 0.00240434690975679, -0.01750620395815836],
                [0.00102021733779979, 0.00279214703020113, 0.00530657675982078],
                [0.00168171921365183, 0.00178250547909488, 0.00274908781463904],
                [0.00679706473916125, -0.0127589706130313, -0.0089965376824202],
                [0.00913127902302183, 0.01302685144580063, 0.00685386705634312],
                [0.01109794534808769, -0.00438573870534469, -0.0053358145115363],
                [-0.01208514260862648, 0.01365753638264123, 0.01206812791952485],
                [-0.01362855558638936, -0.00146861040263824, 0.00906093284477281],
                [0.00666587625609256, -0.00157434351635681, 0.00157821155234659],
                [-0.00371968595647699, -0.00444630226580355, -0.00021902175920297],
                [0.00791739330252776, 0.00810729776920197, -0.00481271020086399],
                [-0.00212081385321295, -0.00873496197538598, -0.00383113551715393],
                [0.00020810819027938, 0.01694212346347319, 0.01154587588359852],
                [0.00102035905274625, -0.01338970052217113, -0.00740874907733129],
                [-0.00809861648157284, 0.0039800283781339, -0.00353126147783306],
                [-0.00614531290019489, -0.01571869766478938, -0.0011870192141982],
                [-0.00697144589551845, 0.00835972553326233, 0.00344246550272658],
                [-0.00045746125560518, -0.00053953437674736, -0.01648164832703997],
                [0.00067758271174474, 0.0046551984437648, -0.00803243988498612],
                [0.00104514491467035, -0.00015729419107496, 0.00718322804358889],
                [-0.00335313838707656, 0.00512768136258867, 0.00740313395620435],
                [-0.00709951943713622, -0.0034666077071482, -0.00246717309068727],
                [0.0081580287143527, -0.00227504064164253, 0.00462850801794592],
                [0.00407550698255931, 0.00281010307583006, 0.00797014925066365],
            ]
        ),
        "stress": np.array(
            [
                [
                    7.0830378606643353e-03,
                    1.7653051504568702e-06,
                    1.2574319088078316e-06,
                ],
                [
                    1.7653051504568702e-06,
                    7.0822819743685574e-03,
                    3.6606343262798787e-06,
                ],
                [
                    1.2574319088078316e-06,
                    3.6606343262798787e-06,
                    7.0824372365694095e-03,
                ],
            ]
        ),
    },
    "doubled": {
        "energy": -6.195052365042283,
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
                "non_conservative_force": ModelOutput(
                    sample_kind="atom",
                    unit="eV/Angstrom",
                    description="D3-corrected direct forces",
                ),
                "non_conservative_stress": ModelOutput(
                    sample_kind="system",
                    unit="eV/Angstrom^3",
                    description="D3-corrected direct stress",
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
            if name == "energy" or name == "energy/doubled":
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

            elif name == "non_conservative_force":
                force_values = torch.jit.annotate(List[torch.Tensor], [])
                force_samples = torch.jit.annotate(List[torch.Tensor], [])
                for i, system in enumerate(systems):
                    n_atoms = system.positions.shape[0]
                    force_values.append(system.positions * 0.0)
                    force_samples.append(
                        torch.cat(
                            [
                                torch.full(
                                    (n_atoms, 1),
                                    i,
                                    dtype=torch.int64,
                                    device=device,
                                ),
                                torch.arange(
                                    n_atoms, dtype=torch.int64, device=device
                                ).reshape(-1, 1),
                            ],
                            dim=1,
                        )
                    )

                block = TensorBlock(
                    values=torch.cat(force_values, dim=0).unsqueeze(-1),
                    samples=Labels(["system", "atom"], torch.cat(force_samples, dim=0)),
                    components=[
                        Labels(
                            "xyz",
                            torch.arange(3, dtype=torch.int64, device=device).reshape(
                                -1, 1
                            ),
                        )
                    ],
                    properties=Labels(
                        "non_conservative_force",
                        torch.tensor([[0]], dtype=torch.int64, device=device),
                    ),
                )
                blocks = torch.jit.annotate(List[TensorBlock], [block])
                results[name] = TensorMap(keys, blocks)

            elif name == "non_conservative_stress":
                stress_values = torch.jit.annotate(List[torch.Tensor], [])
                for system in systems:
                    stress_values.append(
                        torch.zeros((3, 3), dtype=base_values.dtype, device=device)
                        + system.cell.sum() * 0.0
                    )

                block = TensorBlock(
                    values=torch.stack(stress_values, dim=0).unsqueeze(-1),
                    samples=system_labels,
                    components=[
                        Labels(
                            "xyz_1",
                            torch.arange(3, dtype=torch.int64, device=device).reshape(
                                -1, 1
                            ),
                        ),
                        Labels(
                            "xyz_2",
                            torch.arange(3, dtype=torch.int64, device=device).reshape(
                                -1, 1
                            ),
                        ),
                    ],
                    properties=Labels(
                        "non_conservative_stress",
                        torch.tensor([[0]], dtype=torch.int64, device=device),
                    ),
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


def test_dftd3_uses_packaged_parameters_by_default(model_with_extension):
    wrapper = DFTD3.wrap(
        model_with_extension,
        damping_params={"energy": _damping()},
        cutoff=D3_CUTOFF,
        cn_cutoff=D3_CUTOFF,
    )

    nls = wrapper.requested_neighbor_lists()
    assert len(nls) == 1
    assert nls[0].requestors() == ["DFTD3"]


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


def test_dftd3_non_conservative_save_and_reload(tmp_path, model_with_extension, atoms):
    import copy

    wrapped = DFTD3.wrap(
        model_with_extension,
        d3_params=_d3_params(),
        damping_params={"energy": _damping()},
        cutoff=D3_CUTOFF,
        cn_cutoff=D3_CUTOFF,
    )

    path = os.path.join(tmp_path, "dftd3.pt")
    wrapped.save(path)
    reloaded = load_atomistic_model(path)

    wrapped_atoms = copy.deepcopy(atoms)
    wrapped_atoms.calc = MetatomicCalculator(
        reloaded,
        check_consistency=False,
        non_conservative=True,
        uncertainty_threshold=None,
    )

    np.testing.assert_allclose(
        wrapped_atoms.get_forces(),
        _D3_REFERENCE["default"]["forces"],
        atol=1e-10,
        rtol=1e-8,
    )


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


def test_dftd3_autograd_outputs_match_d3_reference(atoms, model_with_extension):
    """The pure-PyTorch corrected energy is naturally differentiable through
    the neighbor-list distances. Verify that the autograd path
    yields conservative forces and stresses matching the frozen D3 smoke reference.
    """
    import copy

    def _voigt_to_full(voigt):
        return np.array(
            [
                [voigt[0], voigt[5], voigt[4]],
                [voigt[5], voigt[1], voigt[3]],
                [voigt[4], voigt[3], voigt[2]],
            ]
        )

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
    wrapped_stress = _voigt_to_full(wrapped_atoms.get_stress())

    d3_forces = _D3_REFERENCE["default"]["forces"]
    d3_stress = _D3_REFERENCE["default"]["stress"]

    np.testing.assert_allclose(wrapped_forces, d3_forces, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(wrapped_stress, d3_stress, atol=1e-12, rtol=1e-8)


@pytest.mark.parametrize("non_conservative", [True, "forces", "stress"])
def test_dftd3_non_conservative_outputs_match_d3_reference(
    atoms, model_with_extension, non_conservative
):
    """Direct force/stress outputs get the same D3 correction as the
    conservative autograd path. The mixed modes exercise direct forces with
    autograd stress and direct stress with autograd forces."""
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
        wrapped,
        check_consistency=False,
        non_conservative=non_conservative,
        uncertainty_threshold=None,
    )

    np.testing.assert_allclose(
        wrapped_atoms.get_forces(),
        _D3_REFERENCE["default"]["forces"],
        atol=1e-10,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        _voigt_to_full(wrapped_atoms.get_stress()),
        _D3_REFERENCE["default"]["stress"],
        atol=1e-12,
        rtol=1e-8,
    )
