r"""
.. _jax-tutorial-model:

Writing models with JAX
=======================

.. py:currentmodule:: metatomic

This tutorial shows how to write a metatomic model with `JAX <https://docs.jax.dev>`_,
using :py:class:`JaxModel`. As an example, we write a shifted Lennard-Jones potential
for argon. The model only computes the energy: :py:class:`JaxModel` computes the forces
and the stress with JAX automatic differentiation, through the pair vectors given by
the simulation engine.

We then check the results against
:py:class:`metatomic.testing.LennardJonesModel`, a NumPy implementation of the same
potential which computes its gradients explicitly, and use the same model directly from
JAX to run a :py:func:`jax.jit`-compiled molecular dynamics simulation.
"""

# sphinx_gallery_thumbnail_number = 1

import ase.build
import ase.md.velocitydistribution
import ase.units
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import vesin
from metatensor import Labels, TensorBlock, TensorMap

from metatomic import (
    JaxModel,
    JaxSystem,
    ModelCapabilities,
    ModelMetadata,
    PairListOptions,
    Quantity,
    System,
    execute_model,
)
from metatomic.testing import LennardJonesModel


# %%
#
# The model declares ``dtype="float64"``, and JAX uses ``float32`` unless told
# otherwise. :py:class:`JaxModel` refuses to run a ``float64`` model without this
# option, instead of silently losing precision.

jax.config.update("jax_enable_x64", True)

# %%
#
# Defining the model
# ------------------
#
# The energy of each system is a sum over the pairs within the cutoff. The pair
# vectors are given as a :py:class:`jax.Array`, so the computation can be a
# :py:func:`jax.jit`-compiled function. It is re-compiled whenever the number of pairs
# changes; padding the pairs to a few fixed sizes avoids this for larger models.


@jax.jit
def lennard_jones(vectors, epsilon, sigma, shift):
    sigma_r_6 = (sigma**2 / jnp.sum(vectors**2, axis=1)) ** 3
    return jnp.sum(4.0 * epsilon * (sigma_r_6**2 - sigma_r_6) - shift)


# %%
#
# A :py:class:`JaxModel` implements :py:meth:`JaxModel.forward` instead of
# ``execute_inner``. ``forward`` receives the systems as :py:class:`JaxSystem`, with
# all the arrays (positions, cell, pairs, custom inputs) as JAX arrays, and returns
# :py:class:`metatensor.TensorMap` containing JAX arrays. Both are JAX pytrees, so JAX
# can differentiate through them.


class JaxLennardJones(JaxModel):
    def __init__(self, *, cutoff, epsilon, sigma):
        self.cutoff = cutoff
        self.epsilon = epsilon
        self.sigma = sigma
        # shift the energy to 0 at the cutoff
        self.shift = 4 * epsilon * ((sigma / cutoff) ** 12 - (sigma / cutoff) ** 6)
        self.pairs = PairListOptions(cutoff=cutoff, full_list=False, strict=True)

    def capabilities(self):
        return ModelCapabilities(
            atomic_types=[18],
            interaction_range=self.cutoff,
            length_unit="angstrom",
            supported_devices=["cpu"],
            dtype="float64",
            outputs=[Quantity(name="energy", unit="eV", sample_kind="system")],
        )

    def metadata(self):
        return ModelMetadata(name="Lennard-Jones (JAX)")

    def requested_pair_lists(self, requested_outputs):
        return [self.pairs]

    def forward(self, systems, selected_atoms, outputs):
        if selected_atoms is not None:
            raise ValueError("this model does not support selected_atoms")

        energies = jnp.stack(
            [
                lennard_jones(
                    system.pairs(self.pairs).values.reshape(-1, 3),
                    self.epsilon,
                    self.sigma,
                    self.shift,
                )
                for system in systems
            ]
        )
        block = TensorBlock(
            values=energies.reshape(-1, 1),
            samples=Labels.range("system", len(systems)),
            components=[],
            properties=Labels.range("energy", 1),
        )
        # the only output declared in the capabilities is the energy
        return [TensorMap(Labels.range("_", 1), [block]) for _ in outputs]


model = JaxLennardJones(cutoff=7.0, epsilon=0.0104, sigma=3.4)

# %%
#
# The model did not declare any gradient of the energy, but :py:class:`JaxModel` adds
# the ones it can compute with autodiff to the capabilities:

print(model.capabilities().outputs[0].gradients)

# %%
#
# Running the model
# -----------------
#
# Simulation engines create a :py:class:`System` for each structure, and compute the
# pair lists requested by the model. Here, we do the same for an ASE ``Atoms``, using
# `vesin <https://github.com/Luthaf/vesin>`_ to compute the pairs.


def system_from_atoms(atoms, options):
    system = System(
        "angstrom",
        atoms.numbers.astype(np.int32),
        atoms.positions,
        atoms.cell[:],
        atoms.pbc,
    )

    neighbors = vesin.NeighborList(cutoff=options.cutoff, full_list=options.full_list)
    i, j, shifts, vectors = neighbors.compute(
        points=atoms.positions,
        box=atoms.cell[:],
        periodic=atoms.pbc,
        quantities="ijSD",
    )
    pairs = TensorBlock(
        values=vectors.reshape(-1, 3, 1),
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            np.column_stack([i, j, shifts]).astype(np.int32),
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )
    system.add_pairs(options, pairs)
    return system


atoms = ase.build.bulk("Ar", "fcc", a=5.26, cubic=True).repeat(2)
atoms.rattle(0.05, seed=0)

(options,) = model.requested_pair_lists([])
system = system_from_atoms(atoms, options)

# %%
#
# We request the energy with its gradients with respect to the positions and the
# strain, i.e. the forces and the virial.

energy = Quantity(
    name="energy", unit="eV", sample_kind="system", gradients=["positions", "strain"]
)

(result,) = execute_model(model, [system], [energy], check_consistency=True)
block = result.block()
forces = -np.asarray(block.gradient("positions").values).reshape(-1, 3)
virial = -np.asarray(block.gradient("strain").values).reshape(3, 3)

print("energy:", block.values[0, 0], "eV")
print("forces on the first atom:", forces[0], "eV/Å")

# %%
#
# The NumPy implementation of the same potential, in
# :py:class:`metatomic.testing.LennardJonesModel`, computes the gradients explicitly.
# Both models give the same energy, forces and virial:

reference = LennardJonesModel(
    atomic_type=18,
    cutoff=7.0,
    epsilon=0.0104,
    sigma=3.4,
    length_unit="angstrom",
    energy_unit="eV",
)
(expected,) = execute_model(reference, [system], [energy])
expected = expected.block()

np.testing.assert_allclose(block.values, expected.values)
np.testing.assert_allclose(
    forces, -np.asarray(expected.gradient("positions").values).reshape(-1, 3)
)
np.testing.assert_allclose(
    virial, -np.asarray(expected.gradient("strain").values).reshape(3, 3)
)

# %%
#
# Equation of state
# -----------------
#
# The gradient with respect to strain gives the pressure,
# :math:`P = -\mathrm{tr}(\partial E / \partial \varepsilon) / 3V`. This is the
# derivative :math:`-\partial E / \partial V` of the energy along an isotropic
# compression, as we can check on the equation of state of the crystal.

lattice = np.linspace(5.1, 5.6, 21)
volumes, energies, pressures = [], [], []
for a in lattice:
    crystal = ase.build.bulk("Ar", "fcc", a=a, cubic=True).repeat(2)
    (out,) = execute_model(model, [system_from_atoms(crystal, options)], [energy])

    strain_gradient = np.asarray(out.block().gradient("strain").values).reshape(3, 3)
    volumes.append(crystal.get_volume())
    energies.append(out.block().values[0, 0])
    pressures.append(-np.trace(strain_gradient) / (3 * crystal.get_volume()))

volumes = np.array(volumes)
finite_differences = -np.gradient(energies, volumes, edge_order=2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), layout="constrained")
ax1.plot(volumes, energies, "o-")
ax1.set(xlabel="volume / Å$^3$", ylabel="energy / eV")
ax2.plot(volumes, pressures, "o-", label="from the strain gradient")
ax2.plot(volumes, finite_differences, "x", label=r"$-\partial E / \partial V$")
ax2.set(xlabel="volume / Å$^3$", ylabel="pressure / eV Å$^{-3}$")
ax2.legend()
plt.show()

# %%
#
# Using the model from JAX
# ------------------------
#
# :py:func:`execute_model` goes through the metatomic C API, so it can not be used
# inside JAX transformations. :py:meth:`JaxModel.evaluate` does the same work with JAX
# only, on :py:class:`JaxSystem`: the pair lists are static data (the samples of the
# pairs), and the pair vectors are computed from the positions and cell. Systems and
# the returned :py:class:`metatensor.TensorMap` are pytrees, so
# :py:meth:`JaxModel.evaluate` works inside :py:func:`jax.jit`, :py:func:`jax.grad`,
# :py:func:`jax.vmap`, ...

i, j, shifts = vesin.NeighborList(cutoff=options.cutoff, full_list=False).compute(
    points=atoms.positions, box=atoms.cell[:], periodic=atoms.pbc, quantities="ijS"
)
samples = Labels(
    ["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
    np.column_stack([i, j, shifts]).astype(np.int32),
)
jax_system = JaxSystem(
    atoms.numbers,
    jnp.asarray(atoms.positions),
    jnp.asarray(atoms.cell[:]),
    atoms.pbc,
    pairs={options: samples},
)

forces_request = Quantity(
    name="energy", unit="eV", sample_kind="system", gradients=["positions"]
)


@jax.jit
def energy_and_forces(system):
    (out,) = model.evaluate([system], [forces_request])
    block = out.block()
    return block.values[0, 0], -block.gradient("positions").values.reshape(-1, 3)


jax_energy, jax_forces = energy_and_forces(jax_system)
np.testing.assert_allclose(jax_energy, block.values[0, 0])
np.testing.assert_allclose(jax_forces, forces)

# %%
#
# The whole simulation can then be compiled: a velocity-Verlet integrator, in ASE
# units, written with :py:func:`jax.lax.scan`. The pair list is fixed during the
# simulation, which is fine for this short run of a solid; longer simulations would
# re-compute it from time to time (with a fixed, padded size to avoid re-compiling).

masses = jnp.asarray(atoms.get_masses())[:, None]
dt = 2.0 * ase.units.fs

ase.md.velocitydistribution.MaxwellBoltzmannDistribution(
    atoms, temperature_K=50, rng=np.random.default_rng(0)
)


def verlet_step(state, _):
    system, velocities, forces = state
    velocities = velocities + 0.5 * dt * forces / masses
    system = system.replace(positions=system.positions + dt * velocities)
    potential, forces = energy_and_forces(system)
    velocities = velocities + 0.5 * dt * forces / masses
    kinetic = 0.5 * jnp.sum(masses * velocities**2)
    return (system, velocities, forces), (potential, kinetic)


@jax.jit
def simulate(system, velocities):
    _, forces = energy_and_forces(system)
    state = (system, velocities, forces)
    _, (potential, kinetic) = jax.lax.scan(verlet_step, state, None, length=500)
    return potential, kinetic


potential, kinetic = simulate(jax_system, jnp.asarray(atoms.get_velocities()))

time = dt * np.arange(1, 501) / ase.units.fs
fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
ax.plot(time, potential - potential[0], label="potential")
ax.plot(time, kinetic - kinetic[0], label="kinetic")
ax.plot(time, potential + kinetic - potential[0] - kinetic[0], label="total")
ax.set(xlabel="time / fs", ylabel="energy change / eV")
ax.legend()
plt.show()
