# Architecture

This page explains how `metatomic_torchsim.MetatomicModel` bridges TorchSim and
metatomic.

## SimState vs list of System

TorchSim represents a simulation as a single batched `SimState` containing all
atoms from all systems, with a `system_idx` tensor tracking ownership. Metatomic
expects a `list[System]` where each `System` holds one periodic structure.

`MetatomicModel.forward` converts between these representations:

1.  Split the batched positions and atomic numbers by `system_idx`
2.  Create one `System` per sub-structure with its own cell
3.  Call the model on the list of systems
4.  Concatenate results back into batched tensors

## Forces via autograd

Metatomic models typically output only total energies. Forces are computed as
the negative gradient of the energy with respect to atomic positions:

    F_i = -dE/dr_i

Before calling the model, each system\'s positions are detached and set to
`requires_grad_(True)`. After the forward pass, `torch.autograd.grad` computes
the derivatives.

## Stress via the strain trick

Stress is computed using the Knuth strain trick. An identity strain tensor (3x3,
`requires_grad=True`) is applied to both positions and cell vectors:

    r' = r @ strain
    h' = h @ strain

The stress per system is then:

    sigma = (1/V) * dE/d(strain)

where V is the cell volume. This gives the full 3x3 stress tensor without finite
differences.

## Neighbor lists

Models specify what neighbor lists they need via
`model.requested_neighbor_lists()`, which returns a list of
`NeighborListOptions` (cutoff radius, full vs half list).

The wrapper computes these using:

- **vesin**: Default backend for both CPU and GPU. Handles half and full
  neighbor lists. Systems on non-CPU/CUDA devices are temporarily moved to CPU
  for the computation.
- **nvalchemiops**: Used automatically on CUDA for full neighbor lists when
  installed. Keeps everything on GPU, avoiding host-device transfers.

The decision happens per-call in `_compute_requested_neighbors`: if all systems
are on CUDA and nvalchemiops is available, full-list requests go through
nvalchemi while half-list requests still use vesin.

## Why a separate package

metatomic-torchsim has its own versioning, release schedule, and dependency set
(`torch-sim-atomistic`). Keeping it separate from metatomic-torch avoids forcing
a torch-sim dependency on users who only need the ASE calculator or other
integrations.

The package is pure Python with no compiled extensions, making it lightweight to
install.
