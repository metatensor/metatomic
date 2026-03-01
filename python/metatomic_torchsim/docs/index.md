# metatomic-torchsim

```{toctree}
:hidden:

tutorials/getting_started
```

```{toctree}
:hidden:
:caption: How-to Guides

howto/model_loading
howto/batched_simulations
```

```{toctree}
:hidden:
:caption: Understanding

explanation/architecture
```

```{toctree}
:hidden:
:caption: Reference

autoapi/metatomic/torchsim/index
changelog
```

**metatomic-torchsim** adapts [metatomic](https://docs.metatensor.org/metatomic/latest/)
atomistic models for use with [TorchSim](https://radical-ai.github.io/torch-sim/),
a differentiable molecular dynamics framework built on PyTorch.

## Features

- Run any metatomic-compatible model (PET-MAD, MACE, etc.) inside TorchSim
  simulations
- Compute energies, forces, and stresses via autograd
- Batch multiple systems in a single forward pass
- GPU-accelerated neighbor lists via nvalchemiops when available

## Quick install

```bash
pip install metatomic-torchsim
```

## Minimal example

```python
from metatomic.torchsim import MetatomicModel
import torch_sim as ts

model = MetatomicModel("model.pt", device="cpu")
sim_state = ts.io.atoms_to_state([atoms], model.device, model.dtype)
results = model(sim_state)

print(results["energy"])   # shape [n_systems]
print(results["forces"])   # shape [n_atoms, 3]
print(results["stress"])   # shape [n_systems, 3, 3]
```
