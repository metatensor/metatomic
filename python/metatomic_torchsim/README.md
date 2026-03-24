# `metatomic-torchsim`

[TorchSim](https://torchsim.github.io/torch-sim/) integration for metatomic
models.

This package allows you to wrap metatomic models as TorchSim `ModelInterface`
instances, enabling their use in TorchSim molecular dynamics and other
simulation workflows.

## Installation

```bash
pip install metatomic-torchsim
```

## Usage

```python
from metatomic_torchsim import MetatomicModel

# From a saved .pt model
model = MetatomicModel("model.pt", device="cuda")

# Use with TorchSim
output = model(sim_state)
energy = output["energy"]
forces = output["forces"]
stress = output["stress"]
```

For full documentation, see the [torch-sim engine
page](https://docs.metatensor.org/metatomic/latest/engines/torch-sim.html) in
metatomic documentation.
