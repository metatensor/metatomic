# metatomic-torchsim

TorchSim integration for metatomic atomistic models.

Wraps metatomic models as TorchSim `ModelInterface` instances, enabling their
use in TorchSim molecular dynamics and other simulation workflows.

## Installation

```bash
pip install metatomic-torchsim
```

For universal potential models, see
[upet](https://github.com/lab-cosmo/upet).

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

For full documentation, see the
[torch-sim engine page](https://docs.metatensor.org/metatomic/latest/engines/torch-sim.html).
