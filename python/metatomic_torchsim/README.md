# metatomic-torchsim

TorchSim integration for metatomic atomistic models.

Wraps metatomic models as TorchSim `ModelInterface` instances, enabling their
use in TorchSim molecular dynamics and other simulation workflows.

## Installation

```bash
pip install metatomic-torchsim
```

To use metatrain checkpoints (`.ckpt` files) or the `pet-mad` shortcut:

```bash
pip install metatomic-torchsim[metatrain]
```

## Usage

```python
from metatomic.torchsim import MetatomicModel

# From a saved .pt model
model = MetatomicModel("model.pt", device="cuda")

# From a metatrain checkpoint (requires metatrain extra)
model = MetatomicModel("model.ckpt", device="cuda")

# PET-MAD shortcut (requires metatrain extra)
model = MetatomicModel("pet-mad", device="cuda")

# Use with TorchSim
output = model(sim_state)
energy = output["energy"]
forces = output["forces"]
stress = output["stress"]
```
