# `metatomic-ase`

[ASE](https://ase-lib.org/) integration for metatomic models.

This package allows you to use metatomic models as ASE
[`Calculator`](https://ase-lib.org/ase/calculators/calculators.html), integrating with any workflow based on ASE.

## Installation

```bash
pip install metatomic-ase
```

## Usage

```python
import ase.io
from metatomic_ase import MetatomicCalculator

# load atoms
atoms = ase.io.read("...")

# create a calculator from a saved .pt model
atoms.calc = MetatomicCalculator("model.pt", device="cuda")

# from here, all the normal ASE functionality is available
print(atoms.get_forces())
print(atoms.get_potential_energies())
```

For full documentation, see the [ASE engine
page](https://docs.metatensor.org/metatomic/latest/engines/ase.html) in
metatomic documentation.
