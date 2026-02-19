# Changelog

All notable changes to metatomic-torch are documented here, following the [keep
a changelog](https://keepachangelog.com/en/1.1.0/) format. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/metatensor/metatomic/)

<!-- Possible sections for each package:

### Added

### Fixed

### Changed

### Removed
-->

### Changed

- Removed support for torch 2.1 and 2.2

## [Version 0.1.8](https://github.com/metatensor/metatomic/releases/tag/metatomic-torch-v0.1.8) - 2026-02-02

### Added

- Added the ability for models to request extra inputs from the simulation
  engine, on top of positions/cell/types. These extra inputs are requested by
  the model and stored on the system by the engine (like for neighbor lists).
  They are provided as TensorMap, stored in the system's data, and follow the
  same metadata structure as the standard outputs (anything that can be an output can also be an input).
- Added support for `charges`, `masses`, `velocities` as standard model inputs
  and outputs
- Added support for PyTorch v2.10

### Changed

- Properties in standard output/inputs should now consistently use singular
  (i.e. `position` instead of `positions`). The plural names are deprecated and
  will be remove in the future.

## [Version 0.1.7](https://github.com/metatensor/metatomic/releases/tag/metatomic-torch-v0.1.7) - 2025-11-26

### Fixed

- Fixed `metatomic.torch.ase_calculator.MetatomicCalculator` to work even when a
  model does not have an energy output (#121)

### Added

- Added support for torch v2.10
- Added support for loading TorchScript extensions from deepmd-kit (#98)

## [Version 0.1.6](https://github.com/metatensor/metatomic/releases/tag/metatomic-torch-v0.1.6) - 2025-11-14

### Added

- Added support for torch v2.9
- `ModelOutput` now has a `description` field, to carry more information
  about a given output.
- Added the `pick_output` function that can be used by simulation engines to
  pick the correct output based on what's available inside a model and which
  variant (if any) the user requested.
- Added `metatomic.torch.ase_calculator.SymmetrizedCalculator`, an ASE
  calculator that can average energy calculations over the O(3), SO(3) or space
  group of a crystal, to make unconstrained, non-equivariant models rotationally
  equivariant up to the integration order.

### Changed

- The `pick_device` function now returns a `torch::DeviceType` instead of a
  string in C++. This does not affect the Python API.
- It is now possible to construct an `AtomisticModel` with a torch module
  already compiled with TorchScript.

### Fixed

- Variants can now be used with non-standard outputs as well (#105)

### Removed

- We dropped support for Python 3.9, and now requires at least Python 3.10

## [Version 0.1.5](https://github.com/metatensor/metatomic/releases/tag/metatomic-torch-v0.1.5) - 2025-10-06

### Added

- Two new functions, `metatomic.torch.save_buffer` and
  `metatomic.torch.load_system_buffer`, allow to serialize and deserialize
  `System` objects to and from `torch.Tensor`

### Changed

- `metatomic.torch.save` and `metatomic.torch.load_system` are now implemented
  in C++
- We now requires at least cmake v3.22 to compile metatomic-torch

## [Version 0.1.4](https://github.com/metatensor/metatomic/releases/tag/metatomic-torch-v0.1.4) - 2025-09-11

### Added

- The code is now compatible with metatensor-torch v0.8.0
- The code is now compatible torch v2.8

- `System.to` accepts a `non_blocking` argument, with the same meaning as
  `torch.Tensor.to`.
- The ASE `MetatomicCalculator` will now send warnings if the model predicts a
  high per-atom uncertainty for its energy output.
- We now have two new standard outputs: `positions` and `momenta`, which can be
  used for direct structure prediction and bypassing time integration.

## [Version 0.1.3](https://github.com/metatensor/metatomic/releases/tag/metatomic-torch-v0.1.3) - 2025-07-25

### Fixed

- The logic to detect OpenMP dependencies in TorchScript extensions now takes
  into account the user's site-package directory (#65)
- `metatomic.torch.ase_calculator` is now lazy-loaded, and can be accessed
  directly after importing `metatomic.torch` (#59)

## [Version 0.1.2](https://github.com/metatensor/metatomic/releases/tag/metatomic-torch-v0.1.2) - 2025-06-06

### Fixed

- `register_autograd_neighbors` is now kept in the code by the TorchScript
  compiler. It was previously silently removed.
- When running `ase_calculator.Metatomic` with `non_conservative=True`, we no
  longer crash for NPT simulations.


## [Version 0.1.1](https://github.com/metatensor/metatomic/releases/tag/metatomic-torch-v0.1.1) - 2025-05-20

### Fixed

- `metatomic_torch` can now be built as part of the same cmake project as
  `metatensor` and `metatensor_torch` (#33)


## [Version 0.1.0](https://github.com/metatensor/metatomic/releases/tag/metatomic-torch-v0.1.0) - 2025-05-05

The first release of metatomic-torch, containing code for atomisitic model
extracted out of [metatensor-torch v0.7.5](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.7.5), with the following additional changes:

- Renamed `MetatensorAtomisticModel` to `AtomisticModel`
- Renamed `MetatensorCalculator` in the ASE interface to `MetatomicCalculator`
