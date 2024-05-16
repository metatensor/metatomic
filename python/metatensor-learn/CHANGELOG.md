# Changelog

All notable changes to metatensor-learn are documented here, following the
[keep a changelog](https://keepachangelog.com/en/1.1.0/) format. This project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/lab-cosmo/metatensor/)

<!-- Possible sections

### Added

### Fixed

### Changed

### Removed
-->

## [Version 0.2.2](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-learn-v0.2.2) - 2024-05-16

### Added

- Added torch-style activation function module maps to `metatensor.learn.nn`: `ReLU`,
  `InvariantReLU`, `SiLU`, and `InvariantSiLU` (#597)
- Added torch-style neural network module maps to `metatensor.learn.nn`:
  `LayerNorm`, `InvariantLayerNorm`, `EquivariantLinear`, `Sequential`, `Tanh`,
  and `InvariantTanh` (#513)

### Fixed

- Set correct device for output of when torch default device is different than input device (#595)

## [Version 0.2.1](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-learn-v0.2.1) - 2024-03-01

### Changed

- `metatensor-learn` is no longer re-exported from `metatensor` and
  `metatensor.torch`, all functions are still available inside
  `metatensor.learn` and `metatensor.torch.learn`.

### Fixed

- Make sure the `Dataset` class is iterable (#500)

## [Version 0.2.0](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-learn-v0.2.0) - 2024-02-07

### Changed

- Pluralization removed for special kwarg `sample_ids` of `IndexedDataset` ->
  `sample_id`, and provided collate functions `group` and `group_and_join`
  updated accordingly.

### Fixed

- Removal of usage of Labels.range in nn modules to support torch.jit.save (#410)

## [Version 0.1.0](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-learn-v0.1.0) - 2024-01-26

### Added

- `ModuleMap` and `Linear` modules, following torch.nn.ModuleDict and
  torch.nn.Linear in PyTorch but adapted for `TensorMap`'s (#427)
- `Dataset` and `DataLoader` facilities, following the corresponding classes in
  PyTorch (#428)
