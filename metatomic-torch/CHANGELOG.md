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
