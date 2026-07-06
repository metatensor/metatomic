# Changelog

All notable changes to metatomic-ase are documented here, following the
[keep a changelog](https://keepachangelog.com/en/1.1.0/) format. This project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/metatensor/metatomic/)

### Added

- `MetatomicCalculator` now resolves and exposes an `energy_ensemble` output
  (respecting `variants`), like it already does for `energy` and
  `energy_uncertainty`.
- `MetatomicCalculator.run_model()` now makes `positions` and `cell` require grad
  before building the `System` whenever any requested output has a non-empty
  `explicit_gradients`, so models that compute their own gradients (e.g. an
  ensemble of forces/stress alongside `energy_ensemble`) can do so through the
  standard `run_model()` call, without any extra autograd handling on the
  caller's side.

<!-- Possible sections
### Fixed

### Changed

### Removed
-->

## [Version 0.1.1](https://github.com/metatensor/metatomic/releases/tag/metatomic-ase-v0.1.1) - 2026-05-13

### Added

- `non_conservative` in `MetatomicCalculator` now also accepts `"forces"` and
  `"stress"`. `"forces"` reads forces directly from the model while still
  computing stress via autograd; `"stress"` does the reverse.

## [Version 0.1.0](https://github.com/metatensor/metatomic/releases/tag/metatomic-ase-v0.1.0) - 2026-03-25

- `metatomic-ase` is now a standalone package, containing the ASE integration
  for metatomic models.
