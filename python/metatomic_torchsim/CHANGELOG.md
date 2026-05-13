# Changelog

All notable changes to metatomic-torchsim are documented here, following the
[keep a changelog](https://keepachangelog.com/en/1.1.0/) format. This project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased](https://github.com/metatensor/metatomic/)

<!-- Possible sections
### Added

### Fixed

### Changed

### Removed
-->

## [Version 0.1.3](https://github.com/metatensor/metatomic/releases/tag/metatomic-torchsim-v0.1.3) - 2026-05-13

### Added

- `non_conservative` in `MetatomicModel` now also accepts `"forces"` and
  `"stress"`. `"forces"` reads forces directly from the model while still
  computing stress via autograd; `"stress"` does the reverse.

## [Version 0.1.2](https://github.com/metatensor/metatomic/releases/tag/metatomic-torchsim-v0.1.2) - 2026-04-22

### Changed

- Removed the upper-version pin on `torch-sim-atomistic` to make updating the
  code in there that re-exports this package easier.

## [Version 0.1.1](https://github.com/metatensor/metatomic/releases/tag/metatomic-torchsim-v0.1.1) - 2026-04-01

### Fixed

- The `metatomic-torchsim` wheel on PyPI now properly declares dependencies

## [Version 0.1.0](https://github.com/metatensor/metatomic/releases/tag/metatomic-torchsim-v0.1.0) - 2026-03-30

- `metatomic-torchsim` is now a standalone package, containing the TorchSim
  integration for metatomic models.

### Added

- Support for output variants via the `variants` parameter, matching the ASE
  calculator's variant selection
- Non-conservative forces and stresses via `non_conservative=True`, reading
  model outputs directly instead of autograd
- Per-atom energy uncertainty warnings via `uncertainty_threshold`, triggered
  when the model provides `energy_uncertainty` with `per_atom=True`
- `additional_outputs` parameter for requesting arbitrary extra model outputs
