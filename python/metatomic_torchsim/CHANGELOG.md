# Changelog

All notable changes to metatomic-torchsim are documented here, following the
[keep a changelog](https://keepachangelog.com/en/1.1.0/) format. This project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- Possible sections for each package:
### Added

### Fixed

### Changed

### Removed
-->

## [Unreleased](https://github.com/metatensor/metatomic/)

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
