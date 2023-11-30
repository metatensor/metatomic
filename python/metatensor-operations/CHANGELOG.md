# Changelog

All notable changes to metatensor-operations are documented here, following the
[keep a changelog](https://keepachangelog.com/en/1.1.0/) format. This project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/lab-cosmo/metatensor/)

### Added

- `detach()` function to detach all values in a TensorMap/TensorBlock from any
  computational graph

### Fixed

### Changed

### Removed

- the `requires_grad` parameter of the `to()` operation. This is now covered by
  `detach()` or an explicit loop over the blocks.


## [Version 0.1.0](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-operations-v0.1.0) - 2023-10-11

### Added

- Creation operations: `empty_like()`, `ones_like()`, `zeros_like()`,
  `random_like()`, `block_from_array()`;
- Linear algebra: `dot()`, `lstsq()`, `solve()`;
- Logic function: `allclose()`, `equal()`, `equal_metadata()`;
- Manipulation operations: `drop_blocks()`, `join()`, `manipulate dimension`,
  `one_hot()`, `remove_gradients()`, `samples reduction`, `slice()`, `split()`,
  `to()`;
- Mathematical functions: `abs()`, `add()`, `divide()`, `multiply()`, `pow()`,
  `subtract()`;
- Set operations: `unique_metadata()`, `sort()`;
