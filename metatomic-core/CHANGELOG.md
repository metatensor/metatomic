# Changelog

All notable changes to metatomic-core are documented here, following the [keep
a changelog](https://keepachangelog.com/en/1.1.0/) format. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/metatensor/metatensor/)

<!-- Possible sections

#### Added

#### Fixed

#### Changed

#### Removed
-->

### Added

- `JaxModel` (Python), to write metatomic models with JAX. `forward()` receives
  the systems as `JaxSystem` with JAX arrays, and returns `metatensor.TensorMap`
  with JAX arrays. The gradients of per-system outputs with respect to positions
  and strain (forces and stress) are computed with JAX automatic differentiation.
- `JaxModel.evaluate()` runs a `JaxModel` on `JaxSystem` with JAX only (instead of
  the C API used by `execute_model`), converting units and computing gradients,
  so it can be used inside `jax.jit`, `jax.grad`, `jax.vmap`, ...
- `JaxSystem` is a JAX pytree, and `JaxSystem.from_system()` creates one from a
  `System`. Pair vectors are computed from the positions and cell.

### Fixed

- Converting the units of model outputs no longer panics when the outputs are
  immutable arrays (e.g. JAX arrays, whose copies are read-only too): the copy is
  scaled in memory owned by metatomic and handed back through
  `mts_array_t.from_dlpack`.
