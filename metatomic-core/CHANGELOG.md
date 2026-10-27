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

### Fixed

- Converting the units of model outputs no longer panics when the outputs are
  immutable arrays (e.g. JAX arrays, whose copies are read-only too): the copy is
  scaled in memory owned by metatomic and handed back through
  `mts_array_t.from_dlpack`.
