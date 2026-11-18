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

- The Lennard-Jones test plugin (`metatomic-lj-plugin`) now computes strain
  gradients of the per-system energy, so engines can test virial and stress
  with it.
