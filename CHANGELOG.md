# Changelog

## 0.1.0 — 2026-09-04

### Added
- `python -m nearl.valid_installation`, static and runtime checks of the
  compiled extension that work without a GPU (#4)
- `CUDA_CHECK` macros: a failed CUDA call now raises at the point of failure
  instead of returning a zero-filled grid (#4)
- `benchmarks/`: pytest-benchmark suite, host/device and per-property
  profiling scripts (#6)
- CI for unit tests, ruff/clang-format, nvcc cross-compilation
  (sm_80/86/90 x CUDA 12.6, 13.2) and the documentation build
- A dynamic-features guide with a runnable example, and a citation section
  in the README (#8)
- Remove AmberTools dependency and add a script to install pytraj `scripts/instant_pytraj.sh`

### Changed
- Van der Waals radii from Bondi (1964) to Alvarez (2013) (#5)
- OpenBabel perception happens once per trajectory and is memoised per
  property (#7)
- `traj_to_obmol` builds from the topology instead of a temporary PDB

### Fixed
- `backboneness` was inverted (#5)
- `cache_properties` read a nonexistent `utils.VDWRADII` and five stale
  feature attributes (#5)
- A time window above 512 frames warns instead of silently truncating (#5)
- `readme` pointed at a nonexistent `README.md`, so the built package
  carried an empty description (#8)
