# sig-rust

Rust port of sig-light: path signature and log signature computation with PyO3 bindings.

## Build Commands

```bash
# Rust
cargo build                    # Build Rust library
cargo test                     # Run Rust tests
cargo clippy --all-targets --all-features -- -D warnings  # Lint
cargo fmt --check              # Format check
cargo llvm-cov --lcov --output-path rust-lcov.info  # Test with coverage

# Python
uv sync --all-extras --group dev  # Install all dependencies
uv run maturin develop --release  # Build + install Python wheel
uv run pytest                     # Run Python tests
uv run ruff check .               # Python lint
uv run ruff format --check .      # Python format check
uv run ty check                   # Python type check
uv run pytest --cov=python/sig_rust --cov-report=term-missing  # Test with coverage

# Benchmarks
uv run python scripts/benchmark.py       # Quick comparison vs iisignature
uv run python scripts/benchmark_full.py  # Full profiling suite
```

## Architecture

- `src/lib.rs` -- crate root, module declarations
- `src/error.rs` -- `SigError` enum (thiserror)
- `src/types.rs` -- `Dim`, `Depth`, `LevelList`, `BatchedLevelList` newtypes
- `src/algebra.rs` -- tensor algebra: multiply, unconcatenate, log, segment signatures, adjoints
- `src/lyndon.rs` -- Lyndon words, factorization, projection matrices (SVD)
- `src/signature.rs` -- `sig()`, `siglength()`, `sigcombine()`, binary tree reduction
- `src/logsignature.rs` -- `logsig()`, `prepare()`, `logsiglength()`, BCH heuristic
- `src/bch.rs` -- Baker-Campbell-Hausdorff compiled program: precomputes bracket structure constants and emits flat FMA instructions for fast log-signature evaluation
- `src/backprop.rs` -- `sigbackprop()`, `sigjacobian()`, `logsigbackprop()`, reversibility trick
- `src/rotational.rs` -- 2D rotation-invariant features
- `src/transforms.rs` -- `sigjoin()`, `sigscale()` + backprop
- `src/python.rs` -- PyO3 bindings with adaptive rayon parallelism

## Key Design Decisions

- `LevelList`: flat `Vec<f64>` with offset-based level access for cache locality. All tensor algebra operates on this type.
- Reversibility trick in backprop: recovers prefix signatures via `tensor_unconcatenate_into` instead of storing all intermediates (O(siglength) vs O(n * siglength) memory).
- BCH vs S method: `logsignature.rs` selects BCH for small configs (d<=3, m<=4) where compiled FMA wins; S method (Horner tensor log) for larger configs.
- Rayon parallelism: `collect_par_or_seq` in `python.rs` uses a work-estimation heuristic (500K threshold, min 16 batch items) to avoid thread pool overhead on small batches.
- Release profile: fat LTO + single codegen unit for maximum optimization.

## CI/CD

- **CI** (`.github/workflows/cicd.yml`): Rust stable+nightly (fmt, clippy, test+coverage), Python 3.10-3.14 (ruff, ty, pytest+coverage). Coverage uploaded to Codecov with `rust` and `python` flags.
- **Publish** (`.github/workflows/publish.yml`): On GitHub release, builds wheels for Linux/macOS/Windows (x86_64 + aarch64) via maturin-action, publishes to PyPI.
- **Dependabot**: Weekly grouped updates for GitHub Actions, Cargo, and pip.
- Codecov token lives in the `codecov` GitHub environment. PyPI uses trusted publishing via the `pypi` environment.

## Testing

- Rust tests: `#[cfg(test)]` modules in `bch.rs` (bracket identities, BCH/S-method agreement, backprop agreement).
- Python tests: `python/tests/test_python_bindings.py` (105 tests) covering all public API functions, finite-difference gradient checks, and iisignature cross-validation.
- iisignature compatibility tests run when `iisignature` is installed (included in the `test` dependency group).
