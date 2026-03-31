# sig-rust

Rust port of sig-light: path signature and log signature computation with PyO3 bindings.

## Build Commands

```bash
cargo build                    # Build Rust library
cargo test                     # Run Rust tests
cargo clippy --all-targets --all-features -- -D warnings  # Lint
cargo fmt --check              # Format check

maturin develop --release      # Build + install Python wheel
uv run pytest                  # Run Python tests
uv run ruff check              # Python lint
uv run ruff format --check     # Python format check
uv run ty check                # Python type check
```

## Architecture

- `src/lib.rs` -- crate root, module declarations
- `src/error.rs` -- `SigError` enum (thiserror)
- `src/types.rs` -- `Dim`, `Depth`, `LevelList`, `BatchedLevelList` newtypes
- `src/algebra.rs` -- tensor algebra: multiply, log, segment signatures, adjoints
- `src/lyndon.rs` -- Lyndon words, factorization, projection matrices (SVD)
- `src/signature.rs` -- `sig()`, `siglength()`, `sigcombine()`, binary tree reduction
- `src/logsignature.rs` -- `logsig()`, `prepare()`, `logsiglength()`
- `src/backprop.rs` -- `sigbackprop()`, `sigjacobian()`, `logsigbackprop()`
- `src/rotational.rs` -- 2D rotation-invariant features
- `src/transforms.rs` -- `sigjoin()`, `sigscale()` + backprop
- `src/python.rs` -- PyO3 bindings
