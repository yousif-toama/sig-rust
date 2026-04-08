# sig-rust

[![CICD](https://github.com/yousif-toama/sig-rust/actions/workflows/cicd.yml/badge.svg)](https://github.com/yousif-toama/sig-rust/actions/workflows/cicd.yml)
[![codecov](https://codecov.io/gh/yousif-toama/sig-rust/graph/badge.svg)](https://codecov.io/gh/yousif-toama/sig-rust)

Fast path signature and log signature computation in Rust with Python bindings. Drop-in replacement for [iisignature](https://github.com/bottler/iisignature) with identical API.

sig-rust computes signatures and log signatures of multidimensional piecewise-linear paths using Chen's identity and truncated tensor algebra. The Rust core provides single-threaded performance on par with iisignature's C++, and automatic multithreading via rayon for batched workloads.

## Installation

```bash
pip install sig-rust
```

Or with uv:

```bash
uv add sig-rust
```

Prebuilt wheels are available for Linux, macOS, and Windows (x86_64 and aarch64). Building from source requires a Rust toolchain.

## Quick Start

```python
import numpy as np
import sig_rust

# Define a 2D path
path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])

# Compute signature at depth 3
signature = sig_rust.sig(path, 3)

# Compute log signature
s = sig_rust.prepare(2, 3)
log_signature = sig_rust.logsig(path, s)

# Backpropagation
deriv = np.ones_like(signature)
grad = sig_rust.sigbackprop(deriv, path, 3)  # gradient w.r.t. path

# Batching: process multiple paths at once (auto-parallelized)
paths = np.random.randn(100, 50, 2)  # 100 paths, 50 points, 2D
sigs = sig_rust.sig(paths, 3)  # shape (100, siglength(2, 3))
```

## API Reference

### Signature

#### `sig(path, m, format=0)`

Compute the signature of a path truncated at depth `m`.

- **path**: numpy array of shape `(..., n, d)`. Extra leading dims are batched.
- **m**: truncation depth (positive integer).
- **format**: output format.
  - `0`: flat array of shape `(..., siglength(d, m))`.
  - `1`: list of `m` arrays, one per level.
  - `2`: cumulative prefix signatures, shape `(..., n-1, siglength(d, m))`.
- **Returns**: the path signature, excluding the level-0 term (always 1).

#### `siglength(d, m)`

Length of the signature output: `d + d^2 + ... + d^m`.

#### `sigcombine(sig1, sig2, d, m)`

Combine two signatures via Chen's identity. Supports batching.

### Log Signature

#### `prepare(d, m, method="auto")`

Precompute data for log signature computation.

- **method**: `"auto"` selects the best method, `"bch"` forces Baker-Campbell-Hausdorff, `"s"` forces the S method (Horner tensor log).

#### `logsig(path, s)`

Compute the log signature in the Lyndon basis. Supports batching.

#### `logsig_expanded(path, s)`

Compute the log signature in the full tensor expansion. Supports batching.

#### `logsiglength(d, m)`

Length of the log signature output (Witt's formula).

#### `basis(s)`

Get the Lyndon bracket labels for the log signature basis elements.

### Backpropagation

#### `sigbackprop(deriv, path, m)`

Gradient of a scalar loss w.r.t. the path, given gradient w.r.t. the signature.

- **deriv**: shape `(..., siglength(d, m))`.
- **path**: shape `(..., n, d)`.
- **Returns**: shape `(..., n, d)`.

#### `sigjacobian(path, m)`

Full Jacobian matrix of `sig()` w.r.t. the path.

- **Returns**: shape `(n, d, siglength(d, m))`.

#### `logsigbackprop(deriv, path, s)`

Gradient of a scalar loss w.r.t. the path, given gradient w.r.t. the log signature.

- **deriv**: shape `(..., logsiglength(d, m))`.
- **path**: shape `(..., n, d)`.
- **s**: prepared data from `prepare(d, m)`.
- **Returns**: shape `(..., n, d)`.

### Transforms

#### `sigjoin(sig, segment, d, m, fixedLast=nan)`

Extend a signature by appending a linear segment.

#### `sigjoinbackprop(deriv, sig, segment, d, m, fixedLast=nan)`

Gradient through `sigjoin`. Returns `(dsig, dsegment)` or `(dsig, dsegment, dfixedLast)`.

#### `sigscale(sig, scales, d, m)`

Rescale a signature as if each path dimension were multiplied by a factor. At level k, the multi-index `(i1,...,ik)` component is multiplied by `scales[i1] * ... * scales[ik]`.

#### `sigscalebackprop(deriv, sig, scales, d, m)`

Gradient through `sigscale`. Returns `(dsig, dscales)`.

### Rotation Invariants (2D paths)

#### `rotinv2dprepare(m, type="a")`

Precompute rotation-invariant features for 2D paths.

#### `rotinv2d(path, s)` / `rotinv2dlength(s)` / `rotinv2dcoeffs(s)`

Compute rotation-invariant features, get their count, or get coefficient matrices.

## Performance

Single-threaded performance vs iisignature (C++) on representative configs:

| Operation | Config | sig-rust | iisignature | Speedup |
|-----------|--------|----------|-------------|---------|
| sig | d=3, m=4, n=100 | 0.020ms | 0.024ms | 1.15x |
| sig | d=2, m=5, n=1000 | 0.179ms | 0.239ms | 1.33x |
| logsig | d=3, m=5, n=100 | 0.066ms | 0.131ms | 1.98x |
| sigbackprop | d=5, m=3, n=100 | 0.096ms | 0.083ms | 0.86x |

Batched operations scale across cores automatically:

| Operation | Config | sig-rust | iisignature | Speedup |
|-----------|--------|----------|-------------|---------|
| sig | B=200, d=3, m=4, n=50 | 1.25ms | 2.45ms | 1.96x |
| sigbackprop | B=200, d=3, m=4, n=50 | 3.07ms | 9.38ms | 3.06x |

vs sig-light (pure Python): 5-154x faster across all operations.

## Algorithm

sig-rust uses the standard approach for computing signatures of piecewise-linear paths:

1. **Segment signature**: for each linear segment with displacement `h`, the signature is the truncated exponential `exp(h) = 1 + h + h^2/2! + h^3/3! + ...` in the tensor algebra.

2. **Chen's identity**: the signature of a concatenated path equals the tensor product of the individual segment signatures: `S(path) = S(seg_1) * S(seg_2) * ... * S(seg_n)`.

3. **Log signature**: computed via Horner's tensor log or compiled Baker-Campbell-Hausdorff program, then projected onto the Lyndon word basis.

4. **Backpropagation**: reverse-mode differentiation through Chen's identity using a reversibility trick (O(siglength) memory instead of O(n * siglength)).

## Development

```bash
git clone https://github.com/yousif-toama/sig-rust.git
cd sig-rust
uv sync --all-extras --group dev --group test

# Rust
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check

# Python
uv run maturin develop --release
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
uv run ty check

# Benchmarks
uv run python scripts/benchmark_full.py
```

## License

MIT
