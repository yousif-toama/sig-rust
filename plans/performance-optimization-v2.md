# Performance Optimization Plan v2

**Date**: 2026-04-08
**Baseline**: Post-v1 optimization (commit pending)
**Goal**: Close remaining gaps where Rust is slower than iisignature C++

## Remaining Gaps

### 1. sigbackprop: 0.70-0.86x across all configs

**Root cause**: Per-iteration allocation in the backward pass.

The reversibility trick (v1) eliminated the O(n) intermediate storage, but
introduced two new costs per segment:

a. **Inverse segment clone + negate**: `inv_seg.copy_from(&seg_sigs[i])` then
   `inv_seg.negate_even_indexed_levels()` then `tensor_multiply_into()`. This
   is three passes over the data (copy, negate, multiply) when iisignature does
   it in one fused pass (`unconcatenateWith` iterates levels high-to-low,
   subtracting outer products in-place).

b. **tensor_multiply_adjoint allocates (da, db)** every iteration: returns two
   new `LevelList`s per call, causing n-1 allocations in the backward loop.

**Fix A — Fused unconcatenate** (`src/algebra.rs`):

Add `tensor_unconcatenate_into(result: &LevelList, seg: &LevelList, prev: &mut LevelList)`
that computes `prev = result * S(-seg)` without cloning or negating:

```rust
pub fn tensor_unconcatenate_into(
    result: &LevelList, seg: &LevelList, prev: &mut LevelList
) {
    // prev = result - seg (base terms)
    prev.set_sum_sub(result, seg);  // needs new method: dst = a - b for even levels, a + b for odd
    // Then subtract outer products, iterating levels high to low
    for level_k in (1..m).rev() {
        for i in 0..level_k {
            let j = level_k - 1 - i;
            let sign = if i % 2 == 0 { -1.0 } else { 1.0 };
            // prev[level_k] -= sign * outer(prev[i], seg[j])
            outer_accumulate_scaled(prev.level(i), seg.level(j), sign, prev.level_mut(level_k));
        }
    }
}
```

This does one pass (no clone, no negate). The sign logic accounts for the
negated even levels of S(-h). Need to verify the math carefully — the
high-to-low iteration order is critical because prev[i] at lower levels
must be finalized before being used for higher levels.

**Fix B — tensor_multiply_adjoint_into** (`src/algebra.rs`):

Add an in-place variant that writes into pre-allocated buffers:

```rust
pub fn tensor_multiply_adjoint_into(
    dresult: &LevelList, a: &LevelList, b: &LevelList,
    da: &mut LevelList, db: &mut LevelList,
) {
    da.copy_from(dresult);
    db.copy_from(dresult);
    for level_k in 1..m {
        for i in 0..level_k {
            let j = level_k - 1 - i;
            // ... same matvec/vecmat logic, writing into da/db
        }
    }
}
```

Then in `sigbackprop_core`, pre-allocate `da` and `db` once outside the loop.

**Expected impact**: 15-25% improvement on sigbackprop, potentially reaching
parity with iisignature (1.0x).

### 2. Rayon threshold tuning for boundary batch sizes

**Root cause**: B=50 with d=3,m=4,n=50 gives work=294K, just above the 250K
threshold, so rayon is used but doesn't fully amortize its overhead.

**Fix**: Raise threshold to ~500K, or use an adaptive approach that considers
the number of available cores:

```rust
const PARALLEL_THRESHOLD: usize = 500_000;
// Also require minimum batch size
if batch_size >= 16 && total_work > PARALLEL_THRESHOLD { ... }
```

The minimum batch_size check ensures rayon is never used when there aren't
enough items to distribute across cores.

**Expected impact**: Fix the 0.56x regression at B=50. May slightly delay
the parallelism crossover for larger batches (B=64 instead of B=32).

### 3. logsig BCH gap (NOT fixable without JIT)

The 0.19-0.86x gap on logsig for small configs is caused by iisignature's
JIT-compiled BCH generating native x86 machine code. This is not fixable
without adding JIT compilation (cranelift or similar), which was ruled out
in v1 due to complexity vs absolute gap (<0.05ms per call).

If this becomes a priority, the approach would be:
- Add `cranelift-jit` dependency
- During `prepare()`, compile the BCH FMA ops into native code
- Execute via function pointer instead of interpreter loop
- Would likely achieve parity with iisignature's JIT

## Priority

| Fix | Effort | Impact |
|-----|--------|--------|
| A: Fused unconcatenate | Medium | sigbackprop 0.7x → ~1.0x |
| B: adjoint_into | Small | sigbackprop +5-10% on top of A |
| Rayon threshold | Trivial | Fix B=50 regression |
