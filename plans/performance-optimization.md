# Performance Optimization Plan: sig-rust vs iisignature

**Date**: 2026-04-08
**Baseline**: sig-rust 0.1.0 vs iisignature 0.24 (C++)
**Hardware**: AMD Ryzen 7 5800X (8 cores / 16 threads), AVX2+FMA
**Compiler**: Rust stable, `target-cpu=native`, LTO=fat, codegen-units=1

## Current Performance Summary

### Single-path signature (sig)

Per-segment cost (amortized over path length, microseconds):

| Config    | Rust (us/seg) | C++ (us/seg) | Rust/C++ |
|-----------|---------------|--------------|----------|
| d=2, m=5  | 0.19          | 0.22         | 0.85x    |
| d=3, m=4  | 0.20          | 0.24         | 0.85x    |
| d=5, m=3  | 0.17          | 0.22         | 0.75x    |
| d=2, m=8  | 1.00          | 1.47         | 0.68x    |
| d=3, m=5  | 0.55          | 0.67         | 0.82x    |
| d=10, m=3 | 0.67          | 1.38         | 0.48x    |
| d=20, m=2 | 0.20          | 0.44         | 0.47x    |

**Status**: Already 15-50% faster than C++ per segment. Higher dimensions and
depths show the biggest wins (2x at d=20,m=2). The compiler auto-vectorizes
with AVX2 (1,608 `<4 x double>` ops in LLVM IR). No BLAS or manual SIMD needed.

Full benchmark ratios (including Python overhead, allocation, displacement
computation) show ~1.1-1.3x on medium/long paths, confirming the per-segment
advantage is partially masked by fixed costs.

### Single-path log signature (logsig)

| Config    | Rust best | C++     | Ratio | Notes                        |
|-----------|-----------|---------|-------|------------------------------|
| d=2, m=3  | 0.002ms   | 0.001ms | 2.55x | C++ uses JIT-compiled BCH    |
| d=2, m=4  | 0.004ms   | 0.001ms | 3.26x | C++ uses JIT-compiled BCH    |
| d=2, m=5  | 0.009ms   | 0.003ms | 2.83x | C++ JIT; Rust S wins over BCH|
| d=2, m=6  | 0.015ms   | 0.008ms | 1.94x | Rust S-method                |
| d=2, m=7  | 0.030ms   | 0.029ms | 1.05x | Both use S-method, near parity|
| d=2, m=8  | 0.053ms   | 0.089ms | 0.60x | Rust S-method wins           |
| d=3, m=3  | 0.004ms   | 0.002ms | 2.30x | C++ JIT BCH                  |
| d=3, m=4  | 0.012ms   | 0.005ms | 2.48x | C++ JIT BCH                  |
| d=3, m=5  | 0.029ms   | 0.035ms | 0.81x | Rust S-method wins           |
| d=5, m=3  | 0.009ms   | 0.006ms | 1.59x | Rust S-method                |
| d=5, m=4  | 0.038ms   | 0.042ms | 0.90x | Rust wins                    |
| d=10, m=2 | 0.005ms   | 0.006ms | 0.75x | Rust wins                    |

**Key insight**: The entire logsig gap comes from iisignature's JIT-compiled BCH
method for small configs (d<=4, m<=4). For configs where both libraries use
S-method (d=2,m=7+; d=3,m=5+; d=5,m=4+), Rust is already faster.

The tensor_log + Lyndon projection adds only 1-8% overhead on top of the sig
computation. The S-method is bottlenecked on sig, not on log/project.

### Single-path backpropagation

| Operation       | Config    | Rust    | C++     | Ratio |
|-----------------|-----------|---------|---------|-------|
| sigbackprop     | d=2, m=5  | 0.8us/s | 0.7us/s | 1.14x |
| sigbackprop     | d=3, m=4  | 1.0us/s | 0.9us/s | 1.15x |
| sigbackprop     | d=5, m=3  | 1.1us/s | 0.9us/s | 1.28x |
| logsigbackprop  | d=3, m=4  | 0.148ms | 0.130ms | 1.14x |

**Root cause**: Rust stores all n intermediate prefix signatures during the
forward pass (O(n * siglength) memory), while iisignature uses a reversibility
trick that requires only O(siglength) memory. At n=100, d=3, m=4, this is
93 KB vs 2.8 KB. The allocation pressure and cache misses explain the 14-28%
slowdown.

sigbackprop costs 5-7x the forward sig computation. Breakdown:
- Forward fold (computing prefix sigs): ~1x sig cost
- Backward fold (tensor_multiply_adjoint chain): ~2x sig cost
- seg_of_segment_adjoint_batch: ~2x sig cost
- Allocation of intermediate storage: overhead

### Batched operations (rayon parallelism)

| Operation  | Batch | Config     | Rust     | C++ serial | Ratio |
|------------|-------|------------|----------|------------|-------|
| sig        | B=1   | d=3,m=4    | 0.056ms  | 0.012ms    | 4.63x |
| sig        | B=4   | d=3,m=4    | 0.339ms  | 0.048ms    | 7.04x |
| sig        | B=8   | d=3,m=4    | 0.580ms  | 0.098ms    | 5.91x |
| sig        | B=32  | d=3,m=4    | 0.955ms  | 0.387ms    | 2.47x |
| sig        | B=64  | d=3,m=4    | 1.149ms  | 0.826ms    | 1.39x |
| sig        | B=200 | d=3,m=4    | 1.470ms  | 2.410ms    | 0.61x |
| logsig     | B=200 | d=3,m=4    | 1.169ms  | 0.996ms    | 1.17x |
| sigbackprop| B=200 | d=3,m=4    | 3.074ms  | 8.944ms    | 0.34x |

**Root cause**: Rayon thread pool wakeup and work distribution adds ~0.5-0.9ms
fixed overhead regardless of batch size. For small batches where each item
takes <0.02ms, this overhead dominates. The crossover point where parallelism
pays off is around B=32-64 for sig, later for logsig.

### Transforms (sigscale, sigjoin)

| Operation | Config   | Rust    | C++     | Ratio |
|-----------|----------|---------|---------|-------|
| sigscale  | d=2, m=5 | 0.001ms | 0.001ms | 0.79x |
| sigscale  | d=3, m=5 | 0.002ms | 0.002ms | 1.17x |
| sigjoin   | d=3, m=4 | 0.001ms | 0.001ms | 1.37x |
| sigjoin   | d=5, m=3 | 0.001ms | 0.001ms | 0.69x |

**Status**: Within noise at these timescales. Sub-microsecond operations where
Python call overhead dominates.

---

## Decisions Made

1. **No JIT compilation**: The JIT advantage is confined to small configs where
   absolute times are <0.05ms. Complexity not justified.
2. **No BLAS dependency**: The compiler's AVX2 auto-vectorization already beats
   iisignature's C++ loops by 15-50%. BLAS call overhead (~2us) exceeds the
   entire operation time for typical tensor sizes (d^k where d=2-5, k=1-5).
3. **No manual SIMD**: 1,608 `<4 x double>` vector operations confirmed in
   LLVM IR. Auto-vectorization is working.
4. **Safe Rust only**: No `unsafe` blocks. Performance wins come from
   algorithmic improvements, not low-level tricks.
5. **Implement reversibility trick**: Eliminates O(n * siglength) allocation
   in sigbackprop. iisignature proves this is faster in practice.
6. **Fix rayon heuristics**: Sequential fallback for small batches.

---

## Optimization Plan

### Phase 1: Rayon Batch Heuristics (Expected: fix 4-7x regression on small batches)

**Problem**: Rayon's thread pool wakeup costs ~0.5-0.9ms. For batch_size=1,
this turns a 0.012ms operation into 0.056ms (4.6x slower).

**Changes in `src/python.rs`**:

1. Add a cost heuristic function that estimates per-item work in microseconds
   based on `(d, m, n)`:
   ```
   estimated_us = n_segments * sum(d^k for k in 1..=m) * COST_PER_ELEM
   ```
   where `COST_PER_ELEM` is calibrated from benchmarks (~0.5ns for sig,
   ~1ns for backprop).

2. Compute `total_work_us = batch_size * estimated_us`. If below a threshold
   (~500us, i.e., the rayon overhead), run sequentially.

3. Apply to all batched operations: `py_sig`, `py_logsig`, `py_sigbackprop`,
   `py_logsigbackprop`, `py_sigcombine`.

**Fallback rule**: If batch_size <= 1 or total estimated work < 500us, always
run sequentially regardless of heuristic.

**Expected impact**:
- B=1: 4.6x slower -> 1.0x (eliminate rayon overhead entirely)
- B=4: 7.0x slower -> ~1.0x
- B=8: 5.9x slower -> ~1.0x
- B=32+: unchanged (rayon still used)

### Phase 2: Sigbackprop Reversibility Trick (Expected: 15-28% faster backprop)

**Problem**: Current `sigbackprop` stores all n intermediate prefix signatures
in a Vec. This causes O(n * siglength) allocation and poor cache behavior.

**Algorithm change in `src/backprop.rs`**:

Replace the current forward-store + backward-read pattern with:

```
Forward pass:
  1. Compute all segment signatures seg_sigs[0..n-1] (keep these, O(n * d) dominant)
  2. Compute only the FINAL prefix signature via left-fold (no intermediates stored)

Backward pass:
  For i = (n-1) down to 1:
    1. Recover sig_{i-1} from sig_i:
       sig_{i-1} = tensor_multiply(sig_i, invert_segment(seg_sigs[i]))
       where invert_segment just negates odd levels (zero-cost)
    2. Compute adjoint: (d_sig_{i-1}, d_seg_i) = tensor_multiply_adjoint(d_sig_i, sig_{i-1}, seg_sigs[i])
    3. Accumulate d_seg_i
```

**New helper function**: `invert_segment_sig(seg: &LevelList) -> LevelList`
that negates odd levels (levels 1, 3, 5, ...). This is O(siglength) with no
multiplication.

**Memory savings**:
- d=3, m=4, n=100: 93 KB -> 2.8 KB (33x reduction)
- d=3, m=4, n=1000: 937 KB -> 2.8 KB (333x reduction)

**Compute tradeoff**: Adds one tensor_multiply per segment (to recover sig_{i-1}),
but eliminates Vec<LevelList> allocation, reduces cache pressure, and avoids
the post-loop conversion to batched Array2 format.

The same approach should be applied to `logsigbackprop_s_method` which has
identical forward-store pattern.

**Expected impact**: 15-28% speedup on sigbackprop, bringing it from 1.15-1.28x
slower than C++ to roughly parity or faster.

### Phase 3: Reduce Allocations in Hot Paths (Expected: 5-15% across all ops)

Several hot-path functions allocate unnecessarily:

#### 3a. `tensor_multiply` in `algebra.rs`

Current: `tensor_multiply` clones `a` into `result`, then mutates.
The `tensor_multiply_into` variant already exists and avoids this.

**Change**: Audit all call sites. Where possible, replace `tensor_multiply`
calls with `tensor_multiply_into` using pre-allocated buffers. Key call sites:

- `backprop_fold` (line 101-104 in backprop.rs): Called n-1 times in the
  backward pass. Each call allocates a new LevelList.
- `sig_levels` already uses `tensor_multiply_into` with double buffering. Good.
- `sigcombine`: One-shot, allocation is fine.

For `backprop_fold`, after implementing Phase 2 (reversibility), the backward
loop can reuse a single pair of buffers via double-buffering, same pattern as
`sig_levels`.

#### 3b. `sig_of_segment_from_slice` per-segment allocation

Current: Each call to `sig_of_segment_from_slice` allocates a new Vec for the
flat data. In `sigbackprop_core`, this is called n-1 times.

**Change**: Add `sig_of_segment_into(displacement, depth, result: &mut LevelList)`
that writes into a pre-allocated buffer. Use in the backward loop where
segment sigs are computed on-the-fly.

Alternatively, since segment sigs are still needed for the reversibility trick,
keep the current allocation pattern but ensure they are computed once and reused.

#### 3c. `tensor_log` workspace allocation

Current: `tensor_log` allocates two LevelList workspaces (`s` and `t`) on every
call. For logsig, this is called once per path, so impact is small (<1%).

**Change**: Accept optional pre-allocated workspaces. Low priority.

#### 3d. `split_signature` / `concat_levels` round-trips

Current: `sigcombine` does split -> multiply -> concat, allocating twice.
`sigbackprop` splits the derivative into a LevelList.

**Change**: These are one-shot operations and the allocation cost is negligible
relative to the tensor_multiply. No change needed.

### Phase 4: BCH Interpreter Optimization (Expected: 20-40% faster BCH)

The Rust BCH interpreter is 2.5-3x slower than iisignature's JIT for small
configs. While the absolute gap is small (<0.05ms), improving BCH helps the
configs where it's selected (d<=3, m<=3; d=2, m<=4).

#### 4a. Sort FMA ops by destination

iisignature sorts compiled ops by `lhs_offset` then `rhs_offset` for cache
locality. The comment in iisignature says "this sorting helps a lot."

**Change in `src/bch.rs`**: After compilation, sort `ops` by `dest` then by
`src_a`. This groups writes to the same destination together, enabling the
CPU to keep the destination in a register and avoid repeated load/store.

```rust
compiled.ops.sort_by_key(|op| (op.dest, op.src_a));
```

#### 4b. Batch writes to same destination

After sorting, consecutive ops writing to the same `dest` can be fused:
load dest once, accumulate all products, store once. This reduces memory
traffic.

**Change**: In `execute_compiled`, detect runs of ops with the same `dest`
and accumulate into a local variable:

```rust
let mut i = 0;
while i < ops.len() {
    let dest = ops[i].dest as usize;
    let mut acc = values[dest];
    while i < ops.len() && ops[i].dest as usize == dest {
        acc += ops[i].coeff * values[ops[i].src_a as usize] * values[ops[i].src_b as usize];
        i += 1;
    }
    values[dest] = acc;
}
```

#### 4c. Tune BCH selection heuristic

Current heuristic: `d >= 2 && m >= 2 && ((d <= 3 && m <= 3) || (d == 2 && m <= 4))`

Measured BCH vs S-method ratios:
- d=2, m=3: BCH=0.35x of S (BCH wins)
- d=2, m=4: BCH=0.72x of S (BCH wins)
- d=2, m=5: BCH=1.67x of S (S wins)
- d=3, m=3: BCH=0.66x of S (BCH wins)
- d=3, m=4: BCH=1.99x of S (S wins)
- d=5, m=3: BCH=1.59x of S (S wins)

**Change**: After BCH optimization, re-measure and adjust. The current heuristic
is roughly correct but may need tightening after Phase 4a/4b changes.

### Phase 5: Optimize tensor_multiply Inner Loop (Expected: 5-10% on sig)

#### 5a. Iterate levels high-to-low

iisignature's `concatenateWith` iterates levels from highest to lowest. This
avoids the `wrapping_sub` + bounds check pattern in the current Rust code.

Current loop pattern:
```rust
for level_k in 0..m {
    for i in 0..=level_k {
        let j = level_k.wrapping_sub(1).wrapping_sub(i);
        if j >= m { continue; }  // branch on every iteration
        outer_accumulate(a.level(i), b.level(j), result.level_mut(level_k));
    }
}
```

**Change**: Restructure to eliminate the bounds check:
```rust
for level_k in 0..m {
    // i + j = level_k - 1, so j = level_k - 1 - i
    // Valid when level_k >= 1 and i < level_k
    if level_k == 0 { continue; }
    for i in 0..level_k {
        let j = level_k - 1 - i;
        outer_accumulate(a.level(i), b.level(j), result.level_mut(level_k));
    }
}
```

This eliminates the `wrapping_sub` and the `if j >= m { continue }` branch.

#### 5b. Pre-split level pointers

The `level()` method does two array lookups (offsets[k] and offsets[k+1]) per
call. In the inner loop, this happens O(m^2) times.

**Change**: Pre-compute level slice pointers before the loop:
```rust
let a_levels: Vec<&[f64]> = (0..m).map(|k| a.level(k)).collect();
let b_levels: Vec<&[f64]> = (0..m).map(|k| b.level(k)).collect();
```

Then use `a_levels[i]` and `b_levels[j]` in the inner loop. This trades a
small Vec allocation for eliminating 2*m^2 bounds-checked array lookups.

### Phase 6: In-place logsig S-method (Expected: <5% on logsig)

Currently `logsig_s_method` computes sig, then tensor_log, then projects.
Each step produces a new allocation.

**Change**: Modify `tensor_log` to accept an output buffer and write in-place.
The projection step already writes into a pre-allocated `out` vec. Low priority
since tensor_log+project is only 1-8% of logsig time.

---

## Priority Order and Expected Cumulative Impact

| Phase | Description                  | Effort | Impact Area        | Expected Gain       |
|-------|------------------------------|--------|--------------------|---------------------|
| 1     | Rayon batch heuristics       | Small  | All batched ops    | 4-7x on small batch |
| 2     | Sigbackprop reversibility    | Medium | sigbackprop, logsigbackprop | 15-28% faster |
| 3     | Allocation reduction         | Medium | All single-path ops| 5-15% faster        |
| 4     | BCH interpreter optimization | Small  | logsig small configs| 20-40% faster BCH  |
| 5     | tensor_multiply inner loop   | Small  | sig, logsig, backprop| 5-10% faster       |
| 6     | In-place logsig              | Small  | logsig             | <5% faster          |

### Projected final performance vs iisignature (C++)

**Single-path:**

| Operation     | Current best | After optimization | Target    |
|---------------|--------------|--------------------|-----------|
| sig           | 0.63-1.31x   | 0.55-1.1x          | >= 1.0x everywhere |
| logsig (BCH)  | 2.3-3.3x     | 1.5-2.0x           | <= 2.0x   |
| logsig (S)    | 0.6-1.9x     | 0.5-1.5x           | >= 0.8x everywhere |
| sigbackprop   | 0.76-0.90x   | 0.65-0.80x         | >= 1.0x everywhere |
| logsigbackprop| 0.74-0.90x   | 0.65-0.80x         | >= 1.0x everywhere |
| sigscale      | 0.76-1.17x   | ~same               | >= 0.8x   |
| sigjoin       | 0.69-1.37x   | ~same               | >= 0.8x   |

Note: ratios < 1.0 mean Rust is faster. The sig forward pass is already
faster than C++ in most configs. The main gaps to close are:
1. Backprop (Phases 2+3)
2. Logsig BCH for small configs (Phase 4)
3. Logsig S-method for medium configs (Phases 3+5)

**Batched (where parallelism matters):**

| Operation     | Current (B=10) | After Phase 1 | After all |
|---------------|----------------|---------------|-----------|
| sig           | 0.16x          | ~1.0x          | ~1.2x    |
| logsig        | 0.07x          | ~0.7x          | ~0.9x    |
| sigbackprop   | 0.45x          | ~0.7x          | ~1.0x    |

| Operation     | Current (B=100) | After Phase 1 | After all |
|---------------|-----------------|---------------|-----------|
| sig           | 0.94x           | ~1.0x          | ~1.2x    |
| logsig        | 0.45x           | ~0.5x          | ~0.6x    |
| sigbackprop   | 1.95x           | ~2.0x          | ~2.5x    |

Note: For large batches (B=100+), rayon parallelism already provides good
speedups for sig and sigbackprop. The logsig gap at large batches is due to
iisignature's JIT BCH being 2-3x faster per item, which compounds.

**Realistic goals:**
- Base goal: Faster everywhere on single-path sig, sigbackprop, logsig (S-method
  configs), and transforms. Fix batch regression.
- Realistic goal: 1.5-2x faster than C++ on sig for medium/long paths. On par
  or faster on backprop. Logsig within 2x on BCH configs, faster on S configs.
- Stretch: 2-3x faster on sig for large dimensions. Batch throughput 3-5x
  faster than serial C++ on 100+ batches.

---

## Verification Plan

After each phase:
1. Run `cargo test` (correctness)
2. Run `cargo clippy --all-targets --all-features -- -D warnings` (lint)
3. Run `uv run pytest` (Python binding correctness)
4. Run `uv run python scripts/benchmark_full.py` (performance regression)
5. Run the micro-benchmark script from this analysis to verify per-segment
   improvement

Save before/after benchmark results for each phase.

---

## Appendix: What We Ruled Out

### JIT Compilation
iisignature generates x86 machine code at runtime for the BCH formula using
SSE2 `movsd`/`mulsd`/`addsd`. This gives 2.5-3x advantage over our interpreted
BCH for small configs (d<=4, m<=4). However:
- Absolute gap is <0.05ms per call
- For large configs where total time matters, both use S-method and Rust wins
- Adding `cranelift-jit` or hand-written asm is high complexity, low payoff
- The optimized interpreter (Phase 4) should close ~40% of the gap

### BLAS Integration
- BLAS call overhead (~2us) exceeds entire operation time for typical tensor
  sizes (d^k where d=2-5, k=1-5 -> 4-3125 elements)
- numpy BLAS `dger` at size 32x32: 10.9us. Our auto-vectorized loop does the
  same work in ~0.02us as part of a larger computation
- Only beneficial at d=20+, m=3+ where Rust is already 2x faster than C++
- Heavy dependency (system BLAS detection, C/Fortran linking)

### Manual SIMD (`std::simd`)
- Compiler already produces 1,608 `<4 x double>` (AVX2) vector operations
- Auto-vectorization confirmed via LLVM IR inspection
- Manual SIMD would at best match what LLVM generates, with maintenance cost
- Requires nightly Rust

### Binary Tree Reduction for sig
- Signatory (Kidger 2021) uses binary tree reduction: pairs of segments are
  combined in parallel, then pairs of pairs, etc. This gives O(log n) depth
  instead of O(n) sequential.
- Only beneficial with GPU or wide SIMD parallelism. On CPU with rayon, the
  thread overhead makes this worse for typical path lengths (n=10-1000).
- Could revisit for very long paths (n=10000+) with rayon, but low priority.

### Fused sig_of_segment + tensor_multiply
- Instead of computing seg_sig then multiplying into accumulator, fuse the
  operations: directly accumulate outer products of displacement into the
  running signature.
- Reduces one pass over the data but significantly complicates the code.
- The compiler may already achieve similar through inlining + LTO.
- Low expected gain (<5%) for high code complexity.
