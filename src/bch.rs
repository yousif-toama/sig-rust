use ndarray::{Array1, Array2};

use crate::lyndon::{lyndon_to_tensor, outer_flat};
use crate::types::{Depth, Dim};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A single nonzero structure constant: `[e_i, e_j]` contributes `coeff` to
/// basis element `dest` within the destination level.
#[derive(Debug, Clone)]
struct BracketEntry {
    dest: usize,
    coeff: f64,
}

/// Structure constants for one `(source_level_1, source_level_2)` pair.
#[derive(Debug, Clone)]
struct BracketBlock {
    n_src1: usize,
    n_src2: usize,
    /// Indexed by `i * n_src2 + j`; each entry is the sparse expansion
    /// of `[e_i, e_j]` in the destination level's Lyndon basis.
    entries: Vec<Vec<BracketEntry>>,
}

/// Uniform branchless FMA instruction:
/// `values[dest] += coeff * values[src_a] * values[src_b]`.
#[derive(Debug, Clone, Copy)]
struct FmaOp {
    dest: u16,
    src_a: u16,
    src_b: u16,
    coeff: f64,
}

/// Compiled BCH program for fast per-segment evaluation.
#[derive(Debug, Clone)]
struct CompiledBch {
    /// Uniform FMA instructions executed in order.
    ops: Vec<FmaOp>,
    /// Total size of the values array needed.
    num_values: usize,
    num_logsig: usize,
    num_disp: usize,
    /// Index of constant 1.0 in the values array.
    const_one_idx: usize,
    /// For each logsig coord, the temp index holding its correction (0 = none).
    output_temps: Vec<u16>,
}

/// Precomputed data for BCH log-signature computation.
#[derive(Debug, Clone)]
pub struct BchData {
    #[allow(dead_code)]
    depth: Depth,
    #[allow(dead_code)]
    bracket_tables: Vec<Vec<BracketBlock>>,
    #[allow(dead_code)]
    level_offsets: Vec<usize>,
    #[allow(dead_code)]
    level_sizes: Vec<usize>,
    total_size: usize,
    compiled: CompiledBch,
}

/// Operand reference for bracket expansion: either a single values index,
/// a sum/difference of two indices (for `logsig ± disp` at level 1),
/// or a known-zero value (e.g. `W_n` at level 1 for n >= 2).
#[derive(Clone, Copy)]
enum Operand {
    Single(u16),
    Sum(u16, u16),
    Zero,
}

// ---------------------------------------------------------------------------
// Precomputation
// ---------------------------------------------------------------------------

/// Build BCH data (structure constants + compiled program).
pub fn compute_bch_data(
    dim: Dim,
    depth: Depth,
    all_words: &[Vec<u8>],
    projection_matrices: &[Array2<f64>],
) -> BchData {
    let d = dim.value();
    let m = depth.value();

    let words_by_level: Vec<Vec<&Vec<u8>>> = (1..=m)
        .map(|k| all_words.iter().filter(|w| w.len() == k).collect())
        .collect();

    let level_sizes: Vec<usize> = words_by_level.iter().map(Vec::len).collect();
    let mut level_offsets = Vec::with_capacity(m + 1);
    level_offsets.push(0);
    for &sz in &level_sizes {
        level_offsets.push(level_offsets.last().expect("non-empty") + sz);
    }
    let total_size = *level_offsets.last().expect("non-empty");

    let tensors: Vec<Vec<Array1<f64>>> = words_by_level
        .iter()
        .map(|ws| ws.iter().map(|w| lyndon_to_tensor(w, d)).collect())
        .collect();

    let mut bracket_tables: Vec<Vec<BracketBlock>> = Vec::with_capacity(m.saturating_sub(1));
    for dest_level in 2..=m {
        bracket_tables.push(build_blocks_for_level(
            dest_level,
            m,
            &level_sizes,
            &tensors,
            projection_matrices,
        ));
    }

    let compiled = compile_bch(
        m,
        d,
        total_size,
        &level_offsets,
        &level_sizes,
        &bracket_tables,
    );

    BchData {
        depth,
        bracket_tables,
        level_offsets,
        level_sizes,
        total_size,
        compiled,
    }
}

/// Build all `BracketBlock`s whose destination is `dest_level`.
fn build_blocks_for_level(
    dest_level: usize,
    max_level: usize,
    level_sizes: &[usize],
    tensors: &[Vec<Array1<f64>>],
    proj_matrices: &[Array2<f64>],
) -> Vec<BracketBlock> {
    let proj = &proj_matrices[dest_level - 1];
    let mut blocks = Vec::new();

    for l1 in 1..dest_level {
        let l2 = dest_level - l1;
        if l2 > max_level {
            continue;
        }
        let n1 = level_sizes[l1 - 1];
        let n2 = level_sizes[l2 - 1];
        if n1 == 0 || n2 == 0 {
            continue;
        }

        let mut entries = Vec::with_capacity(n1 * n2);
        for i in 0..n1 {
            for j in 0..n2 {
                let t_i = &tensors[l1 - 1][i];
                let t_j = &tensors[l2 - 1][j];
                let comm = &outer_flat(t_i, t_j) - &outer_flat(t_j, t_i);
                let coords = proj.dot(&comm);
                let sparse: Vec<BracketEntry> = coords
                    .iter()
                    .enumerate()
                    .filter(|&(_, c)| c.abs() > 1e-14)
                    .map(|(k, &c)| BracketEntry { dest: k, coeff: c })
                    .collect();
                entries.push(sparse);
            }
        }
        blocks.push(BracketBlock {
            n_src1: n1,
            n_src2: n2,
            entries,
        });
    }
    blocks
}

// ---------------------------------------------------------------------------
// BCH program compiler
// ---------------------------------------------------------------------------

/// Bernoulli coefficients K_{2p} = B_{2p} / (2p)!.
const BERNOULLI_K: [f64; 4] = [1.0 / 12.0, -1.0 / 720.0, 1.0 / 30240.0, -1.0 / 1_209_600.0];

/// Compile the BCH formula into a flat list of uniform FMA instructions.
fn compile_bch(
    max_depth: usize,
    dim: usize,
    total: usize,
    level_offsets: &[usize],
    level_sizes: &[usize],
    bracket_tables: &[Vec<BracketBlock>],
) -> CompiledBch {
    let logsig_base = 0usize;
    let disp_base = total;
    let const_one = total + dim;
    let mut next_temp = total + dim + 1;
    let mut ops: Vec<FmaOp> = Vec::new();

    let mut output_temps = vec![0u16; total];
    for slot in &mut output_temps[level_offsets[1]..total] {
        *slot = alloc_temp(&mut next_temp);
    }

    let mut w_temps: Vec<Vec<u16>> = vec![vec![0u16; total]];

    for degree in 2..=max_depth {
        let mut wn = vec![0u16; total];
        for slot in &mut wn[level_offsets[1]..] {
            *slot = alloc_temp(&mut next_temp);
        }

        let tables = BracketTables {
            max_depth,
            level_offsets,
            level_sizes,
            blocks: bracket_tables,
        };

        if degree == 2 {
            // [x-h, x+h] = 2[x, h] → emit (1/degree) * [x, h_ext]
            emit_bracket_ops(
                &mut ops,
                &tables,
                |pos| Operand::Single((logsig_base + pos) as u16),
                |pos| {
                    if pos < dim {
                        Operand::Single((disp_base + pos) as u16)
                    } else {
                        Operand::Zero
                    }
                },
                1.0 / degree as f64,
                &wn,
            );
        } else {
            // [x-h, W] = [x, W] - [h_ext, W]
            let scale = 0.5 / degree as f64;
            emit_bracket_ops(
                &mut ops,
                &tables,
                |pos| Operand::Single((logsig_base + pos) as u16),
                |pos| operand_w_prev(pos, degree - 1, dim, logsig_base, disp_base, &w_temps),
                scale,
                &wn,
            );
            emit_bracket_ops(
                &mut ops,
                &tables,
                |pos| {
                    if pos < dim {
                        Operand::Single((disp_base + pos) as u16)
                    } else {
                        Operand::Zero
                    }
                },
                |pos| operand_w_prev(pos, degree - 1, dim, logsig_base, disp_base, &w_temps),
                -scale,
                &wn,
            );
        }

        let layout = ValueLayout {
            dim,
            total,
            logsig_base,
            disp_base,
            const_one: const_one as u16,
        };
        emit_bernoulli(
            &mut ops,
            degree,
            &tables,
            &layout,
            &w_temps,
            &wn,
            &mut next_temp,
        );

        // Accumulate W_degree into output corrections using const_one
        let co = const_one as u16;
        for r in level_offsets[1]..total {
            ops.push(FmaOp {
                dest: output_temps[r],
                src_a: wn[r],
                src_b: co,
                coeff: 1.0,
            });
        }

        w_temps.push(wn);
    }

    CompiledBch {
        ops,
        num_values: next_temp,
        num_logsig: total,
        num_disp: dim,
        const_one_idx: const_one,
        output_temps,
    }
}

fn alloc_temp(next: &mut usize) -> u16 {
    let t = *next;
    *next += 1;
    t as u16
}

/// Allocate temps for levels 2+ only. Level-1 slots are left as 0.
fn alloc_temp_vec_levels2(total: usize, level1_end: usize, next: &mut usize) -> Vec<u16> {
    let mut v = vec![0u16; total];
    for slot in &mut v[level1_end..] {
        *slot = alloc_temp(next);
    }
    v
}

/// `W_{degree}[pos]`: for `W_1`, `logsig+disp` at level 1, `logsig` elsewhere.
/// For `W_n` (n >= 2), `Zero` at level 1, temps at higher levels.
fn operand_w_prev(
    pos: usize,
    degree: usize,
    dim: usize,
    logsig_base: usize,
    disp_base: usize,
    w_temps: &[Vec<u16>],
) -> Operand {
    if degree == 1 {
        if pos < dim {
            Operand::Sum((logsig_base + pos) as u16, (disp_base + pos) as u16)
        } else {
            Operand::Single((logsig_base + pos) as u16)
        }
    } else if pos < dim {
        // W_n for n >= 2 has no level-1 content
        Operand::Zero
    } else {
        Operand::Single(w_temps[degree - 1][pos])
    }
}

/// Bracket structure tables needed for emission.
struct BracketTables<'a> {
    max_depth: usize,
    level_offsets: &'a [usize],
    level_sizes: &'a [usize],
    blocks: &'a [Vec<BracketBlock>],
}

/// Emit raw ops for the bracket `[left, right]` scaled by `scale`.
fn emit_bracket_ops(
    ops: &mut Vec<FmaOp>,
    tables: &BracketTables<'_>,
    left_fn: impl Fn(usize) -> Operand,
    right_fn: impl Fn(usize) -> Operand,
    scale: f64,
    dest_temps: &[u16],
) {
    for dest_level in 2..=tables.max_depth {
        let dest_base = tables.level_offsets[dest_level - 1];
        let blocks = &tables.blocks[dest_level - 2];
        let mut block_idx = 0;

        for l1 in 1..dest_level {
            let l2 = dest_level - l1;
            if l2 > tables.max_depth
                || tables.level_sizes[l1 - 1] == 0
                || tables.level_sizes[l2 - 1] == 0
            {
                continue;
            }
            let blk = &blocks[block_idx];
            block_idx += 1;
            let ctx = BlockContext {
                l1_off: tables.level_offsets[l1 - 1],
                l2_off: tables.level_offsets[l2 - 1],
                dest_base,
                scale,
                dest_temps,
            };
            emit_block_ops(ops, blk, &ctx, &left_fn, &right_fn);
        }
    }
}

/// Parameters for bracket block emission.
struct BlockContext<'a> {
    l1_off: usize,
    l2_off: usize,
    dest_base: usize,
    scale: f64,
    dest_temps: &'a [u16],
}

/// Emit raw ops for one bracket block.
fn emit_block_ops(
    ops: &mut Vec<FmaOp>,
    blk: &BracketBlock,
    ctx: &BlockContext<'_>,
    left_fn: &impl Fn(usize) -> Operand,
    right_fn: &impl Fn(usize) -> Operand,
) {
    for i in 0..blk.n_src1 {
        let left = left_fn(ctx.l1_off + i);
        for j in 0..blk.n_src2 {
            let right = right_fn(ctx.l2_off + j);
            for e in &blk.entries[i * blk.n_src2 + j] {
                let dest = ctx.dest_temps[ctx.dest_base + e.dest];
                emit_product(ops, dest, e.coeff * ctx.scale, left, right);
            }
        }
    }
}

/// Expand an operand into `(sign, index)` pairs.
/// `Single(a)` -> `[(1.0, a)]`, `Sum(a,b)` -> `[(1.0, a), (1.0, b)]`,
/// `Diff(a,b)` -> `[(1.0, a), (-1.0, b)]`.
fn expand_operand(op: Operand) -> [(f64, u16); 2] {
    match op {
        Operand::Zero => [(0.0, 0), (0.0, 0)],
        Operand::Single(a) => [(1.0, a), (0.0, 0)],
        Operand::Sum(a, b) => [(1.0, a), (1.0, b)],
    }
}

/// Emit FMA op(s) for `dest += coeff * left * right`, expanding Sum operands.
fn emit_product(ops: &mut Vec<FmaOp>, dest: u16, coeff: f64, left: Operand, right: Operand) {
    if coeff.abs() < 1e-18 {
        return;
    }
    let lefts = expand_operand(left);
    let rights = expand_operand(right);
    for &(ls, la) in &lefts {
        if ls == 0.0 {
            continue;
        }
        for &(rs, rb) in &rights {
            if rs == 0.0 {
                continue;
            }
            ops.push(FmaOp {
                dest,
                coeff: coeff * ls * rs,
                src_a: la,
                src_b: rb,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Bernoulli corrections compiler
// ---------------------------------------------------------------------------

/// Layout indices for the values array used during compilation.
struct ValueLayout {
    dim: usize,
    total: usize,
    logsig_base: usize,
    disp_base: usize,
    const_one: u16,
}

/// Emit Bernoulli correction ops for degree >= 4.
fn emit_bernoulli(
    ops: &mut Vec<FmaOp>,
    degree: usize,
    tables: &BracketTables<'_>,
    layout: &ValueLayout,
    w_temps: &[Vec<u16>],
    wn_dest: &[u16],
    next_temp: &mut usize,
) {
    let max_p = (degree - 2) / 2;
    if max_p == 0 {
        return;
    }

    let dim = layout.dim;
    let total = layout.total;
    let logsig_base = layout.logsig_base;
    let disp_base = layout.disp_base;

    for p in 1..=max_p {
        if p > BERNOULLI_K.len() {
            break;
        }
        let k2p = BERNOULLI_K[p - 1];
        let inv_deg = 1.0 / degree as f64;

        let level1_end = tables.level_offsets[1];
        for comp in compositions(degree - 1, 2 * p) {
            // Optimization 1: skip dead compositions.
            // The innermost bracket is [W_{r_last}, W_1].
            // When r_last == 1 this is [W_1, W_1] = 0 by antisymmetry.
            if *comp.last().expect("non-empty") == 1 {
                continue;
            }

            let mut current_temps = alloc_temp_vec_levels2(total, level1_end, next_temp);

            // Innermost: [W_{r_last}, W_1] = [W_r, x] + [W_r, h_ext]
            let r_last = comp[comp.len() - 1];
            emit_bracket_ops(
                ops,
                tables,
                |pos| operand_w_prev(pos, r_last, dim, logsig_base, disp_base, w_temps),
                |pos| Operand::Single((logsig_base + pos) as u16),
                1.0,
                &current_temps,
            );
            emit_bracket_ops(
                ops,
                tables,
                |pos| operand_w_prev(pos, r_last, dim, logsig_base, disp_base, w_temps),
                |pos| {
                    if pos < dim {
                        Operand::Single((disp_base + pos) as u16)
                    } else {
                        Operand::Zero
                    }
                },
                1.0,
                &current_temps,
            );

            for idx in (0..comp.len() - 1).rev() {
                let r_k = comp[idx];
                let new_inter = alloc_temp_vec_levels2(total, level1_end, next_temp);

                emit_bracket_ops(
                    ops,
                    tables,
                    |pos| operand_w_prev(pos, r_k, dim, logsig_base, disp_base, w_temps),
                    |pos| {
                        if pos < dim {
                            Operand::Zero
                        } else {
                            Operand::Single(current_temps[pos])
                        }
                    },
                    1.0,
                    &new_inter,
                );
                current_temps = new_inter;
            }

            let final_scale = k2p * inv_deg;
            for r in tables.level_offsets[1]..total {
                ops.push(FmaOp {
                    dest: wn_dest[r],
                    src_a: current_temps[r],
                    src_b: layout.const_one,
                    coeff: final_scale,
                });
            }
        }
    }
}

/// All ordered compositions of `n` into exactly `k` positive parts.
fn compositions(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return if n == 0 { vec![vec![]] } else { vec![] };
    }
    if k > n {
        return vec![];
    }
    if k == 1 {
        return vec![vec![n]];
    }
    let mut result = Vec::new();
    for first in 1..=(n - k + 1) {
        for mut rest in compositions(n - first, k - 1) {
            rest.insert(0, first);
            result.push(rest);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Compiled BCH executor
// ---------------------------------------------------------------------------

/// Execute the compiled BCH program for one segment.
///
/// Batches consecutive ops that write to the same destination into a local
/// accumulator to reduce memory traffic. Falls back to single-op execution
/// when an op reads from its own destination (self-referencing).
fn execute_compiled(logsig: &mut [f64], disp: &[f64], prog: &CompiledBch, values: &mut [f64]) {
    values[..prog.num_logsig].copy_from_slice(logsig);
    values[prog.num_logsig..prog.num_logsig + prog.num_disp].copy_from_slice(disp);
    values[prog.const_one_idx] = 1.0;
    values[prog.const_one_idx + 1..].fill(0.0);

    execute_ops(&prog.ops, values);

    for (r, &temp_idx) in prog.output_temps.iter().enumerate() {
        if temp_idx != 0 {
            logsig[r] += values[temp_idx as usize];
        }
    }
    for i in 0..prog.num_disp {
        logsig[i] += disp[i];
    }
}

/// Execute FMA ops with destination batching for reduced memory traffic.
fn execute_ops(ops: &[FmaOp], values: &mut [f64]) {
    let mut i = 0;
    while i < ops.len() {
        let dest = ops[i].dest;
        let dest_idx = dest as usize;
        let mut acc = values[dest_idx];
        // Batch consecutive ops to the same dest when safe (no self-reference)
        while i < ops.len() && ops[i].dest == dest && ops[i].src_a != dest && ops[i].src_b != dest {
            let op = &ops[i];
            acc += op.coeff * values[op.src_a as usize] * values[op.src_b as usize];
            i += 1;
        }
        values[dest_idx] = acc;
        // Handle self-referencing op if that's what stopped the batch
        if i < ops.len() && ops[i].dest == dest {
            let op = &ops[i];
            values[dest_idx] += op.coeff * values[op.src_a as usize] * values[op.src_b as usize];
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the log signature using the compiled BCH method.
pub fn logsig_bch(path: &Array2<f64>, bch: &BchData, dim: Dim, _depth: Depth) -> Array1<f64> {
    let n_pts = path.nrows();
    let d = dim.value();
    let total = bch.total_size;

    if n_pts < 2 {
        return Array1::zeros(total);
    }

    let prog = &bch.compiled;
    let mut values = vec![0.0; prog.num_values];
    let mut logsig = vec![0.0; total];

    // First displacement -> level-1 coordinates only
    for j in 0..d {
        logsig[j] = path[[1, j]] - path[[0, j]];
    }

    // Combine subsequent displacements via compiled BCH
    let mut disp = vec![0.0; d];
    for seg in 2..n_pts {
        for j in 0..d {
            disp[j] = path[[seg, j]] - path[[seg - 1, j]];
        }
        execute_compiled(&mut logsig, &disp, prog, &mut values);
    }

    Array1::from_vec(logsig)
}

/// Compute log-signature backpropagation using the BCH method.
pub fn logsig_bch_backprop(
    deriv: &Array1<f64>,
    path: &Array2<f64>,
    bch: &BchData,
    dim: Dim,
    _depth: Depth,
) -> Array2<f64> {
    let n_pts = path.nrows();
    let d = dim.value();
    let total = bch.total_size;

    if n_pts < 2 {
        return Array2::zeros((n_pts, d));
    }

    let prog = &bch.compiled;
    let num_segs = n_pts - 1;
    let displacements: Vec<Vec<f64>> = (0..num_segs)
        .map(|i| (0..d).map(|j| path[[i + 1, j]] - path[[i, j]]).collect())
        .collect();

    // Forward pass: store intermediate log-signatures
    let mut values = vec![0.0; prog.num_values];
    let mut intermediates: Vec<Vec<f64>> = Vec::with_capacity(num_segs);
    let mut logsig = vec![0.0; total];
    logsig[..d].copy_from_slice(&displacements[0][..d]);
    intermediates.push(logsig.clone());

    for disp in &displacements[1..] {
        execute_compiled(&mut logsig, disp, prog, &mut values);
        intermediates.push(logsig.clone());
    }

    // Backward pass: use AD on the compiled program
    let mut dlogsig = deriv.as_slice().expect("contiguous").to_vec();
    let mut dpath = Array2::zeros((n_pts, d));

    for seg in (1..num_segs).rev() {
        let x_before = &intermediates[seg - 1];
        let disp = &displacements[seg];
        let mut dx = vec![0.0; total];
        let mut dh = vec![0.0; d];
        adjoint_compiled(
            &dlogsig,
            x_before,
            disp,
            prog,
            &mut values,
            &mut dx,
            &mut dh,
        );
        for j in 0..d {
            dpath[[seg, j]] -= dh[j];
            dpath[[seg + 1, j]] += dh[j];
        }
        dlogsig = dx;
    }

    // First segment: logsig = displacement directly
    for j in 0..d {
        dpath[[0, j]] -= dlogsig[j];
        dpath[[1, j]] += dlogsig[j];
    }

    dpath
}

/// Adjoint of `execute_compiled`.
fn adjoint_compiled(
    dout: &[f64],
    x_before: &[f64],
    disp: &[f64],
    prog: &CompiledBch,
    values: &mut [f64],
    dx: &mut [f64],
    dh: &mut [f64],
) {
    // Recompute forward values
    values[..prog.num_logsig].copy_from_slice(x_before);
    values[prog.num_logsig..prog.num_logsig + prog.num_disp].copy_from_slice(disp);
    values[prog.const_one_idx] = 1.0;
    values[prog.const_one_idx + 1..].fill(0.0);
    execute_ops(&prog.ops, values);

    // Backward through output: dout flows to corrections and level-1 addition
    let mut dvalues = vec![0.0; prog.num_values];
    for (r, &temp_idx) in prog.output_temps.iter().enumerate() {
        if temp_idx != 0 {
            dvalues[temp_idx as usize] += dout[r];
        }
    }
    // Level-1 addition: logsig[i] += disp[i] → dx[i] += dout[i], dh[i] += dout[i]
    for i in 0..prog.num_disp {
        dvalues[i] += dout[i];
        dh[i] += dout[i];
    }
    // Higher-level identity: logsig[r] is unchanged → dx[r] += dout[r]
    for r in prog.num_disp..prog.num_logsig {
        dvalues[r] += dout[r];
    }

    // Backward through FMA operations (reverse order, branchless)
    // Each op: values[dest] += coeff * values[src_a] * values[src_b]
    for op in prog.ops.iter().rev() {
        let d_dest = dvalues[op.dest as usize];
        if d_dest == 0.0 {
            continue;
        }
        let grad = op.coeff * d_dest;
        dvalues[op.src_a as usize] += grad * values[op.src_b as usize];
        dvalues[op.src_b as usize] += grad * values[op.src_a as usize];
    }

    // Extract dx (from logsig_orig section) and dh (from disp section)
    for i in 0..prog.num_logsig {
        dx[i] += dvalues[i];
    }
    for i in 0..prog.num_disp {
        dh[i] += dvalues[prog.num_logsig + i];
    }
}

// ---------------------------------------------------------------------------
// Lie bracket (kept for tests)
// ---------------------------------------------------------------------------

/// Compute `[a, b]` in the Lyndon basis and ADD to `result`.
#[cfg(test)]
fn lie_bracket(a: &[f64], b: &[f64], bch: &BchData, result: &mut [f64]) {
    let m = bch.depth.value();
    for dest_level in 2..=m {
        let dest_off = bch.level_offsets[dest_level - 1];
        let blocks = &bch.bracket_tables[dest_level - 2];
        let mut block_idx = 0;
        for l1 in 1..dest_level {
            let l2 = dest_level - l1;
            if l2 > m || bch.level_sizes[l1 - 1] == 0 || bch.level_sizes[l2 - 1] == 0 {
                continue;
            }
            let blk = &blocks[block_idx];
            block_idx += 1;
            let a_off = bch.level_offsets[l1 - 1];
            let b_off = bch.level_offsets[l2 - 1];
            for i in 0..blk.n_src1 {
                let ai = a[a_off + i];
                if ai == 0.0 {
                    continue;
                }
                for j in 0..blk.n_src2 {
                    let bj = b[b_off + j];
                    if bj == 0.0 {
                        continue;
                    }
                    let val = ai * bj;
                    for e in &blk.entries[i * blk.n_src2 + j] {
                        result[dest_off + e.dest] += val * e.coeff;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logsignature;
    use crate::lyndon::generate_lyndon_words;

    fn setup(d: usize, m: usize) -> BchData {
        let dim = Dim::new(d).expect("valid dim");
        let depth = Depth::new(m).expect("valid depth");
        let all_words = generate_lyndon_words(d, m);
        let proj = crate::lyndon::build_projection_matrices(d, m, &all_words);
        compute_bch_data(dim, depth, &all_words, &proj)
    }

    #[test]
    fn test_bracket_antisymmetry() {
        let bch = setup(2, 4);
        let n = bch.total_size;
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        a[0] = 1.0;
        a[1] = 0.5;
        b[0] = 0.3;
        b[1] = -0.7;

        let mut ab = vec![0.0; n];
        let mut ba = vec![0.0; n];
        lie_bracket(&a, &b, &bch, &mut ab);
        lie_bracket(&b, &a, &bch, &mut ba);

        for i in 0..n {
            assert!(
                (ab[i] + ba[i]).abs() < 1e-12,
                "antisymmetry failed at {i}: ab={} ba={}",
                ab[i],
                ba[i]
            );
        }
    }

    #[test]
    fn test_bracket_jacobi_identity() {
        let bch = setup(3, 3);
        let n = bch.total_size;
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut c = vec![0.0; n];
        a[0] = 1.0;
        b[1] = 1.0;
        c[2] = 1.0;

        let mut bc = vec![0.0; n];
        let mut ca = vec![0.0; n];
        let mut ab = vec![0.0; n];
        lie_bracket(&b, &c, &bch, &mut bc);
        lie_bracket(&c, &a, &bch, &mut ca);
        lie_bracket(&a, &b, &bch, &mut ab);

        let mut t1 = vec![0.0; n];
        let mut t2 = vec![0.0; n];
        let mut t3 = vec![0.0; n];
        lie_bracket(&a, &bc, &bch, &mut t1);
        lie_bracket(&b, &ca, &bch, &mut t2);
        lie_bracket(&c, &ab, &bch, &mut t3);

        for i in 0..n {
            let sum = t1[i] + t2[i] + t3[i];
            assert!(sum.abs() < 1e-12, "Jacobi failed at {i}: sum={sum}");
        }
    }

    #[test]
    fn test_bch_matches_s_method() {
        for d in 2..=4 {
            for m in 2..=5 {
                let dim = Dim::new(d).expect("valid");
                let depth = Depth::new(m).expect("valid");
                let s = logsignature::prepare_with_method(dim, depth, false);
                let bch = setup(d, m);

                let n_pts = 8;
                let mut path_data = Vec::with_capacity(n_pts * d);
                for i in 0..n_pts {
                    for j in 0..d {
                        path_data.push(((i * 7 + j * 13 + 3) % 17) as f64 / 17.0 - 0.5);
                    }
                }
                let path = Array2::from_shape_vec((n_pts, d), path_data).expect("shape");

                let ls_s = logsignature::logsig(&path, &s);
                let ls_bch = logsig_bch(&path, &bch, dim, depth);

                for k in 0..ls_s.len() {
                    let diff = (ls_s[k] - ls_bch[k]).abs();
                    assert!(
                        diff < 1e-10,
                        "d={d} m={m} idx={k}: S={} BCH={} diff={diff}",
                        ls_s[k],
                        ls_bch[k]
                    );
                }
            }
        }
    }

    #[test]
    fn test_bch_backprop_matches_s_method() {
        use crate::backprop;

        for d in 2..=3 {
            for m in 2..=4 {
                let dim = Dim::new(d).expect("valid");
                let depth = Depth::new(m).expect("valid");
                let s = logsignature::prepare_with_method(dim, depth, false);
                let bch = setup(d, m);

                let n_pts = 6;
                let mut path_data = Vec::with_capacity(n_pts * d);
                for i in 0..n_pts {
                    for j in 0..d {
                        path_data.push(((i * 11 + j * 7 + 5) % 19) as f64 / 19.0 - 0.5);
                    }
                }
                let path = Array2::from_shape_vec((n_pts, d), path_data).expect("shape");

                let lsl = logsignature::logsiglength(dim, depth);
                let mut deriv_data = Vec::with_capacity(lsl);
                for i in 0..lsl {
                    deriv_data.push(((i * 3 + 1) % 13) as f64 / 13.0 - 0.5);
                }
                let deriv = Array1::from_vec(deriv_data);

                let grad_s = backprop::logsigbackprop(&deriv, &path, &s);
                let grad_bch = logsig_bch_backprop(&deriv, &path, &bch, dim, depth);

                for a in 0..n_pts {
                    for b in 0..d {
                        let diff = (grad_s[[a, b]] - grad_bch[[a, b]]).abs();
                        assert!(
                            diff < 1e-9,
                            "d={d} m={m} [{a},{b}]: S={} BCH={} diff={diff}",
                            grad_s[[a, b]],
                            grad_bch[[a, b]]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_straight_line_logsig() {
        let d = 3;
        let m = 4;
        let dim = Dim::new(d).expect("valid");
        let depth = Depth::new(m).expect("valid");
        let bch = setup(d, m);

        let path =
            Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 4.0, 6.0])
                .expect("shape");

        let ls = logsig_bch(&path, &bch, dim, depth);

        assert!((ls[0] - 2.0).abs() < 1e-12);
        assert!((ls[1] - 4.0).abs() < 1e-12);
        assert!((ls[2] - 6.0).abs() < 1e-12);

        for i in d..ls.len() {
            assert!(ls[i].abs() < 1e-12, "idx {i}: expected 0, got {}", ls[i]);
        }
    }

    #[test]
    fn test_compiled_program_stats() {
        for &(d, m, max_ops) in &[(2, 3, 20), (2, 5, 500), (3, 4, 700), (5, 3, 400)] {
            let bch = setup(d, m);
            let prog = &bch.compiled;
            let total = prog.ops.len();
            assert!(
                total <= max_ops,
                "d={d},m={m}: {total} ops exceeds limit {max_ops}"
            );
        }
    }

    #[test]
    fn test_compositions() {
        assert_eq!(compositions(3, 2), vec![vec![1, 2], vec![2, 1]]);
        assert_eq!(compositions(4, 2), vec![vec![1, 3], vec![2, 2], vec![3, 1]]);
        assert_eq!(
            compositions(5, 4),
            vec![
                vec![1, 1, 1, 2],
                vec![1, 1, 2, 1],
                vec![1, 2, 1, 1],
                vec![2, 1, 1, 1]
            ]
        );
        // Edge cases
        assert_eq!(compositions(0, 0), vec![Vec::<usize>::new()]);
        assert!(compositions(1, 0).is_empty());
        assert!(compositions(2, 5).is_empty());
    }
}
