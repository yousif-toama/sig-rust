use ndarray::{Array1, Array2};

use crate::types::{BatchedLevelList, Depth, Dim, LevelList};

// ---------------------------------------------------------------------------
// Core tensor algebra operations
// ---------------------------------------------------------------------------

/// Concatenation product in the truncated tensor algebra (implicit 1 at level 0).
///
/// Computes `(1 + a) * (1 + b)` and returns the non-unit part.
/// At combined level k: `result[k] = a[k] + b[k] + sum outer(a[i], b[j])`.
pub fn tensor_multiply(a: &LevelList, b: &LevelList) -> LevelList {
    let m = a.depth();
    let mut result = a.clone();
    result.add_assign(b);

    for level_k in 1..m {
        for i in 0..level_k {
            let j = level_k - 1 - i;
            let a_i = a.level(i);
            let b_j = b.level(j);
            outer_accumulate(a_i, b_j, result.level_mut(level_k));
        }
    }

    result
}

/// Fused unconcatenation: given `result = prev ⊗ seg`, recover `prev`.
///
/// Solves `prev[k] = result[k] - seg[k] - Σ outer(prev[i], seg[j])` iterating
/// low to high (each level depends only on lower levels of `prev`).
/// Avoids the clone + negate + multiply pattern of the naive approach.
pub fn tensor_unconcatenate_into(result: &LevelList, seg: &LevelList, prev: &mut LevelList) {
    let m = result.depth();
    prev.set_sub(result, seg);

    for level_k in 1..m {
        for i in 0..level_k {
            let j = level_k - 1 - i;
            let seg_j = seg.level(j);
            let (prev_i, dest) = prev.levels_split(i, level_k);
            outer_subtract(prev_i, seg_j, dest);
        }
    }
}

/// In-place adjoint of `tensor_multiply`: writes gradients into pre-allocated buffers.
///
/// Equivalent to `tensor_multiply_adjoint` but avoids allocating `da` and `db`.
pub fn tensor_multiply_adjoint_into(
    dresult: &LevelList,
    a: &LevelList,
    b: &LevelList,
    da: &mut LevelList,
    db: &mut LevelList,
) {
    let m = a.depth();
    da.copy_from(dresult);
    db.copy_from(dresult);

    for level_k in 1..m {
        for i in 0..level_k {
            let j = level_k - 1 - i;
            let si = a.level_len(i);
            let sj = b.level_len(j);
            let dr = dresult.level(level_k);
            matvec_add_slice(dr, si, sj, b.level(j), da.level_mut(i));
            vecmat_add_slice(a.level(i), dr, si, sj, db.level_mut(j));
        }
    }
}

/// In-place concatenation product: writes `(1 + a) * (1 + b)` into `result`.
///
/// Avoids all allocations when `result` is pre-sized.
pub fn tensor_multiply_into(a: &LevelList, b: &LevelList, result: &mut LevelList) {
    let m = a.depth();
    result.set_sum(a, b);

    for level_k in 1..m {
        for i in 0..level_k {
            let j = level_k - 1 - i;
            let a_i = a.level(i);
            let b_j = b.level(j);
            outer_accumulate(a_i, b_j, result.level_mut(level_k));
        }
    }
}

/// Concatenation product with implicit 0 at level 0 (no identity terms).
pub fn tensor_multiply_nil(a: &LevelList, b: &LevelList) -> LevelList {
    let m = a.depth();
    let dim = a.dim();
    let depth = Depth::new(m).expect("m > 0");
    let mut result = LevelList::zeros(dim, depth);

    for level_k in 1..m {
        for i in 0..level_k {
            let j = level_k - 1 - i;
            outer_accumulate(a.level(i), b.level(j), result.level_mut(level_k));
        }
    }

    result
}

/// Logarithm in the truncated tensor algebra using Horner's method.
///
/// Computes `log(1 + x) = x - x*(x/2 - x*(x/3 - ...))` with level truncation.
/// At nesting depth p, only levels 1..=(m-p) contribute, saving ~50-75% of work.
pub fn tensor_log(levels: &LevelList) -> LevelList {
    let m = levels.depth();
    let dim = levels.dim();

    if m <= 1 {
        return levels.clone();
    }

    // Horner evaluation: log(1+x) = x(1 - x(1/2 - x(1/3 - ...)))
    // s starts as empty (m-1 levels), t is workspace (m levels)
    let depth_m_minus_1 = Depth::new(m - 1).expect("m >= 2");
    let depth_m = Depth::new(m).expect("m >= 1");
    let mut s = LevelList::zeros(dim, depth_m_minus_1);
    let mut t = LevelList::zeros(dim, depth_m);

    for depth in (1..=m).rev() {
        let constant = 1.0 / depth as f64;
        let max_lev = 1 + m - depth; // how many levels matter at this nesting

        // t = nil_mul(x, s) up to level max_lev (only levels 2..=max_lev)
        // This does nothing the first iteration (when s is all zeros)
        for lev in 2..=max_lev {
            // Zero t at this level
            t.level_mut(lev - 1).fill(0.0);
            // Accumulate outer products: sum over left_lev + right_lev = lev
            for left_lev in 1..lev {
                let right_lev = lev - left_lev;
                if right_lev > s.depth() {
                    continue;
                }
                outer_accumulate(
                    levels.level(left_lev - 1),
                    s.level(right_lev - 1),
                    t.level_mut(lev - 1),
                );
            }
        }

        // s = constant * x - t, up to level max_lev
        if depth > 1 {
            for lev in 1..=max_lev {
                let s_slice = s.level_mut(lev - 1);
                let x_slice = levels.level(lev - 1);
                if lev == 1 {
                    // No t contribution at level 1 (tensor product starts at lev 2)
                    for (si, &xi) in s_slice.iter_mut().zip(x_slice.iter()) {
                        *si = constant * xi;
                    }
                } else {
                    let t_slice = t.level(lev - 1);
                    for ((si, &xi), &ti) in
                        s_slice.iter_mut().zip(x_slice.iter()).zip(t_slice.iter())
                    {
                        *si = constant * xi - ti;
                    }
                }
            }
        }
    }

    // Final result: x - t (t holds x*s at the last iteration)
    // Level 1 of result = x level 1 (unchanged)
    // Levels 2..=m of result = x - t
    let mut result = levels.clone();
    for lev in 2..=m {
        let result_slice = result.level_mut(lev - 1);
        let t_slice = t.level(lev - 1);
        for (r, &tv) in result_slice.iter_mut().zip(t_slice.iter()) {
            *r -= tv;
        }
    }

    result
}

/// Signature of a single linear segment from a slice (truncated exponential).
///
/// `level[k] = h^{tensor k} / k!` for k = 1..m.
pub fn sig_of_segment_from_slice(displacement: &[f64], depth: Depth) -> LevelList {
    let d = displacement.len();
    let dim = Dim::new(d).expect("displacement must be non-empty");
    let m = depth.value();

    // Build flat data directly: level 0 = displacement, level k = prev (x) disp / (k+1)
    let total: usize = (1..=m).map(|k| dim.pow(k)).sum();
    let mut data = Vec::with_capacity(total);
    let mut offsets = Vec::with_capacity(m + 1);
    offsets.push(0);

    // Level 0
    data.extend_from_slice(displacement);
    offsets.push(data.len());

    for k in 2..=m {
        let prev_start = offsets[k - 2];
        let prev_end = offsets[k - 1];
        let prev_len = prev_end - prev_start;
        let new_len = prev_len * d;
        let scale = 1.0 / k as f64;

        // Reserve space and get pointer to new level
        let new_start = data.len();
        data.resize(new_start + new_len, 0.0);

        // We need to read from prev level and write to new level.
        // Since they don't overlap, we can split the slice.
        let (prev_part, new_part) = data.split_at_mut(new_start);
        let prev = &prev_part[prev_start..prev_end];
        outer_into_scaled(prev, displacement, scale, new_part);

        offsets.push(data.len());
    }

    LevelList::from_flat_with_offsets(data, offsets, dim)
}

/// Signature of a single linear segment (truncated exponential).
///
/// `level[k] = h^{tensor k} / k!` for k = 1..m.
pub fn sig_of_segment(displacement: &Array1<f64>, depth: Depth) -> LevelList {
    sig_of_segment_from_slice(displacement.as_slice().expect("contiguous"), depth)
}

/// Batched signature of linear segments (vectorized truncated exponential).
///
/// `displacements` has shape `(n, d)`.
pub fn sig_of_segment_batch(displacements: &Array2<f64>, depth: Depth) -> BatchedLevelList {
    let n = displacements.nrows();
    let d = displacements.ncols();
    let dim = Dim::new(d).expect("displacement dimension must be positive");
    let m = depth.value();

    let mut levels: Vec<Array2<f64>> = vec![displacements.to_owned()];

    for k in 2..=m {
        let prev = &levels[levels.len() - 1];
        let prev_cols = prev.ncols();
        let new_cols = prev_cols * d;
        let mut new_level = Array2::zeros((n, new_cols));
        let scale = 1.0 / k as f64;

        for row in 0..n {
            let prev_row = prev.row(row);
            let disp_row = displacements.row(row);
            outer_into_scaled(
                prev_row.as_slice().expect("contiguous"),
                disp_row.as_slice().expect("contiguous"),
                scale,
                new_level.row_mut(row).as_slice_mut().expect("contiguous"),
            );
        }
        levels.push(new_level);
    }

    BatchedLevelList::new(levels, dim, n)
}

/// Split a flat signature into a `LevelList`.
pub fn split_signature(flat: &Array1<f64>, dim: Dim, depth: Depth) -> LevelList {
    LevelList::from_flat(flat.as_slice().expect("contiguous"), dim, depth)
}

/// Concatenate a `LevelList` into a flat array.
pub fn concat_levels(levels: &LevelList) -> Array1<f64> {
    Array1::from_vec(levels.to_flat())
}

// ---------------------------------------------------------------------------
// Batched operations
// ---------------------------------------------------------------------------

/// Batched tensor multiply (implicit 1 at level 0).
#[allow(clippy::many_single_char_names)]
pub fn tensor_multiply_batch(lhs: &BatchedLevelList, rhs: &BatchedLevelList) -> BatchedLevelList {
    let depth = lhs.depth();
    let dim = lhs.dim();
    let batch = lhs.batch_size();

    let mut result: Vec<Array2<f64>> = (0..depth)
        .map(|k| &lhs.levels[k] + &rhs.levels[k])
        .collect();

    for (level_k, result_k) in result.iter_mut().enumerate().skip(1) {
        for i in 0..level_k {
            let j = level_k - 1 - i;
            for row in 0..batch {
                outer_accumulate(
                    lhs.levels[i].row(row).as_slice().expect("contiguous"),
                    rhs.levels[j].row(row).as_slice().expect("contiguous"),
                    result_k.row_mut(row).as_slice_mut().expect("contiguous"),
                );
            }
        }
    }

    BatchedLevelList::new(result, dim, batch)
}

// ---------------------------------------------------------------------------
// Adjoint (reverse-mode derivative) operations
// ---------------------------------------------------------------------------

/// Adjoint of `tensor_multiply`: gradients w.r.t. both inputs.
pub fn tensor_multiply_adjoint(
    dresult: &LevelList,
    a: &LevelList,
    b: &LevelList,
) -> (LevelList, LevelList) {
    let m = a.depth();
    let mut da = dresult.clone();
    let mut db = dresult.clone();

    for level_k in 1..m {
        for i in 0..level_k {
            let j = level_k - 1 - i;
            let si = a.level_len(i);
            let sj = b.level_len(j);
            let dr = dresult.level(level_k);
            // da[i] += dr @ b[j]  (matrix-vector product)
            matvec_add_slice(dr, si, sj, b.level(j), da.level_mut(i));
            // db[j] += a[i] @ dr  (vector-matrix product)
            vecmat_add_slice(a.level(i), dr, si, sj, db.level_mut(j));
        }
    }

    (da, db)
}

/// Adjoint of `tensor_multiply_nil`.
pub fn tensor_multiply_nil_adjoint(
    dresult: &LevelList,
    a: &LevelList,
    b: &LevelList,
) -> (LevelList, LevelList) {
    let m = a.depth();
    let dim = a.dim();
    let depth = Depth::new(m).expect("m > 0");
    let mut da = LevelList::zeros(dim, depth);
    let mut db = LevelList::zeros(dim, depth);

    for level_k in 1..m {
        for i in 0..level_k {
            let j = level_k - 1 - i;
            let si = a.level_len(i);
            let sj = b.level_len(j);
            let dr = dresult.level(level_k);
            matvec_add_slice(dr, si, sj, b.level(j), da.level_mut(i));
            vecmat_add_slice(a.level(i), dr, si, sj, db.level_mut(j));
        }
    }

    (da, db)
}

/// Adjoint of `tensor_log`.
///
/// Uses the naive power series adjoint (correct with Horner forward pass).
pub fn tensor_log_adjoint(dresult: &LevelList, levels: &LevelList) -> LevelList {
    let m = levels.depth();
    let dim = levels.dim();
    let depth_val = Depth::new(m).expect("m > 0");

    // Recompute forward powers (naive, for correct adjoint)
    let mut powers: Vec<LevelList> = vec![levels.clone()];
    for _ in 1..m {
        let next = tensor_multiply_nil(levels, powers.last().expect("non-empty"));
        powers.push(next);
    }

    // Direct gradient contribution from each power
    let mut dpowers: Vec<LevelList> = Vec::with_capacity(m);
    for n in 0..m {
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let coeff = sign / (n + 1) as f64;
        let mut dp = LevelList::zeros(dim, depth_val);
        for (d, &r) in dp.data_mut().iter_mut().zip(dresult.data().iter()) {
            *d = r * coeff;
        }
        dpowers.push(dp);
    }

    // Backprop through chain: powers[n] = nil_mul(x, powers[n-1])
    let mut dx = LevelList::zeros(dim, depth_val);

    for n in (1..m).rev() {
        let (da, db) = tensor_multiply_nil_adjoint(&dpowers[n], levels, &powers[n - 1]);
        dx.add_assign(&da);
        dpowers[n - 1].add_assign(&db);
    }

    dx.add_assign(&dpowers[0]);
    dx
}

/// Adjoint of `sig_of_segment`: gradient w.r.t. displacement.
pub fn sig_of_segment_adjoint(
    dresult: &LevelList,
    displacement: &Array1<f64>,
    depth: Depth,
) -> Array1<f64> {
    let d = displacement.len();
    let m = depth.value();
    let h = displacement.as_slice().expect("contiguous");

    // Recompute forward levels as flat vecs
    let mut level_data: Vec<Vec<f64>> = vec![h.to_vec()];
    for k in 2..=m {
        let prev = &level_data[level_data.len() - 1];
        let mut new_level = vec![0.0; prev.len() * d];
        outer_into_views(prev, h, &mut new_level);
        let scale = 1.0 / k as f64;
        for v in &mut new_level {
            *v *= scale;
        }
        level_data.push(new_level);
    }

    // Copy dresult levels into mutable vecs for backward pass
    let mut dlevel: Vec<Vec<f64>> = (0..m).map(|k| dresult.level(k).to_vec()).collect();
    let mut dh = vec![0.0; d];

    for k in (1..m).rev() {
        let divisor = (k + 1) as f64;
        let size_prev = d.pow(k as u32);

        // Scale dlevel[k] by 1/divisor
        let scaled: Vec<f64> = dlevel[k].iter().map(|&v| v / divisor).collect();

        // dlevel[k-1] += mat @ h
        matvec_add_slice(&scaled, size_prev, d, h, &mut dlevel[k - 1]);

        // dh += levels[k-1] @ mat
        vecmat_add_slice(&level_data[k - 1], &scaled, size_prev, d, &mut dh);
    }

    // dlevel[0] contributes directly to dh
    for (dh_i, &dl) in dh.iter_mut().zip(dlevel[0].iter()) {
        *dh_i += dl;
    }

    Array1::from_vec(dh)
}

/// Batched adjoint of `sig_of_segment`.
pub fn sig_of_segment_adjoint_batch(
    dresults: &[Array2<f64>],
    displacements: &Array2<f64>,
    depth: Depth,
) -> Array2<f64> {
    let (n, d) = (displacements.nrows(), displacements.ncols());
    let m = depth.value();

    // Recompute forward levels (batched)
    let mut levels: Vec<Array2<f64>> = vec![displacements.to_owned()];
    for k in 2..=m {
        let prev = &levels[levels.len() - 1];
        let prev_cols = prev.ncols();
        let new_cols = prev_cols * d;
        let mut new_level = Array2::zeros((n, new_cols));
        let scale = 1.0 / k as f64;
        for row in 0..n {
            outer_into_scaled(
                prev.row(row).as_slice().expect("c"),
                displacements.row(row).as_slice().expect("c"),
                scale,
                new_level.row_mut(row).as_slice_mut().expect("c"),
            );
        }
        levels.push(new_level);
    }

    let mut dlevel: Vec<Array2<f64>> = dresults.to_vec();
    let mut dh = Array2::zeros((n, d));

    for k in (1..m).rev() {
        let divisor = (k + 1) as f64;
        let scaled = &dlevel[k] / divisor;
        let size_prev = d.pow(k as u32);

        for row in 0..n {
            let scaled_row = scaled.row(row);
            let mat = scaled_row.as_slice().expect("c");
            let disp_row = displacements.row(row);
            let h = disp_row.as_slice().expect("c");
            let prev_row = levels[k - 1].row(row);
            let prev = prev_row.as_slice().expect("c");

            // dlevel[k-1][row] += mat @ h
            matvec_add_slice(
                mat,
                size_prev,
                d,
                h,
                dlevel[k - 1].row_mut(row).as_slice_mut().expect("c"),
            );

            // dh[row] += prev @ mat
            vecmat_add_slice(
                prev,
                mat,
                size_prev,
                d,
                dh.row_mut(row).as_slice_mut().expect("c"),
            );
        }
    }

    // levels[0] = h, so dlevel[0] contributes directly
    dh += &dlevel[0];
    dh
}

// ---------------------------------------------------------------------------
// Helper: outer product, matvec, vecmat
// ---------------------------------------------------------------------------

/// Compute outer product from slices.
fn outer_into_views(a: &[f64], b: &[f64], out: &mut [f64]) {
    for (&av, chunk) in a.iter().zip(out.chunks_exact_mut(b.len())) {
        for (o, &bv) in chunk.iter_mut().zip(b.iter()) {
            *o = av * bv;
        }
    }
}

/// Compute outer product and SUBTRACT from result: `result -= outer(a, b)`.
fn outer_subtract(a: &[f64], b: &[f64], result: &mut [f64]) {
    for (&av, chunk) in a.iter().zip(result.chunks_exact_mut(b.len())) {
        for (r, &bv) in chunk.iter_mut().zip(b.iter()) {
            *r -= av * bv;
        }
    }
}

/// Compute outer product and ADD to result (fused, no intermediate buffer).
fn outer_accumulate(a: &[f64], b: &[f64], result: &mut [f64]) {
    for (&av, chunk) in a.iter().zip(result.chunks_exact_mut(b.len())) {
        for (r, &bv) in chunk.iter_mut().zip(b.iter()) {
            *r += av * bv;
        }
    }
}

/// Compute scaled outer product: out[i,j] = a[i] * b[j] * scale.
fn outer_into_scaled(a: &[f64], b: &[f64], scale: f64, out: &mut [f64]) {
    for (&av, chunk) in a.iter().zip(out.chunks_exact_mut(b.len())) {
        let scaled_av = av * scale;
        for (o, &bv) in chunk.iter_mut().zip(b.iter()) {
            *o = scaled_av * bv;
        }
    }
}

/// `result += mat @ vec` where mat is `(rows, cols)` stored flat.
fn matvec_add_slice(mat: &[f64], _rows: usize, cols: usize, vec: &[f64], result: &mut [f64]) {
    for (result_i, mat_row) in result.iter_mut().zip(mat.chunks_exact(cols)) {
        let mut sum = 0.0;
        for (&m, &v) in mat_row.iter().zip(vec.iter()) {
            sum += m * v;
        }
        *result_i += sum;
    }
}

/// `result += vec @ mat` where mat is `(rows, cols)` stored flat.
fn vecmat_add_slice(vec: &[f64], mat: &[f64], _rows: usize, cols: usize, result: &mut [f64]) {
    for (&vi, mat_row) in vec.iter().zip(mat.chunks_exact(cols)) {
        for (r, &m) in result.iter_mut().zip(mat_row.iter()) {
            *r += vi * m;
        }
    }
}
