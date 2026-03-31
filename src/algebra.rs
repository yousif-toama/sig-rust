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
    let dim = a.dim();
    let mut result: Vec<Array1<f64>> = (0..m).map(|k| &a.levels[k] + &b.levels[k]).collect();

    let max_size = result.last().map_or(0, Array1::len);
    let mut buf = vec![0.0_f64; max_size];

    for (level_k, result_k) in result.iter_mut().enumerate() {
        for i in 0..=level_k {
            let j = level_k.wrapping_sub(1).wrapping_sub(i);
            if j >= m {
                continue; // j < 0 in the original
            }
            let si = a.levels[i].len();
            let sj = b.levels[j].len();
            outer_into(&a.levels[i], &b.levels[j], &mut buf[..si * sj]);
            for (idx, &val) in buf[..si * sj].iter().enumerate() {
                result_k[idx] += val;
            }
        }
    }

    LevelList::new(result, dim)
}

/// In-place concatenation product: writes `(1 + a) * (1 + b)` into `result`.
///
/// `buf` must be at least as large as the biggest level (d^m elements).
/// Avoids all allocations when `result` is pre-sized.
pub fn tensor_multiply_into(
    a: &LevelList,
    b: &LevelList,
    result: &mut LevelList,
    buf: &mut [f64],
) {
    let m = a.depth();

    for k in 0..m {
        result.levels[k].assign(&a.levels[k]);
        result.levels[k] += &b.levels[k];
    }

    for level_k in 0..m {
        for i in 0..=level_k {
            let j = level_k.wrapping_sub(1).wrapping_sub(i);
            if j >= m {
                continue;
            }
            let si = a.levels[i].len();
            let sj = b.levels[j].len();
            outer_into_views(
                a.levels[i].as_slice().expect("contiguous"),
                b.levels[j].as_slice().expect("contiguous"),
                &mut buf[..si * sj],
            );
            let result_k = result.levels[level_k]
                .as_slice_mut()
                .expect("contiguous");
            for (idx, &val) in buf[..si * sj].iter().enumerate() {
                result_k[idx] += val;
            }
        }
    }
}

/// Concatenation product with implicit 0 at level 0 (no identity terms).
pub fn tensor_multiply_nil(a: &LevelList, b: &LevelList) -> LevelList {
    let m = a.depth();
    let dim = a.dim();
    let mut result: Vec<Array1<f64>> = (0..m).map(|k| Array1::zeros(a.levels[k].len())).collect();

    let max_size = result.last().map_or(0, Array1::len);
    let mut buf = vec![0.0_f64; max_size];

    for (level_k, result_k) in result.iter_mut().enumerate() {
        for i in 0..=level_k {
            let j = level_k.wrapping_sub(1).wrapping_sub(i);
            if j >= m {
                continue;
            }
            let si = a.levels[i].len();
            let sj = b.levels[j].len();
            outer_into(&a.levels[i], &b.levels[j], &mut buf[..si * sj]);
            for (idx, &val) in buf[..si * sj].iter().enumerate() {
                result_k[idx] += val;
            }
        }
    }

    LevelList::new(result, dim)
}

/// Logarithm in the truncated tensor algebra.
///
/// Given x representing `(1 + x)`, computes `log(1 + x) = x - x^2/2 + x^3/3 - ...`.
pub fn tensor_log(levels: &LevelList) -> LevelList {
    let m = levels.depth();
    let dim = levels.dim();

    // result starts as a copy of x (the n=0 term, coefficient +1)
    let mut result: Vec<Array1<f64>> = levels.levels.iter().map(Array1::clone).collect();

    // Compute powers inline: prev_power tracks x^(n) using nil multiplication
    // powers[0] = x (reference to input), powers[n] = x^(n+1)
    let mut prev_power = levels.clone();
    for n in 1..m {
        let next = tensor_multiply_nil(levels, &prev_power);
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let coeff = sign / (n + 1) as f64;
        for (result_k, power_k) in result.iter_mut().zip(&next.levels) {
            result_k.scaled_add(coeff, power_k);
        }
        prev_power = next;
    }

    LevelList::new(result, dim)
}

/// Signature of a single linear segment from a slice (truncated exponential).
///
/// `level[k] = h^{tensor k} / k!` for k = 1..m.
pub fn sig_of_segment_from_slice(displacement: &[f64], depth: Depth) -> LevelList {
    let d = displacement.len();
    let dim = Dim::new(d).expect("displacement must be non-empty");
    let m = depth.value();
    let mut levels: Vec<Array1<f64>> = vec![Array1::from_vec(displacement.to_vec())];

    for k in 2..=m {
        let prev = &levels[levels.len() - 1];
        let mut new_level = Array1::zeros(prev.len() * d);
        outer_into_views(
            prev.as_slice().expect("contiguous"),
            displacement,
            new_level.as_slice_mut().expect("contiguous"),
        );
        new_level /= k as f64;
        levels.push(new_level);
    }

    LevelList::new(levels, dim)
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

        for row in 0..n {
            let prev_row = prev.row(row);
            let disp_row = displacements.row(row);
            outer_into_views(
                prev_row.as_slice().expect("contiguous"),
                disp_row.as_slice().expect("contiguous"),
                new_level.row_mut(row).as_slice_mut().expect("contiguous"),
            );
        }
        new_level /= k as f64;
        levels.push(new_level);
    }

    BatchedLevelList::new(levels, dim, n)
}

/// Split a flat signature into a `LevelList`.
pub fn split_signature(flat: &Array1<f64>, dim: Dim, depth: Depth) -> LevelList {
    LevelList::from_flat(flat, dim, depth)
}

/// Concatenate a `LevelList` into a flat array.
pub fn concat_levels(levels: &LevelList) -> Array1<f64> {
    levels.to_flat()
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

    for (level_k, result_k) in result.iter_mut().enumerate() {
        for i in 0..=level_k {
            let j = level_k.wrapping_sub(1).wrapping_sub(i);
            if j >= depth {
                continue;
            }
            let si = lhs.levels[i].ncols();
            let sj = rhs.levels[j].ncols();
            let mut outer = Array2::zeros((batch, si * sj));
            for row in 0..batch {
                outer_into_views(
                    lhs.levels[i].row(row).as_slice().expect("contiguous"),
                    rhs.levels[j].row(row).as_slice().expect("contiguous"),
                    outer.row_mut(row).as_slice_mut().expect("contiguous"),
                );
            }
            *result_k = &*result_k + &outer;
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
    let dim = a.dim();
    let mut da: Vec<Array1<f64>> = dresult.levels.clone();
    let mut db: Vec<Array1<f64>> = dresult.levels.clone();

    for (level_k, dr_level) in dresult.levels.iter().enumerate() {
        for (i, (da_i, a_level_i)) in da.iter_mut().zip(&a.levels).enumerate().take(level_k + 1) {
            let j = level_k.wrapping_sub(1).wrapping_sub(i);
            if j >= m {
                continue;
            }
            let si = a_level_i.len();
            let sj = b.levels[j].len();
            // dr reshaped as (si, sj) matrix
            let dr = dr_level.as_slice().expect("contiguous");
            // da[i] += dr @ b[j]  (matrix-vector product)
            matvec_add(dr, si, sj, b.levels[j].as_slice().expect("c"), da_i);
            // db[j] += a[i] @ dr  (vector-matrix product)
            vecmat_add(a_level_i.as_slice().expect("c"), dr, si, sj, &mut db[j]);
        }
    }

    (LevelList::new(da, dim), LevelList::new(db, dim))
}

/// Adjoint of `tensor_multiply_nil`.
pub fn tensor_multiply_nil_adjoint(
    dresult: &LevelList,
    a: &LevelList,
    b: &LevelList,
) -> (LevelList, LevelList) {
    let m = a.depth();
    let dim = a.dim();
    let mut da: Vec<Array1<f64>> = (0..m).map(|k| Array1::zeros(a.levels[k].len())).collect();
    let mut db: Vec<Array1<f64>> = (0..m).map(|k| Array1::zeros(b.levels[k].len())).collect();

    for (level_k, dr_level) in dresult.levels.iter().enumerate() {
        for (i, (da_i, a_level_i)) in da.iter_mut().zip(&a.levels).enumerate().take(level_k + 1) {
            let j = level_k.wrapping_sub(1).wrapping_sub(i);
            if j >= m {
                continue;
            }
            let si = a_level_i.len();
            let sj = b.levels[j].len();
            let dr = dr_level.as_slice().expect("contiguous");
            matvec_add(dr, si, sj, b.levels[j].as_slice().expect("c"), da_i);
            vecmat_add(a_level_i.as_slice().expect("c"), dr, si, sj, &mut db[j]);
        }
    }

    (LevelList::new(da, dim), LevelList::new(db, dim))
}

/// Adjoint of `tensor_log`.
pub fn tensor_log_adjoint(dresult: &LevelList, levels: &LevelList) -> LevelList {
    let m = levels.depth();
    let dim = levels.dim();

    // Recompute forward powers
    let mut powers: Vec<LevelList> = vec![levels.clone()];
    for _ in 1..m {
        let next = tensor_multiply_nil(levels, powers.last().expect("non-empty"));
        powers.push(next);
    }

    // Direct gradient contribution from each power
    let mut dpowers: Vec<Vec<Array1<f64>>> = Vec::with_capacity(m);
    for n in 0..m {
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let coeff = sign / (n + 1) as f64;
        let dp: Vec<Array1<f64>> = (0..m).map(|k| &dresult.levels[k] * coeff).collect();
        dpowers.push(dp);
    }

    // Backprop through chain: powers[n] = nil_mul(x, powers[n-1])
    let mut dx: Vec<Array1<f64>> = (0..m)
        .map(|k| Array1::zeros(levels.levels[k].len()))
        .collect();

    for n in (1..m).rev() {
        let dp_level_list = LevelList::new(dpowers[n].clone(), dim);
        let (da, db) = tensor_multiply_nil_adjoint(&dp_level_list, levels, &powers[n - 1]);
        for k in 0..m {
            dx[k] += &da.levels[k];
            dpowers[n - 1][k] += &db.levels[k];
        }
    }

    // dpowers[0] is the accumulated gradient for powers[0] = x
    for k in 0..m {
        dx[k] += &dpowers[0][k];
    }

    LevelList::new(dx, dim)
}

/// Adjoint of `sig_of_segment`: gradient w.r.t. displacement.
pub fn sig_of_segment_adjoint(
    dresult: &LevelList,
    displacement: &Array1<f64>,
    depth: Depth,
) -> Array1<f64> {
    let d = displacement.len();
    let m = depth.value();
    let h = displacement;

    // Recompute forward levels
    let mut levels: Vec<Array1<f64>> = vec![h.clone()];
    for k in 2..=m {
        let prev = &levels[levels.len() - 1];
        let mut new_level = Array1::zeros(prev.len() * d);
        outer_into(prev, h, new_level.as_slice_mut().expect("contiguous"));
        new_level /= k as f64;
        levels.push(new_level);
    }

    let mut dlevel: Vec<Array1<f64>> = dresult.levels.clone();
    let mut dh = Array1::zeros(d);

    for k in (1..m).rev() {
        let divisor = (k + 1) as f64;
        let scaled: Array1<f64> = &dlevel[k] / divisor;
        let size_prev = d.pow(k as u32);

        // dlevel[k-1] += mat @ h
        let mat = scaled.as_slice().expect("contiguous");
        matvec_add(
            mat,
            size_prev,
            d,
            h.as_slice().expect("c"),
            &mut dlevel[k - 1],
        );

        // dh += levels[k-1] @ mat
        vecmat_add(
            levels[k - 1].as_slice().expect("c"),
            mat,
            size_prev,
            d,
            &mut dh,
        );
    }

    dh += &dlevel[0];
    dh
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
        for row in 0..n {
            outer_into_views(
                prev.row(row).as_slice().expect("c"),
                displacements.row(row).as_slice().expect("c"),
                new_level.row_mut(row).as_slice_mut().expect("c"),
            );
        }
        new_level /= k as f64;
        levels.push(new_level);
    }

    let mut dlevel: Vec<Array2<f64>> = dresults.to_vec();
    let mut dh = Array2::zeros((n, d));

    for k in (1..m).rev() {
        let divisor = (k + 1) as f64;
        let scaled = &dlevel[k] / divisor;
        let size_prev = d.pow(k as u32);

        for row in 0..n {
            let mat_vec: Vec<f64> = scaled.row(row).to_vec();
            let h_vec: Vec<f64> = displacements.row(row).to_vec();
            let prev_vec: Vec<f64> = levels[k - 1].row(row).to_vec();

            // dlevel[k-1][row] += mat @ h
            matvec_add_slice(
                &mat_vec,
                size_prev,
                d,
                &h_vec,
                dlevel[k - 1].row_mut(row).as_slice_mut().expect("c"),
            );

            // dh[row] += prev @ mat
            vecmat_add_slice(
                &prev_vec,
                &mat_vec,
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

/// Compute outer product `a (x) b` and write into flat buffer.
fn outer_into(a: &Array1<f64>, b: &Array1<f64>, out: &mut [f64]) {
    let a_slice = a.as_slice().expect("contiguous");
    let b_slice = b.as_slice().expect("contiguous");
    outer_into_views(a_slice, b_slice, out);
}

/// Compute outer product from slices.
fn outer_into_views(a: &[f64], b: &[f64], out: &mut [f64]) {
    let sb = b.len();
    for (i, &av) in a.iter().enumerate() {
        let start = i * sb;
        for (j, &bv) in b.iter().enumerate() {
            out[start + j] = av * bv;
        }
    }
}

/// `result += mat @ vec` where mat is `(rows, cols)` stored flat.
fn matvec_add(mat: &[f64], rows: usize, cols: usize, vec: &[f64], result: &mut Array1<f64>) {
    let r = result.as_slice_mut().expect("contiguous");
    matvec_add_slice(mat, rows, cols, vec, r);
}

fn matvec_add_slice(mat: &[f64], rows: usize, cols: usize, vec: &[f64], result: &mut [f64]) {
    for (i, result_i) in result.iter_mut().enumerate().take(rows) {
        let row_start = i * cols;
        let mut sum = 0.0;
        for j in 0..cols {
            sum += mat[row_start + j] * vec[j];
        }
        *result_i += sum;
    }
}

/// `result += vec @ mat` where mat is `(rows, cols)` stored flat.
fn vecmat_add(vec: &[f64], mat: &[f64], rows: usize, cols: usize, result: &mut Array1<f64>) {
    let r = result.as_slice_mut().expect("contiguous");
    vecmat_add_slice(vec, mat, rows, cols, r);
}

fn vecmat_add_slice(vec: &[f64], mat: &[f64], rows: usize, cols: usize, result: &mut [f64]) {
    for (i, &vi) in vec.iter().enumerate().take(rows) {
        let row_start = i * cols;
        for j in 0..cols {
            result[j] += vi * mat[row_start + j];
        }
    }
}
