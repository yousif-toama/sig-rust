use ndarray::{Array1, Array2};

use crate::algebra::{
    sig_of_segment_adjoint_batch, sig_of_segment_batch, split_signature, tensor_log_adjoint,
    tensor_multiply, tensor_multiply_adjoint,
};
use crate::logsignature::PreparedData;
use crate::signature::{sig_levels, siglength};
use crate::types::{Depth, Dim};

/// Gradient of a scalar loss through the signature w.r.t. the path.
pub fn sigbackprop(deriv: &Array1<f64>, path: &Array2<f64>, depth: Depth) -> Array2<f64> {
    let n = path.nrows();
    let d = path.ncols();

    if n < 2 {
        return Array2::zeros((n, d));
    }

    // Compute displacements via vectorized subtraction
    let displacements =
        &path.slice(ndarray::s![1.., ..]) - &path.slice(ndarray::s![..n - 1, ..]);

    let dim = Dim::new(d).expect("d > 0");
    sigbackprop_core(deriv, &displacements, dim, depth)
}

/// Core sigbackprop logic operating on displacements.
fn sigbackprop_core(
    deriv: &Array1<f64>,
    displacements: &Array2<f64>,
    dim: Dim,
    depth: Depth,
) -> Array2<f64> {
    let num_segs = displacements.nrows();
    let d = displacements.ncols();
    let m = depth.value();

    // Batch compute all segment signatures
    let seg_batch = sig_of_segment_batch(displacements, depth);

    // Convert to list of LevelLists for the fold
    let seg_sigs: Vec<_> = (0..num_segs).map(|i| seg_batch.get(i)).collect();

    // Sequential forward fold (store intermediates for backprop)
    let mut acc: Vec<_> = vec![seg_sigs[0].clone()];
    for seg in &seg_sigs[1..] {
        let next = tensor_multiply(acc.last().expect("non-empty"), seg);
        acc.push(next);
    }

    // Backward through the fold
    let dlevels = split_signature(deriv, dim, depth);
    let dseg = backprop_fold(&dlevels, &acc, &seg_sigs);

    // Stack dseg into batched arrays
    let mut dseg_batch: Vec<Array2<f64>> = Vec::with_capacity(m);
    for k in 0..m {
        let mut arr = Array2::zeros((num_segs, dseg[0].levels[k].len()));
        for (row_idx, ds) in dseg.iter().enumerate() {
            arr.row_mut(row_idx).assign(&ds.levels[k]);
        }
        dseg_batch.push(arr);
    }

    let dh = sig_of_segment_adjoint_batch(&dseg_batch, displacements, depth);

    // Convert displacement gradients to path gradients
    let n = num_segs + 1;
    let mut dpath = Array2::zeros((n, d));
    for i in 0..num_segs {
        for j in 0..d {
            dpath[[i, j]] -= dh[[i, j]];
            dpath[[i + 1, j]] += dh[[i, j]];
        }
    }

    dpath
}

/// Backpropagate through the left fold of tensor multiplications.
fn backprop_fold(
    dacc_final: &crate::types::LevelList,
    acc: &[crate::types::LevelList],
    seg_sigs: &[crate::types::LevelList],
) -> Vec<crate::types::LevelList> {
    let num_segs = seg_sigs.len();
    let m = seg_sigs[0].depth();
    let dim = seg_sigs[0].dim();

    let mut dseg: Vec<crate::types::LevelList> = (0..num_segs)
        .map(|_| crate::types::LevelList::zeros(dim, Depth::new(m).expect("m > 0")))
        .collect();

    let mut dacc_i = dacc_final.clone();

    for i in (1..num_segs).rev() {
        let (da, db) = tensor_multiply_adjoint(&dacc_i, &acc[i - 1], &seg_sigs[i]);
        dacc_i = da;
        for k in 0..m {
            dseg[i].levels[k] += &db.levels[k];
        }
    }

    // dseg[0] = dacc[0] since acc[0] = seg[0]
    for k in 0..m {
        dseg[0].levels[k] += &dacc_i.levels[k];
    }

    dseg
}

/// Compute the full Jacobian of the signature w.r.t. the path.
///
/// `jacobian[a, b, c]` = `d(sig_c)` / `d(path[a, b])`.
pub fn sigjacobian(path: &Array2<f64>, depth: Depth) -> ndarray::Array3<f64> {
    let n = path.nrows();
    let d = path.ncols();
    let dim = Dim::new(d).expect("d > 0");
    let sig_len = siglength(dim, depth);

    let mut jacobian = ndarray::Array3::zeros((n, d, sig_len));

    let mut e = Array1::zeros(sig_len);
    for c in 0..sig_len {
        e[c] = 1.0;
        let grad = sigbackprop(&e, path, depth);
        for a in 0..n {
            for b in 0..d {
                jacobian[[a, b, c]] = grad[[a, b]];
            }
        }
        e[c] = 0.0;
    }

    jacobian
}

/// Gradient through the log signature w.r.t. path.
pub fn logsigbackprop(deriv: &Array1<f64>, path: &Array2<f64>, s: &PreparedData) -> Array2<f64> {
    let n = path.nrows();
    let d = path.ncols();
    let m = s.depth.value();

    if n < 2 {
        return Array2::zeros((n, d));
    }

    // Step 1: Unproject from Lyndon basis to full tensor levels
    let dlog_levels = unproject_lyndon(deriv, s);

    // Step 2: Backprop through tensor_log
    let sig_levs = sig_levels(path, s.depth);
    let dsig_levels = tensor_log_adjoint(&dlog_levels, &sig_levs);

    // Step 3: Backprop through sig computation
    let displacements =
        &path.slice(ndarray::s![1.., ..]) - &path.slice(ndarray::s![..n - 1, ..]);
    let num_segs = n - 1;

    // Batch compute segment signatures
    let seg_batch = sig_of_segment_batch(&displacements, s.depth);
    let seg_sigs: Vec<_> = (0..num_segs).map(|i| seg_batch.get(i)).collect();

    let mut acc: Vec<_> = vec![seg_sigs[0].clone()];
    for seg in &seg_sigs[1..] {
        let next = tensor_multiply(acc.last().expect("non-empty"), seg);
        acc.push(next);
    }

    let dseg = backprop_fold(&dsig_levels, &acc, &seg_sigs);

    // Batched adjoint
    let mut dseg_batch: Vec<Array2<f64>> = Vec::with_capacity(m);
    for k in 0..m {
        let mut arr = Array2::zeros((num_segs, dseg[0].levels[k].len()));
        for (row_idx, ds) in dseg.iter().enumerate() {
            arr.row_mut(row_idx).assign(&ds.levels[k]);
        }
        dseg_batch.push(arr);
    }

    let dh = sig_of_segment_adjoint_batch(&dseg_batch, &displacements, s.depth);

    // Path gradient
    let mut dpath = Array2::zeros((n, d));
    for i in 0..num_segs {
        for j in 0..d {
            dpath[[i, j]] -= dh[[i, j]];
            dpath[[i + 1, j]] += dh[[i, j]];
        }
    }

    dpath
}

/// Unproject from Lyndon basis to full tensor levels.
fn unproject_lyndon(deriv: &Array1<f64>, s: &PreparedData) -> crate::types::LevelList {
    let m = s.depth.value();
    let dim = s.dim;
    let mut dlog_levels: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut offset = 0;

    for k in 0..m {
        let proj_matrix = &s.projection_matrices[k];
        let num_coords = proj_matrix.nrows();

        if num_coords == 0 {
            dlog_levels.push(Array1::zeros(dim.pow(k + 1)));
            continue;
        }

        let dcoords = deriv.slice(ndarray::s![offset..offset + num_coords]);
        dlog_levels.push(proj_matrix.t().dot(&dcoords));
        offset += num_coords;
    }

    crate::types::LevelList::new(dlog_levels, dim)
}
