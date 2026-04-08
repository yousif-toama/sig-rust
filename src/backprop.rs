use ndarray::{Array1, Array2};

use crate::algebra::{
    sig_of_segment_adjoint, sig_of_segment_from_slice, split_signature, tensor_log_adjoint,
    tensor_multiply_adjoint_into, tensor_multiply_into, tensor_unconcatenate_into,
};
use crate::logsignature::PreparedData;
use crate::signature::{sig_levels, siglength};
use crate::types::{Depth, Dim, LevelList};

/// Gradient of a scalar loss through the signature w.r.t. the path.
pub fn sigbackprop(deriv: &Array1<f64>, path: &Array2<f64>, depth: Depth) -> Array2<f64> {
    let n = path.nrows();
    let d = path.ncols();

    if n < 2 {
        return Array2::zeros((n, d));
    }

    // Compute displacements via vectorized subtraction
    let displacements = &path.slice(ndarray::s![1.., ..]) - &path.slice(ndarray::s![..n - 1, ..]);

    let dim = Dim::new(d).expect("d > 0");
    sigbackprop_core(deriv, &displacements, dim, depth)
}

/// Core sigbackprop logic operating on displacements.
///
/// Uses the reversibility trick: instead of storing all intermediate prefix
/// signatures, recovers them by multiplying by the inverse segment signature.
/// For a straight-line segment S(h), the inverse is S(-h), computed by
/// negating even-indexed levels.
fn sigbackprop_core(
    deriv: &Array1<f64>,
    displacements: &Array2<f64>,
    dim: Dim,
    depth: Depth,
) -> Array2<f64> {
    let num_segs = displacements.nrows();
    let d = displacements.ncols();

    // Compute all segment signatures (needed for inverse and adjoint)
    let seg_sigs: Vec<_> = (0..num_segs)
        .map(|i| {
            sig_of_segment_from_slice(displacements.row(i).as_slice().expect("contiguous"), depth)
        })
        .collect();

    // Forward fold: compute ONLY the final prefix signature (double-buffered)
    let mut current_sig = seg_sigs[0].clone();
    if num_segs > 1 {
        let mut temp = LevelList::zeros(dim, depth);
        for seg in &seg_sigs[1..] {
            tensor_multiply_into(&current_sig, seg, &mut temp);
            std::mem::swap(&mut current_sig, &mut temp);
        }
    }

    // Backward pass using reversibility trick
    let mut dacc = split_signature(deriv, dim, depth);
    let n = num_segs + 1;
    let mut dpath = Array2::zeros((n, d));
    let mut prev_sig = LevelList::zeros(dim, depth);
    let mut da_buf = LevelList::zeros(dim, depth);
    let mut db_buf = LevelList::zeros(dim, depth);

    for i in (1..num_segs).rev() {
        // Recover prev_sig = sig_{i-1} via unconcatenation:
        // current_sig = prev_sig ⊗ seg_sigs[i], solve for prev_sig
        tensor_unconcatenate_into(&current_sig, &seg_sigs[i], &mut prev_sig);

        // Compute adjoint of tensor_multiply(sig_{i-1}, seg_sigs[i])
        tensor_multiply_adjoint_into(&dacc, &prev_sig, &seg_sigs[i], &mut da_buf, &mut db_buf);
        std::mem::swap(&mut dacc, &mut da_buf);

        // Compute displacement gradient for this segment
        let dh_i = sig_of_segment_adjoint(&db_buf, &displacements.row(i).to_owned(), depth);
        for j in 0..d {
            dpath[[i, j]] -= dh_i[j];
            dpath[[i + 1, j]] += dh_i[j];
        }

        // Shift: current_sig = prev_sig for next iteration
        std::mem::swap(&mut current_sig, &mut prev_sig);
    }

    // First segment: dseg[0] = dacc
    let dh_0 = sig_of_segment_adjoint(&dacc, &displacements.row(0).to_owned(), depth);
    for j in 0..d {
        dpath[[0, j]] -= dh_0[j];
        dpath[[1, j]] += dh_0[j];
    }

    dpath
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
    if let Some(bch) = &s.bch_data {
        return crate::bch::logsig_bch_backprop(deriv, path, bch, s.dim, s.depth);
    }
    logsigbackprop_s_method(deriv, path, s)
}

/// Gradient through the log signature using the S method.
///
/// Uses the same reversibility trick as `sigbackprop_core`.
fn logsigbackprop_s_method(
    deriv: &Array1<f64>,
    path: &Array2<f64>,
    s: &PreparedData,
) -> Array2<f64> {
    let n = path.nrows();
    let d = path.ncols();

    if n < 2 {
        return Array2::zeros((n, d));
    }

    // Step 1: Unproject from Lyndon basis to full tensor levels
    let dlog_levels = unproject_lyndon(deriv, s);

    // Step 2: Backprop through tensor_log
    let sig_levs = sig_levels(path, s.depth);
    let dsig_levels = tensor_log_adjoint(&dlog_levels, &sig_levs);

    // Step 3: Backprop through sig computation using reversibility
    let displacements = &path.slice(ndarray::s![1.., ..]) - &path.slice(ndarray::s![..n - 1, ..]);
    let num_segs = n - 1;
    let dim = s.dim;
    let depth = s.depth;

    let seg_sigs: Vec<_> = (0..num_segs)
        .map(|i| {
            sig_of_segment_from_slice(displacements.row(i).as_slice().expect("contiguous"), depth)
        })
        .collect();

    // Forward fold: compute only the final prefix signature
    let mut current_sig = seg_sigs[0].clone();
    if num_segs > 1 {
        let mut temp = LevelList::zeros(dim, depth);
        for seg in &seg_sigs[1..] {
            tensor_multiply_into(&current_sig, seg, &mut temp);
            std::mem::swap(&mut current_sig, &mut temp);
        }
    }

    // Backward pass using reversibility
    let mut dacc = dsig_levels;
    let mut dpath = Array2::zeros((n, d));
    let mut prev_sig = LevelList::zeros(dim, depth);
    let mut da_buf = LevelList::zeros(dim, depth);
    let mut db_buf = LevelList::zeros(dim, depth);

    for i in (1..num_segs).rev() {
        tensor_unconcatenate_into(&current_sig, &seg_sigs[i], &mut prev_sig);

        tensor_multiply_adjoint_into(&dacc, &prev_sig, &seg_sigs[i], &mut da_buf, &mut db_buf);
        std::mem::swap(&mut dacc, &mut da_buf);

        let dh_i = sig_of_segment_adjoint(&db_buf, &displacements.row(i).to_owned(), depth);
        for j in 0..d {
            dpath[[i, j]] -= dh_i[j];
            dpath[[i + 1, j]] += dh_i[j];
        }

        std::mem::swap(&mut current_sig, &mut prev_sig);
    }

    let dh_0 = sig_of_segment_adjoint(&dacc, &displacements.row(0).to_owned(), depth);
    for j in 0..d {
        dpath[[0, j]] -= dh_0[j];
        dpath[[1, j]] += dh_0[j];
    }

    dpath
}

/// Unproject from Lyndon basis to full tensor levels.
fn unproject_lyndon(deriv: &Array1<f64>, s: &PreparedData) -> LevelList {
    let m = s.depth.value();
    let dim = s.dim;

    let total: usize = (1..=m).map(|k| dim.pow(k)).sum();
    let mut data = vec![0.0; total];
    let mut offsets = Vec::with_capacity(m + 1);
    offsets.push(0);
    let mut offset = 0;
    for k in 1..=m {
        offset += dim.pow(k);
        offsets.push(offset);
    }

    let mut deriv_offset = 0;
    for k in 0..m {
        let proj_matrix = &s.projection_matrices[k];
        let num_coords = proj_matrix.nrows();

        if num_coords == 0 {
            continue;
        }

        let dcoords = deriv.slice(ndarray::s![deriv_offset..deriv_offset + num_coords]);
        let level_result = proj_matrix.t().dot(&dcoords);
        let level_slice = &mut data[offsets[k]..offsets[k + 1]];
        level_slice.copy_from_slice(level_result.as_slice().expect("c"));
        deriv_offset += num_coords;
    }

    LevelList::from_flat_with_offsets(data, offsets, dim)
}
