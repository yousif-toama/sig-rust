use ndarray::{Array1, Array2};

use crate::algebra::{
    concat_levels, sig_of_segment_from_slice, split_signature, tensor_multiply,
    tensor_multiply_into,
};
use crate::error::SigError;
use crate::types::{Depth, Dim, LevelList};

/// Length of the signature output (levels 1 through m).
///
/// `d + d^2 + ... + d^m = d*(d^m - 1)/(d - 1)` for `d > 1`, else `m`.
pub fn siglength(dim: Dim, depth: Depth) -> usize {
    let d = dim.value();
    let m = depth.value();
    if d == 1 {
        return m;
    }
    d * (d.pow(m as u32) - 1) / (d - 1)
}

/// Output format for signature computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigFormat {
    /// Flat array of shape `(siglength(d, m),)`.
    Flat,
    /// List of m arrays, one per level.
    Levels,
    /// Cumulative prefix signatures, shape `(n-1, siglength(d, m))`.
    Cumulative,
}

impl SigFormat {
    pub fn from_int(v: usize) -> Result<Self, SigError> {
        match v {
            0 => Ok(Self::Flat),
            1 => Ok(Self::Levels),
            2 => Ok(Self::Cumulative),
            _ => Err(SigError::InvalidDepth(v)), // reuse error
        }
    }
}

/// Result of signature computation (varies by format).
pub enum SigResult {
    Flat(Array1<f64>),
    Levels(LevelList),
    Cumulative(Array2<f64>),
}

/// Compute the signature of a path truncated at depth m.
pub fn sig(path: &Array2<f64>, depth: Depth, format: SigFormat) -> SigResult {
    match format {
        SigFormat::Cumulative => SigResult::Cumulative(sig_cumulative(path, depth)),
        SigFormat::Levels => SigResult::Levels(sig_levels(path, depth)),
        SigFormat::Flat => SigResult::Flat(concat_levels(&sig_levels(path, depth))),
    }
}

/// Compute the signature as a level-list.
///
/// Uses sequential left-fold of segment signatures via Chen's identity.
pub fn sig_levels(path: &Array2<f64>, depth: Depth) -> LevelList {
    let n = path.nrows();
    let d = path.ncols();
    let dim = Dim::new(d).expect("path dimension must be positive");

    if n < 2 {
        return LevelList::zeros(dim, depth);
    }

    // Compute displacements via vectorized subtraction
    let displacements = &path.slice(ndarray::s![1.., ..]) - &path.slice(ndarray::s![..n - 1, ..]);

    let num_segs = n - 1;

    // Sequential left-fold with double buffering to avoid allocations
    let mut acc =
        sig_of_segment_from_slice(displacements.row(0).as_slice().expect("contiguous"), depth);

    if num_segs > 1 {
        let mut temp = LevelList::zeros(dim, depth);

        for i in 1..num_segs {
            let seg = sig_of_segment_from_slice(
                displacements.row(i).as_slice().expect("contiguous"),
                depth,
            );
            tensor_multiply_into(&acc, &seg, &mut temp);
            std::mem::swap(&mut acc, &mut temp);
        }
    }

    acc
}

/// Compute cumulative prefix signatures (format=2).
pub fn sig_cumulative(path: &Array2<f64>, depth: Depth) -> Array2<f64> {
    let n = path.nrows();
    let d = path.ncols();
    let dim = Dim::new(d).expect("path dimension must be positive");
    let sl = siglength(dim, depth);

    if n < 2 {
        return Array2::zeros((0, sl));
    }

    // Compute displacements via vectorized subtraction
    let displacements = &path.slice(ndarray::s![1.., ..]) - &path.slice(ndarray::s![..n - 1, ..]);

    let mut result = Array2::zeros((n - 1, sl));

    // Sequential accumulation with double buffering
    let mut acc =
        sig_of_segment_from_slice(displacements.row(0).as_slice().expect("contiguous"), depth);
    result.row_mut(0).assign(&concat_levels(&acc));

    if n > 2 {
        let mut temp = LevelList::zeros(dim, depth);

        for i in 1..n - 1 {
            let seg = sig_of_segment_from_slice(
                displacements.row(i).as_slice().expect("contiguous"),
                depth,
            );
            tensor_multiply_into(&acc, &seg, &mut temp);
            std::mem::swap(&mut acc, &mut temp);
            result.row_mut(i).assign(&concat_levels(&acc));
        }
    }

    result
}

/// Combine two signatures via Chen's identity.
pub fn sigcombine(sig1: &Array1<f64>, sig2: &Array1<f64>, dim: Dim, depth: Depth) -> Array1<f64> {
    let levels1 = split_signature(sig1, dim, depth);
    let levels2 = split_signature(sig2, dim, depth);
    let result = tensor_multiply(&levels1, &levels2);
    concat_levels(&result)
}
