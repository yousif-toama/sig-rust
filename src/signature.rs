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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn dim(d: usize) -> Dim {
        Dim::new(d).expect("valid dim")
    }

    fn depth(m: usize) -> Depth {
        Depth::new(m).expect("valid depth")
    }

    #[test]
    fn test_siglength_d1() {
        assert_eq!(siglength(dim(1), depth(5)), 5);
    }

    #[test]
    fn test_siglength_d2_m2() {
        // 2 + 4 = 6
        assert_eq!(siglength(dim(2), depth(2)), 6);
    }

    #[test]
    fn test_siglength_d3_m3() {
        // 3 + 9 + 27 = 39
        assert_eq!(siglength(dim(3), depth(3)), 39);
    }

    #[test]
    fn test_siglength_formula() {
        for d in 2usize..=5 {
            for m in 1usize..=4 {
                let expected = d * (d.pow(m as u32) - 1) / (d - 1);
                assert_eq!(siglength(dim(d), depth(m)), expected);
            }
        }
    }

    #[test]
    fn test_sigformat_from_int() {
        assert_eq!(SigFormat::from_int(0).expect("ok"), SigFormat::Flat);
        assert_eq!(SigFormat::from_int(1).expect("ok"), SigFormat::Levels);
        assert_eq!(SigFormat::from_int(2).expect("ok"), SigFormat::Cumulative);
        assert!(SigFormat::from_int(3).is_err());
    }

    #[test]
    fn test_sig_flat_format() {
        let path =
            Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0]).expect("valid");
        let result = sig(&path, depth(2), SigFormat::Flat);
        match result {
            SigResult::Flat(flat) => assert_eq!(flat.len(), 6),
            _ => unreachable!("expected Flat variant"),
        }
    }

    #[test]
    fn test_sig_levels_format() {
        let path =
            Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0]).expect("valid");
        let result = sig(&path, depth(2), SigFormat::Levels);
        match result {
            SigResult::Levels(ll) => {
                assert_eq!(ll.depth(), 2);
                assert_eq!(ll.level(0).len(), 2);
                assert_eq!(ll.level(1).len(), 4);
            }
            _ => unreachable!("expected Levels variant"),
        }
    }

    #[test]
    fn test_sig_single_point() {
        let path = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).expect("valid");
        let levels = sig_levels(&path, depth(3));
        assert!(levels.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_sig_straight_line() {
        // Straight line: all points collinear, so sig = exp(total_displacement)
        let path = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("valid");
        let levels = sig_levels(&path, depth(3));
        // Total displacement = 2
        assert_relative_eq!(levels.level(0)[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(levels.level(1)[0], 2.0, epsilon = 1e-12); // 2^2/2
        assert_relative_eq!(levels.level(2)[0], 4.0 / 3.0, epsilon = 1e-12); // 2^3/6
    }

    #[test]
    fn test_sig_cumulative_shape() {
        let path = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0])
            .expect("valid");
        let cum = sig_cumulative(&path, depth(2));
        assert_eq!(cum.nrows(), 3); // n-1 rows
        assert_eq!(cum.ncols(), 6); // siglength(2, 2)
    }

    #[test]
    fn test_sigcombine_chen_identity() {
        let path = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.5, 1.5, 1.0, 2.0, 0.5, 3.0, 1.0],
        )
        .expect("valid");
        let d = dim(2);
        let m = depth(3);

        // Sig of full path
        let full = concat_levels(&sig_levels(&path, m));

        // Sig of first 3 points and last 3 points (overlapping at point 2)
        let path1 = path.slice(ndarray::s![..3, ..]).to_owned();
        let path2 = path.slice(ndarray::s![2.., ..]).to_owned();
        let sig1 = concat_levels(&sig_levels(&path1, m));
        let sig2 = concat_levels(&sig_levels(&path2, m));

        let combined = sigcombine(&sig1, &sig2, d, m);
        for (a, b) in full.iter().zip(combined.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_sig_levels_matches_flat() {
        let path = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.5, 1.5, 1.0, 2.0, 0.0])
            .expect("valid");
        let m = depth(3);

        let levels = sig_levels(&path, m);
        let from_levels = concat_levels(&levels);

        let result = sig(&path, m, SigFormat::Flat);
        match result {
            SigResult::Flat(flat) => {
                for (a, b) in from_levels.iter().zip(flat.iter()) {
                    assert_relative_eq!(a, b, epsilon = 1e-14);
                }
            }
            _ => unreachable!("expected Flat"),
        }
    }
}
