use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::algebra::concat_levels;
use crate::error::SigError;
use crate::signature::sig_levels;
use crate::types::Depth;

/// Precomputed data for rotation-invariant signature features.
#[derive(Debug, Clone)]
pub struct RotInv2DPreparedData {
    pub depth: Depth,
    pub inv_type: String,
    pub coefficients: Vec<Array2<f64>>,
    pub total_length: usize,
}

/// Precompute coefficient matrices for rotation-invariant features.
pub fn rotinv2dprepare(depth: Depth, inv_type: &str) -> Result<RotInv2DPreparedData, SigError> {
    if inv_type != "a" {
        return Err(SigError::UnsupportedInvType(inv_type.to_string()));
    }

    // Change-of-basis: z = x + iy, z* = x - iy
    let p1 = vec![
        vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
        vec![Complex64::new(0.0, 1.0), Complex64::new(0.0, -1.0)],
    ];

    let m = depth.value();
    let mut coefficients: Vec<Array2<f64>> = Vec::new();
    let mut total = 0;

    for j in 1..=m / 2 {
        let level = 2 * j;
        let basis = invariant_basis_at_level(&p1, level, j);
        total += basis.nrows();
        coefficients.push(basis);
    }

    Ok(RotInv2DPreparedData {
        depth,
        inv_type: inv_type.to_string(),
        coefficients,
        total_length: total,
    })
}

/// Compute orthonormal invariant coefficient vectors at one even level.
fn invariant_basis_at_level(p1: &[Vec<Complex64>], level: usize, half_level: usize) -> Array2<f64> {
    let size = 1 << level; // 2^level

    // Build P_level = P1^{tensor level} (Kronecker product)
    let mut pk = vec![vec![Complex64::new(0.0, 0.0); size]; size];
    // Start with p1
    for i in 0..2 {
        for j in 0..2 {
            pk[i][j] = p1[i][j];
        }
    }
    let mut current_size = 2;

    for _ in 1..level {
        let new_size = current_size * 2;
        let mut new_pk = vec![vec![Complex64::new(0.0, 0.0); new_size]; new_size];
        for i in 0..current_size {
            for j in 0..current_size {
                for pi in 0..2 {
                    for pj in 0..2 {
                        new_pk[i * 2 + pi][j * 2 + pj] = pk[i][j] * p1[pi][pj];
                    }
                }
            }
        }
        pk = new_pk;
        current_size = new_size;
    }

    // Find balanced indices
    let balanced = balanced_indices(level, half_level);

    // Extract complex coefficient vectors (columns of pk)
    let num_balanced = balanced.len();
    let mut real_parts = Array2::zeros((size, num_balanced));
    let mut imag_parts = Array2::zeros((size, num_balanced));

    for (col, &idx) in balanced.iter().enumerate() {
        for row in 0..size {
            real_parts[[row, col]] = pk[row][idx].re;
            imag_parts[[row, col]] = pk[row][idx].im;
        }
    }

    // Build real candidate vectors
    let total_cols = 2 * num_balanced;
    let mut candidates = Array2::zeros((size, total_cols));
    candidates
        .slice_mut(ndarray::s![.., ..num_balanced])
        .assign(&real_parts);
    candidates
        .slice_mut(ndarray::s![.., num_balanced..])
        .assign(&imag_parts);

    // SVD to find orthonormal basis of the column space
    let (u, s_vals) = svd_thin(&candidates);

    let tol = if s_vals.is_empty() {
        1e-14
    } else {
        (s_vals[0] * 1e-10).max(1e-14)
    };

    let rank = s_vals.iter().filter(|&&s| s > tol).count();

    // Return U[:, :rank].T
    let mut result = Array2::zeros((rank, size));
    for i in 0..rank {
        for j in 0..size {
            result[[i, j]] = u[[j, i]];
        }
    }
    result
}

/// Column indices for balanced binary sequences.
fn balanced_indices(level: usize, half_level: usize) -> Vec<usize> {
    let mut indices = Vec::new();
    // Enumerate all combinations of `half_level` positions from `level`
    let mut positions = vec![0usize; half_level];
    if half_level == 0 {
        indices.push(0);
        return indices;
    }

    // Initialize
    for (idx, pos) in positions.iter_mut().enumerate() {
        *pos = idx;
    }

    loop {
        // Compute index from positions
        let mut idx = 0;
        for &pos in &positions {
            idx += 1 << (level - 1 - pos);
        }
        indices.push(idx);

        // Next combination
        let mut i = half_level;
        loop {
            if i == 0 {
                indices.sort_unstable();
                return indices;
            }
            i -= 1;
            positions[i] += 1;
            if positions[i] <= level - half_level + i {
                break;
            }
        }
        for j in i + 1..half_level {
            positions[j] = positions[j - 1] + 1;
        }
    }
}

/// Compute rotation-invariant features of a 2D path signature.
pub fn rotinv2d(path: &Array2<f64>, s: &RotInv2DPreparedData) -> Result<Array1<f64>, SigError> {
    let d = path.ncols();
    if d != 2 {
        return Err(SigError::Not2DPath(d));
    }

    let levels = sig_levels(path, s.depth);
    let signature = concat_levels(&levels);

    let mut result = Array1::zeros(s.total_length);
    let mut sig_offset = 0;
    let mut result_offset = 0;
    let mut coeff_idx = 0;

    for level in 1..=s.depth.value() {
        let level_size = 2usize.pow(level as u32);
        let sig_level = signature.slice(ndarray::s![sig_offset..sig_offset + level_size]);
        sig_offset += level_size;

        if level % 2 == 0 && coeff_idx < s.coefficients.len() {
            let coeffs = &s.coefficients[coeff_idx];
            let n_inv = coeffs.nrows();
            let product = coeffs.dot(&sig_level);
            result
                .slice_mut(ndarray::s![result_offset..result_offset + n_inv])
                .assign(&product);
            result_offset += n_inv;
            coeff_idx += 1;
        }
    }

    Ok(result)
}

/// Number of rotation-invariant features.
pub fn rotinv2dlength(s: &RotInv2DPreparedData) -> usize {
    s.total_length
}

/// Get the coefficient matrices for each even level.
pub fn rotinv2dcoeffs(s: &RotInv2DPreparedData) -> Vec<Array2<f64>> {
    s.coefficients.clone()
}

/// Thin SVD using Jacobi method: returns (U, `singular_values`).
fn svd_thin(mat: &Array2<f64>) -> (Array2<f64>, Vec<f64>) {
    let (m_rows, n_cols) = mat.dim();
    if m_rows == 0 || n_cols == 0 {
        return (Array2::zeros((m_rows, 0)), Vec::new());
    }

    let (u, sigmas, _vt) = crate::lyndon::jacobi_svd(mat);
    (u, sigmas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn depth_val(m: usize) -> Depth {
        Depth::new(m).expect("valid depth")
    }

    #[test]
    fn test_rotinv2dprepare_type_a() {
        let s = rotinv2dprepare(depth_val(4), "a").expect("should succeed");
        assert_eq!(s.inv_type, "a");
        assert!(!s.coefficients.is_empty());
        assert!(s.total_length > 0);
    }

    #[test]
    fn test_rotinv2dprepare_unsupported_type() {
        assert!(rotinv2dprepare(depth_val(4), "b").is_err());
        assert!(rotinv2dprepare(depth_val(4), "xyz").is_err());
    }

    #[test]
    fn test_rotinv2dprepare_depth1() {
        // Depth 1: no even levels, so no coefficients
        let s = rotinv2dprepare(depth_val(1), "a").expect("should succeed");
        assert_eq!(s.total_length, 0);
        assert!(s.coefficients.is_empty());
    }

    #[test]
    fn test_rotinv2d_not_2d_error() {
        let s = rotinv2dprepare(depth_val(4), "a").expect("ok");
        let path_3d =
            Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
                .expect("valid");
        assert!(rotinv2d(&path_3d, &s).is_err());
    }

    #[test]
    fn test_rotinv2d_output_length() {
        let s = rotinv2dprepare(depth_val(4), "a").expect("ok");
        let path = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
            .expect("valid");
        let result = rotinv2d(&path, &s).expect("ok");
        assert_eq!(result.len(), rotinv2dlength(&s));
    }

    #[test]
    fn test_rotinv2d_rotation_invariance() {
        let s = rotinv2dprepare(depth_val(4), "a").expect("ok");
        let path = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
            .expect("valid");

        let inv_orig = rotinv2d(&path, &s).expect("ok");

        // Rotate path by pi/3
        let theta = std::f64::consts::FRAC_PI_3;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let mut rotated = Array2::zeros(path.dim());
        for i in 0..path.nrows() {
            rotated[[i, 0]] = cos_t * path[[i, 0]] - sin_t * path[[i, 1]];
            rotated[[i, 1]] = sin_t * path[[i, 0]] + cos_t * path[[i, 1]];
        }

        let inv_rotated = rotinv2d(&rotated, &s).expect("ok");

        for (a, b) in inv_orig.iter().zip(inv_rotated.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_rotinv2dlength_matches_coefficients() {
        let s = rotinv2dprepare(depth_val(6), "a").expect("ok");
        let from_coeffs: usize = rotinv2dcoeffs(&s).iter().map(Array2::nrows).sum();
        assert_eq!(rotinv2dlength(&s), from_coeffs);
    }

    #[test]
    fn test_balanced_indices_zero_half() {
        // half_level=0: only index 0 (no bits set)
        let indices = balanced_indices(2, 0);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_balanced_indices_level2() {
        // level=2, half_level=1: positions where exactly 1 of 2 bits is set
        let indices = balanced_indices(2, 1);
        assert_eq!(indices, vec![1, 2]);
    }

    #[test]
    fn test_balanced_indices_level4() {
        let indices = balanced_indices(4, 2);
        // C(4,2) = 6 balanced indices
        assert_eq!(indices.len(), 6);
    }

    #[test]
    fn test_rotinv2dcoeffs_shapes() {
        let s = rotinv2dprepare(depth_val(6), "a").expect("ok");
        let coeffs = rotinv2dcoeffs(&s);
        assert_eq!(coeffs.len(), 3); // levels 2, 4, 6
        for (j, c) in coeffs.iter().enumerate() {
            let level = 2 * (j + 1);
            assert_eq!(c.ncols(), 2usize.pow(level as u32));
        }
    }
}
