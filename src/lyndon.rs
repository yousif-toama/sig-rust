use ndarray::{Array1, Array2};

use crate::error::SigError;

/// Generate all Lyndon words over alphabet `{0, ..., d-1}` up to `max_length`.
///
/// Uses Duval's algorithm.
pub fn generate_lyndon_words(d: usize, max_length: usize) -> Vec<Vec<u8>> {
    let mut words = Vec::new();
    if max_length == 0 || d == 0 {
        return words;
    }
    duval_generate(d as u8, max_length, &mut words);
    words
}

fn duval_generate(d: u8, max_length: usize, result: &mut Vec<Vec<u8>>) {
    let mut w: Vec<i16> = vec![-1];
    while !w.is_empty() {
        // Increment last character
        let last = w.len() - 1;
        w[last] += 1;
        if w.len() <= max_length {
            result.push(w.iter().map(|&c| c as u8).collect());
        }
        // Repeat pattern to fill up to max_length
        let mut i = 0;
        while w.len() < max_length {
            w.push(w[i]);
            i += 1;
        }
        // Find rightmost character that can be incremented
        while !w.is_empty() && w[w.len() - 1] == i16::from(d) - 1 {
            w.pop();
        }
    }
}

/// Standard (CFL) factorization of a Lyndon word.
///
/// Returns `(u, v)` where `word = u ++ v`, v is the longest proper Lyndon suffix.
pub fn standard_factorization(word: &[u8]) -> Result<(Vec<u8>, Vec<u8>), SigError> {
    let n = word.len();
    for split in 1..n {
        let suffix = &word[split..];
        if is_lyndon(suffix) {
            return Ok((word[..split].to_vec(), suffix.to_vec()));
        }
    }
    Err(SigError::NoFactorization(word.to_vec()))
}

/// Check whether a word is a Lyndon word.
pub fn is_lyndon(word: &[u8]) -> bool {
    let n = word.len();
    if n == 0 {
        return false;
    }
    for i in 1..n {
        let rotation: Vec<u8> = word[i..].iter().chain(word[..i].iter()).copied().collect();
        if rotation <= word.to_vec() {
            return false;
        }
    }
    true
}

/// Convert a Lyndon word to its standard Lie bracket expression.
pub fn lyndon_bracket(word: &[u8], one_indexed: bool) -> String {
    if word.len() == 1 {
        let idx = if one_indexed {
            u16::from(word[0]) + 1
        } else {
            u16::from(word[0])
        };
        return idx.to_string();
    }
    let (u, v) = standard_factorization(word).expect("valid Lyndon word");
    format!(
        "[{},{}]",
        lyndon_bracket(&u, one_indexed),
        lyndon_bracket(&v, one_indexed)
    )
}

/// Expand a Lyndon word to its tensor representation via Lie brackets.
///
/// Single letter `i` -> basis vector `e_i` in `R^d`.
/// Composite `w = uv` -> `[T(u), T(v)] = T(u) (x) T(v) - T(v) (x) T(u)`.
pub fn lyndon_to_tensor(word: &[u8], d: usize) -> Array1<f64> {
    if word.len() == 1 {
        let mut e = Array1::zeros(d);
        e[word[0] as usize] = 1.0;
        return e;
    }
    let (u, v) = standard_factorization(word).expect("valid Lyndon word");
    let t_u = lyndon_to_tensor(&u, d);
    let t_v = lyndon_to_tensor(&v, d);

    // Lie bracket: [u, v] = u tensor v - v tensor u
    let uv = outer_flat(&t_u, &t_v);
    let vu = outer_flat(&t_v, &t_u);
    uv - vu
}

/// Build projection matrices for extracting Lyndon coordinates.
///
/// For each level k, constructs a matrix from Lyndon tensor columns,
/// then computes its pseudoinverse via SVD.
pub fn build_projection_matrices(d: usize, m: usize, lyndon_words: &[Vec<u8>]) -> Vec<Array2<f64>> {
    let mut matrices = Vec::with_capacity(m);

    for k in 1..=m {
        let words_at_k: Vec<&Vec<u8>> = lyndon_words.iter().filter(|w| w.len() == k).collect();

        if words_at_k.is_empty() {
            matrices.push(Array2::zeros((0, d.pow(k as u32))));
            continue;
        }

        let tensor_dim = d.pow(k as u32);
        let num_words = words_at_k.len();
        let mut basis_matrix = Array2::zeros((tensor_dim, num_words));

        for (col, word) in words_at_k.iter().enumerate() {
            let tensor = lyndon_to_tensor(word, d);
            basis_matrix.column_mut(col).assign(&tensor);
        }

        // Compute pseudoinverse via SVD
        let pinv = pseudoinverse(&basis_matrix);
        matrices.push(pinv);
    }

    matrices
}

/// Compute the pseudoinverse of a matrix via Jacobi SVD.
///
/// Uses one-sided Jacobi rotations for robust SVD computation.
fn pseudoinverse(mat: &Array2<f64>) -> Array2<f64> {
    let (m_rows, n_cols) = mat.dim();
    if m_rows == 0 || n_cols == 0 {
        return Array2::zeros((n_cols, m_rows));
    }

    // Compute SVD via one-sided Jacobi on A^T A
    let (u, sigmas, vt) = jacobi_svd(mat);

    let tol = if sigmas.is_empty() {
        1e-14
    } else {
        sigmas[0] * 1e-12 * (m_rows.max(n_cols) as f64)
    };

    // pinv = V * diag(1/sigma) * U^T for non-zero singular values
    let k = sigmas.len();
    let mut result = Array2::zeros((n_cols, m_rows));
    for i in 0..k {
        if sigmas[i] > tol {
            let inv_s = 1.0 / sigmas[i];
            for r in 0..n_cols {
                for c in 0..m_rows {
                    result[[r, c]] += vt[[i, r]] * inv_s * u[[c, i]];
                }
            }
        }
    }
    result
}

/// One-sided Jacobi SVD: returns (U, `singular_values`, V^T).
///
/// Computes thin SVD where rank = min(m, n).
pub fn jacobi_svd(mat: &Array2<f64>) -> (Array2<f64>, Vec<f64>, Array2<f64>) {
    let (m_rows, n_cols) = mat.dim();
    let k = m_rows.min(n_cols);

    // Work on a copy: B = A^T, so B^T B = A A^T or we work on A directly
    // Use the approach: start with V = I, repeatedly apply Jacobi rotations
    // to A*V to orthogonalize columns.
    let mut av = mat.to_owned(); // m x n, we'll orthogonalize columns
    let mut v = Array2::eye(n_cols); // n x n

    // Jacobi sweeps
    for _ in 0..100 {
        let mut converged = true;
        for p in 0..n_cols {
            for q in (p + 1)..n_cols {
                // Compute 2x2 Gram matrix of columns p and q
                let col_p = av.column(p);
                let col_q = av.column(q);
                let app: f64 = col_p.dot(&col_p);
                let aqq: f64 = col_q.dot(&col_q);
                let apq: f64 = col_p.dot(&col_q);

                if apq.abs() < 1e-15 * (app * aqq).sqrt().max(1e-30) {
                    continue;
                }
                converged = false;

                // Compute Jacobi rotation angle
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let cs = 1.0 / (1.0 + t * t).sqrt();
                let sn = t * cs;

                // Apply rotation to AV columns
                for row in 0..m_rows {
                    let a_p = av[[row, p]];
                    let a_q = av[[row, q]];
                    av[[row, p]] = cs * a_p - sn * a_q;
                    av[[row, q]] = sn * a_p + cs * a_q;
                }

                // Apply rotation to V columns
                for row in 0..n_cols {
                    let v_p = v[[row, p]];
                    let v_q = v[[row, q]];
                    v[[row, p]] = cs * v_p - sn * v_q;
                    v[[row, q]] = sn * v_p + cs * v_q;
                }
            }
        }
        if converged {
            break;
        }
    }

    // Extract singular values and U from orthogonalized AV
    let mut sigmas = Vec::with_capacity(k);
    let mut u = Array2::zeros((m_rows, k));

    for j in 0..k {
        let col = av.column(j);
        let norm: f64 = col.dot(&col).sqrt();
        sigmas.push(norm);
        if norm > 1e-30 {
            for i in 0..m_rows {
                u[[i, j]] = av[[i, j]] / norm;
            }
        }
    }

    // Sort by decreasing singular value
    let mut indices: Vec<usize> = (0..k).collect();
    indices.sort_by(|&a, &b| sigmas[b].partial_cmp(&sigmas[a]).expect("no NaN"));

    let sorted_sigmas: Vec<f64> = indices.iter().map(|&i| sigmas[i]).collect();
    let mut sorted_u = Array2::zeros((m_rows, k));
    let mut sorted_vt = Array2::zeros((k, n_cols));
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        sorted_u.column_mut(new_idx).assign(&u.column(old_idx));
        sorted_vt.row_mut(new_idx).assign(&v.column(old_idx));
    }

    (sorted_u, sorted_sigmas, sorted_vt)
}

/// Flat outer product of two 1D arrays.
pub(crate) fn outer_flat(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    let mut result = Array1::zeros(a.len() * b.len());
    let out = result.as_slice_mut().expect("contiguous");
    let a_s = a.as_slice().expect("contiguous");
    let b_s = b.as_slice().expect("contiguous");
    let sb = b_s.len();
    for (i, &av) in a_s.iter().enumerate() {
        for (j, &bv) in b_s.iter().enumerate() {
            out[i * sb + j] = av * bv;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // --- generate_lyndon_words ---

    #[test]
    fn test_generate_lyndon_words_d2_m3() {
        let words = generate_lyndon_words(2, 3);
        assert_eq!(
            words,
            vec![vec![0], vec![0, 0, 1], vec![0, 1], vec![0, 1, 1], vec![1],]
        );
    }

    #[test]
    fn test_generate_lyndon_words_d3_m1() {
        let words = generate_lyndon_words(3, 1);
        assert_eq!(words, vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn test_generate_lyndon_words_empty() {
        assert!(generate_lyndon_words(0, 5).is_empty());
        assert!(generate_lyndon_words(3, 0).is_empty());
    }

    // --- is_lyndon ---

    #[test]
    fn test_is_lyndon_single_letter() {
        assert!(is_lyndon(&[0]));
        assert!(is_lyndon(&[2]));
    }

    #[test]
    fn test_is_lyndon_true() {
        assert!(is_lyndon(&[0, 1]));
        assert!(is_lyndon(&[0, 0, 1]));
    }

    #[test]
    fn test_is_lyndon_false() {
        assert!(!is_lyndon(&[1, 0]));
        assert!(!is_lyndon(&[0, 0]));
        assert!(!is_lyndon(&[]));
    }

    // --- standard_factorization ---

    #[test]
    fn test_standard_factorization_simple() {
        let (u, v) = standard_factorization(&[0, 1]).expect("valid");
        assert_eq!(u, vec![0]);
        assert_eq!(v, vec![1]);

        let (u, v) = standard_factorization(&[0, 0, 1]).expect("valid");
        assert_eq!(u, vec![0]);
        assert_eq!(v, vec![0, 1]);
    }

    #[test]
    fn test_standard_factorization_error_single_letter() {
        assert!(standard_factorization(&[0]).is_err());
    }

    // --- lyndon_bracket ---

    #[test]
    fn test_lyndon_bracket_single() {
        assert_eq!(lyndon_bracket(&[0], true), "1");
        assert_eq!(lyndon_bracket(&[1], false), "1");
        assert_eq!(lyndon_bracket(&[0], false), "0");
    }

    #[test]
    fn test_lyndon_bracket_composite() {
        assert_eq!(lyndon_bracket(&[0, 1], true), "[1,2]");
        assert_eq!(lyndon_bracket(&[0, 0, 1], true), "[1,[1,2]]");
    }

    // --- lyndon_to_tensor ---

    #[test]
    fn test_lyndon_to_tensor_single() {
        let t = lyndon_to_tensor(&[0], 2);
        assert_eq!(t.as_slice().expect("c"), &[1.0, 0.0]);

        let t = lyndon_to_tensor(&[1], 2);
        assert_eq!(t.as_slice().expect("c"), &[0.0, 1.0]);
    }

    #[test]
    fn test_lyndon_to_tensor_bracket() {
        // [0,1] in d=2: e0 (x) e1 - e1 (x) e0 = [0,1,-1,0]
        let t = lyndon_to_tensor(&[0, 1], 2);
        assert_eq!(t.as_slice().expect("c"), &[0.0, 1.0, -1.0, 0.0]);
    }

    // --- build_projection_matrices ---

    #[test]
    fn test_build_projection_matrices_shapes() {
        let words = generate_lyndon_words(2, 3);
        let matrices = build_projection_matrices(2, 3, &words);

        assert_eq!(matrices.len(), 3);
        // Level 1: 2 words of length 1, tensor dim = 2 -> (2, 2) pinv
        assert_eq!(matrices[0].dim(), (2, 2));
        // Level 2: 1 word of length 2, tensor dim = 4 -> (1, 4) pinv
        assert_eq!(matrices[1].dim(), (1, 4));
        // Level 3: 2 words of length 3, tensor dim = 8 -> (2, 8) pinv
        assert_eq!(matrices[2].dim(), (2, 8));
    }

    // --- jacobi_svd ---

    #[test]
    fn test_jacobi_svd_diagonal() {
        let mat = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 2.0]).expect("valid");
        let (u, sigmas, vt) = jacobi_svd(&mat);

        // Singular values should be [2, 1] (sorted descending)
        assert_relative_eq!(sigmas[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(sigmas[1], 1.0, epsilon = 1e-10);

        // Reconstruct: U * diag(S) * Vt should equal original
        let k = sigmas.len();
        let mut reconstructed: Array2<f64> = Array2::zeros(mat.dim());
        for i in 0..k {
            for r in 0..mat.nrows() {
                for c in 0..mat.ncols() {
                    reconstructed[[r, c]] += u[[r, i]] * sigmas[i] * vt[[i, c]];
                }
            }
        }
        for r in 0..mat.nrows() {
            for c in 0..mat.ncols() {
                assert_relative_eq!(reconstructed[[r, c]], mat[[r, c]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_jacobi_svd_reconstruction() {
        let mat =
            Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid");
        let (u, sigmas, vt) = jacobi_svd(&mat);
        let k = sigmas.len();
        let mut reconstructed: Array2<f64> = Array2::zeros(mat.dim());
        for i in 0..k {
            for r in 0..mat.nrows() {
                for c in 0..mat.ncols() {
                    reconstructed[[r, c]] += u[[r, i]] * sigmas[i] * vt[[i, c]];
                }
            }
        }
        for r in 0..mat.nrows() {
            for c in 0..mat.ncols() {
                assert_relative_eq!(reconstructed[[r, c]], mat[[r, c]], epsilon = 1e-10);
            }
        }
    }

    // --- outer_flat ---

    #[test]
    fn test_outer_flat_simple() {
        let a = Array1::from_vec(vec![1.0, 2.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);
        let result = outer_flat(&a, &b);
        assert_eq!(result.as_slice().expect("c"), &[3.0, 4.0, 6.0, 8.0]);
    }
}
