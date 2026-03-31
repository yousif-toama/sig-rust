use ndarray::Array1;

use crate::algebra::{concat_levels, tensor_log};
use crate::lyndon::{build_projection_matrices, generate_lyndon_words, lyndon_bracket};
use crate::signature::{sig_levels, siglength};
use crate::types::{Depth, Dim};

/// Precomputed data for log signature computation.
#[derive(Debug, Clone)]
pub struct PreparedData {
    pub dim: Dim,
    pub depth: Depth,
    pub lyndon_words: Vec<Vec<u8>>,
    pub basis_labels: Vec<String>,
    pub projection_matrices: Vec<ndarray::Array2<f64>>,
}

/// Precompute data for log signature computation.
pub fn prepare(dim: Dim, depth: Depth) -> PreparedData {
    let d = dim.value();
    let m = depth.value();
    let all_words = generate_lyndon_words(d, m);

    let mut lyndon_words: Vec<Vec<u8>> = Vec::new();
    let mut basis_labels: Vec<String> = Vec::new();

    for k in 1..=m {
        let words_at_k: Vec<&Vec<u8>> = all_words.iter().filter(|w| w.len() == k).collect();
        for w in &words_at_k {
            lyndon_words.push((*w).clone());
            basis_labels.push(lyndon_bracket(w, true));
        }
    }

    let projection_matrices = build_projection_matrices(d, m, &all_words);

    PreparedData {
        dim,
        depth,
        lyndon_words,
        basis_labels,
        projection_matrices,
    }
}

/// Get the Lyndon bracket labels for the log signature basis.
pub fn basis(s: &PreparedData) -> Vec<String> {
    s.basis_labels.clone()
}

/// Compute the log signature of a path (S method).
///
/// Pipeline: sig -> `tensor_log` -> project onto Lyndon basis.
pub fn logsig(path: &ndarray::Array2<f64>, s: &PreparedData) -> Array1<f64> {
    let n = path.nrows();

    if n < 2 {
        return Array1::zeros(logsiglength(s.dim, s.depth));
    }

    let levels = sig_levels(path, s.depth);
    let log_levels = tensor_log(&levels);

    let mut projected: Vec<f64> = Vec::new();

    for k in 0..s.depth.value() {
        let proj_matrix = &s.projection_matrices[k];
        if proj_matrix.nrows() == 0 {
            continue;
        }
        let coords = proj_matrix.dot(&log_levels.levels[k]);
        projected.extend(coords.iter());
    }

    Array1::from_vec(projected)
}

/// Compute the log signature in the full tensor expansion (X method).
pub fn logsig_expanded(path: &ndarray::Array2<f64>, s: &PreparedData) -> Array1<f64> {
    let n = path.nrows();

    if n < 2 {
        return Array1::zeros(siglength(s.dim, s.depth));
    }

    let levels = sig_levels(path, s.depth);
    let log_levels = tensor_log(&levels);
    concat_levels(&log_levels)
}

/// Length of the log signature output (number of Lyndon words up to length m).
pub fn logsiglength(dim: Dim, depth: Depth) -> usize {
    let d = dim.value();
    let m = depth.value();
    let mut total = 0;
    for k in 1..=m {
        total += necklace_count(d, k);
    }
    total
}

/// Number of Lyndon words (primitive necklaces) of length k over d letters.
fn necklace_count(d: usize, k: usize) -> usize {
    let mut total: i64 = 0;
    for j in divisors(k) {
        total += i64::from(mobius(k / j)) * d.pow(j as u32) as i64;
    }
    (total / k as i64) as usize
}

/// All positive divisors of n.
fn divisors(n: usize) -> Vec<usize> {
    let mut divs = Vec::new();
    for i in 1..=n {
        if n.is_multiple_of(i) {
            divs.push(i);
        }
    }
    divs
}

/// Mobius function mu(n).
fn mobius(n: usize) -> i32 {
    if n == 1 {
        return 1;
    }

    let mut factors = 0;
    let mut remaining = n;
    let mut d = 2;
    while d * d <= remaining {
        if remaining.is_multiple_of(d) {
            remaining /= d;
            if remaining.is_multiple_of(d) {
                return 0; // squared factor
            }
            factors += 1;
        }
        d += if d == 2 { 1 } else { 2 };
    }
    if remaining > 1 {
        factors += 1;
    }

    if factors % 2 == 0 { 1 } else { -1 }
}
