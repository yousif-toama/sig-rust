use ndarray::Array1;

use crate::algebra::{concat_levels, tensor_log};
use crate::bch::BchData;
use crate::lyndon::{generate_lyndon_words, lyndon_bracket, lyndon_to_tensor};
use crate::signature::{sig_levels, siglength};
use crate::types::{Depth, Dim};

/// A single Lyndon word that maps 1:1 from a sig index with a scalar factor.
#[derive(Debug, Clone)]
struct SimpleProjection {
    dest: usize,
    source: usize,
    factor: f64,
}

/// A block of Lyndon words sharing the same letter multiset.
/// Lower triangular with unit diagonal — solved by forward substitution.
#[derive(Debug, Clone)]
struct TriangularBlock {
    /// Indices into the full tensor level (d^k elements).
    sources: Vec<usize>,
    /// Indices into this level's logsig output.
    dests: Vec<usize>,
    /// Row-major lower triangular matrix (n x n), diagonal = 1.
    matrix: Vec<f64>,
}

/// Sparse projection data for one level.
#[derive(Debug, Clone)]
struct LevelProjection {
    simples: Vec<SimpleProjection>,
    triangles: Vec<TriangularBlock>,
}

/// Precomputed data for log signature computation.
#[derive(Debug, Clone)]
pub struct PreparedData {
    pub dim: Dim,
    pub depth: Depth,
    pub lyndon_words: Vec<Vec<u8>>,
    pub basis_labels: Vec<String>,
    pub projection_matrices: Vec<ndarray::Array2<f64>>,
    sparse_projections: Vec<LevelProjection>,
    /// BCH data for fast log-sig (None = use S method).
    pub bch_data: Option<BchData>,
}

/// Precompute data for log signature computation (auto-selects method).
pub fn prepare(dim: Dim, depth: Depth) -> PreparedData {
    prepare_with_method(dim, depth, should_use_bch(dim, depth))
}

/// Precompute data with explicit method selection.
pub fn prepare_with_method(dim: Dim, depth: Depth, use_bch: bool) -> PreparedData {
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

    let projection_matrices = crate::lyndon::build_projection_matrices(d, m, &all_words);
    let sparse_projections = build_sparse_projections(d, m, &all_words);

    let bch_data = if use_bch {
        Some(crate::bch::compute_bch_data(
            dim,
            depth,
            &all_words,
            &projection_matrices,
        ))
    } else {
        None
    };

    PreparedData {
        dim,
        depth,
        lyndon_words,
        basis_labels,
        projection_matrices,
        sparse_projections,
        bch_data,
    }
}

/// Heuristic: BCH is faster for small depth/dimension combos.
///
/// The compiled BCH program beats the S method when the Lyndon basis
/// is small enough that the branchless FMA loop is cheaper than
/// tensor multiply + log + project.
fn should_use_bch(dim: Dim, depth: Depth) -> bool {
    let d = dim.value();
    let m = depth.value();
    // Empirically: BCH wins for d <= 3 and m <= 3, or d=2 and m <= 4
    d >= 2 && m >= 2 && ((d <= 3 && m <= 3) || (d == 2 && m <= 4))
}

/// Get the Lyndon bracket labels for the log signature basis.
pub fn basis(s: &PreparedData) -> Vec<String> {
    s.basis_labels.clone()
}

/// Compute the log signature of a path.
///
/// Uses BCH method if available, otherwise S method (sig -> log -> project).
pub fn logsig(path: &ndarray::Array2<f64>, s: &PreparedData) -> Array1<f64> {
    if let Some(bch) = &s.bch_data {
        return crate::bch::logsig_bch(path, bch, s.dim, s.depth);
    }
    logsig_s_method(path, s)
}

/// Compute the log signature using the S method.
///
/// Pipeline: sig -> `tensor_log` -> project onto Lyndon basis.
fn logsig_s_method(path: &ndarray::Array2<f64>, s: &PreparedData) -> Array1<f64> {
    let n = path.nrows();

    if n < 2 {
        return Array1::zeros(logsiglength(s.dim, s.depth));
    }

    let levels = sig_levels(path, s.depth);
    let log_levels = tensor_log(&levels);

    let total_len = logsiglength(s.dim, s.depth);
    let mut out = vec![0.0; total_len];
    let mut write_offset = 0;

    for k in 0..s.depth.value() {
        let sig_level = log_levels.level(k);
        let proj = &s.sparse_projections[k];
        let level_out_len =
            proj.simples.len() + proj.triangles.iter().map(|t| t.dests.len()).sum::<usize>();

        // Simple scalar projections
        for simple in &proj.simples {
            out[write_offset + simple.dest] = sig_level[simple.source] * simple.factor;
        }

        // Triangular forward substitution blocks
        for tri in &proj.triangles {
            let n_block = tri.dests.len();
            for dest_idx in 0..n_block {
                let mut sum = 0.0;
                for source_idx in 0..dest_idx {
                    sum += tri.matrix[dest_idx * n_block + source_idx]
                        * out[write_offset + tri.dests[source_idx]];
                }
                // Diagonal is 1, so: out[dest] = sig[source] - sum
                out[write_offset + tri.dests[dest_idx]] = sig_level[tri.sources[dest_idx]] - sum;
            }
        }

        write_offset += level_out_len;
    }

    Array1::from_vec(out)
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

// ---------------------------------------------------------------------------
// Sparse projection construction
// ---------------------------------------------------------------------------

/// Sorted letter multiset of a word (used for grouping).
fn letter_multiset(word: &[u8]) -> Vec<u8> {
    let mut sorted = word.to_vec();
    sorted.sort_unstable();
    sorted
}

/// Convert a word to a flat tensor index: word [a,b,c] -> a*d^2 + b*d + c.
fn word_to_index(word: &[u8], d: usize) -> usize {
    let mut idx = 0;
    for &letter in word {
        idx = idx * d + letter as usize;
    }
    idx
}

/// Build sparse projection data for all levels.
fn build_sparse_projections(d: usize, m: usize, all_words: &[Vec<u8>]) -> Vec<LevelProjection> {
    let mut projections = Vec::with_capacity(m);

    for k in 1..=m {
        let words_at_k: Vec<&Vec<u8>> = all_words.iter().filter(|w| w.len() == k).collect();

        if words_at_k.is_empty() {
            projections.push(LevelProjection {
                simples: Vec::new(),
                triangles: Vec::new(),
            });
            continue;
        }

        projections.push(build_level_projection(d, k, &words_at_k));
    }

    projections
}

/// Build sparse projection for a single level.
fn build_level_projection(d: usize, k: usize, words: &[&Vec<u8>]) -> LevelProjection {
    // Level 1: direct copy, no projection needed
    if k == 1 {
        let simples: Vec<SimpleProjection> = words
            .iter()
            .enumerate()
            .map(|(dest, word)| SimpleProjection {
                dest,
                source: word[0] as usize,
                factor: 1.0,
            })
            .collect();
        return LevelProjection {
            simples,
            triangles: Vec::new(),
        };
    }

    // Group word indices by their letter multiset
    let mut groups: Vec<(Vec<u8>, Vec<usize>)> = Vec::new();
    for (dest_idx, word) in words.iter().enumerate() {
        let ms = letter_multiset(word);
        if let Some(group) = groups.iter_mut().find(|(key, _)| *key == ms) {
            group.1.push(dest_idx);
        } else {
            groups.push((ms, vec![dest_idx]));
        }
    }

    let mut simples = Vec::new();
    let mut triangles = Vec::new();

    for (_multiset, dest_indices) in &groups {
        if dest_indices.len() == 1 {
            let dest = dest_indices[0];
            let word = words[dest];
            let source = word_to_index(word, d);
            let tensor = lyndon_to_tensor(word, d);
            let coeff = tensor[source];
            simples.push(SimpleProjection {
                dest,
                source,
                factor: if coeff.abs() > 1e-15 {
                    1.0 / coeff
                } else {
                    1.0
                },
            });
        } else {
            let n_block = dest_indices.len();
            let sources: Vec<usize> = dest_indices
                .iter()
                .map(|&di| word_to_index(words[di], d))
                .collect();
            let dests: Vec<usize> = dest_indices.clone();

            // Build expansion matrix at source indices
            let mut matrix = vec![0.0; n_block * n_block];
            for (col, &di) in dest_indices.iter().enumerate() {
                let tensor = lyndon_to_tensor(words[di], d);
                for (row, &src_idx) in sources.iter().enumerate() {
                    matrix[row * n_block + col] = tensor[src_idx];
                }
            }

            triangles.push(TriangularBlock {
                sources,
                dests,
                matrix,
            });
        }
    }

    LevelProjection { simples, triangles }
}
