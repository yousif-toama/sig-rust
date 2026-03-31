use ndarray::{Array1, Array2};

use crate::error::SigError;

/// Path dimension (number of coordinates per point).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim(usize);

impl Dim {
    pub fn new(d: usize) -> Result<Self, SigError> {
        if d == 0 {
            return Err(SigError::InvalidDim(d));
        }
        Ok(Self(d))
    }

    pub fn value(self) -> usize {
        self.0
    }

    /// Compute d^k.
    pub fn pow(self, k: usize) -> usize {
        self.0.pow(k as u32)
    }
}

/// Truncation depth (signature levels 1..=m).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Depth(usize);

impl Depth {
    pub fn new(m: usize) -> Result<Self, SigError> {
        if m == 0 {
            return Err(SigError::InvalidDepth(m));
        }
        Ok(Self(m))
    }

    pub fn value(self) -> usize {
        self.0
    }
}

/// Truncated tensor algebra element as a list of per-level flat vectors.
///
/// `levels[k]` has length `d^(k+1)` for k in `0..m`.
/// The implicit scalar at level 0 depends on context (1 for unit, 0 for nil).
#[derive(Debug, Clone)]
pub struct LevelList {
    pub levels: Vec<Array1<f64>>,
    dim: Dim,
}

impl LevelList {
    pub fn new(levels: Vec<Array1<f64>>, dim: Dim) -> Self {
        Self { levels, dim }
    }

    pub fn zeros(dim: Dim, depth: Depth) -> Self {
        let levels = (1..=depth.value())
            .map(|k| Array1::zeros(dim.pow(k)))
            .collect();
        Self { levels, dim }
    }

    pub fn dim(&self) -> Dim {
        self.dim
    }

    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// Split a flat signature array into a `LevelList`.
    pub fn from_flat(flat: &Array1<f64>, dim: Dim, depth: Depth) -> Self {
        let mut levels = Vec::with_capacity(depth.value());
        let mut offset = 0;
        for k in 1..=depth.value() {
            let size = dim.pow(k);
            levels.push(flat.slice(ndarray::s![offset..offset + size]).to_owned());
            offset += size;
        }
        Self { levels, dim }
    }

    /// Concatenate all levels into a flat signature array.
    pub fn to_flat(&self) -> Array1<f64> {
        let total_len: usize = self.levels.iter().map(Array1::len).sum();
        let mut flat = Array1::zeros(total_len);
        let mut offset = 0;
        for level in &self.levels {
            let len = level.len();
            flat.slice_mut(ndarray::s![offset..offset + len])
                .assign(level);
            offset += len;
        }
        flat
    }
}

/// Batched level-list: `levels[k]` has shape `(batch_size, d^(k+1))`.
#[derive(Debug, Clone)]
pub struct BatchedLevelList {
    pub levels: Vec<Array2<f64>>,
    dim: Dim,
    batch_size: usize,
}

impl BatchedLevelList {
    pub fn new(levels: Vec<Array2<f64>>, dim: Dim, batch_size: usize) -> Self {
        Self {
            levels,
            dim,
            batch_size,
        }
    }

    pub fn dim(&self) -> Dim {
        self.dim
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// Extract a single element from the batch as a `LevelList`.
    pub fn get(&self, i: usize) -> LevelList {
        let levels = self
            .levels
            .iter()
            .map(|lev| lev.row(i).to_owned())
            .collect();
        LevelList::new(levels, self.dim)
    }

    /// Split batched levels into left (even indices) and right (odd indices).
    pub fn split_even_odd(&self) -> (Self, Self) {
        let half = self.batch_size / 2;
        let mut left_levels = Vec::with_capacity(self.levels.len());
        let mut right_levels = Vec::with_capacity(self.levels.len());

        for lev in &self.levels {
            let rows = lev.nrows();
            let cols = lev.ncols();
            let mut left = Array2::zeros((half, cols));
            let mut right = Array2::zeros((half, cols));
            for i in 0..half {
                left.row_mut(i).assign(&lev.row(2 * i));
                right.row_mut(i).assign(&lev.row(2 * i + 1));
            }
            // If odd number of rows, the last element is the remainder
            // (handled by the caller)
            let _ = rows; // suppress unused warning
            left_levels.push(left);
            right_levels.push(right);
        }

        (
            Self::new(left_levels, self.dim, half),
            Self::new(right_levels, self.dim, half),
        )
    }

    /// Extract the last row as a `BatchedLevelList` of size 1.
    pub fn last_row(&self) -> Self {
        let levels = self
            .levels
            .iter()
            .map(|lev| {
                lev.slice(ndarray::s![self.batch_size - 1..self.batch_size, ..])
                    .to_owned()
            })
            .collect();
        Self::new(levels, self.dim, 1)
    }

    /// Remove the last row, returning a new `BatchedLevelList`.
    pub fn without_last(&self) -> Self {
        let levels = self
            .levels
            .iter()
            .map(|lev| lev.slice(ndarray::s![..self.batch_size - 1, ..]).to_owned())
            .collect();
        Self::new(levels, self.dim, self.batch_size - 1)
    }

    /// Append another `BatchedLevelList` vertically.
    pub fn append(&self, other: &Self) -> Self {
        let new_batch = self.batch_size + other.batch_size;
        let levels = self
            .levels
            .iter()
            .zip(other.levels.iter())
            .map(|(a, b)| {
                ndarray::concatenate(ndarray::Axis(0), &[a.view(), b.view()])
                    .expect("compatible shapes")
            })
            .collect();
        Self::new(levels, self.dim, new_batch)
    }
}
