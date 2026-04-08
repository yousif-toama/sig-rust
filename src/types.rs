use ndarray::Array2;

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

/// Truncated tensor algebra element stored as a single flat allocation.
///
/// Level k (0-indexed) has `d^(k+1)` elements. All levels are concatenated
/// contiguously in `data`, with `offsets` marking boundaries.
#[derive(Debug, Clone)]
pub struct LevelList {
    data: Vec<f64>,
    offsets: Vec<usize>,
    dim: Dim,
}

impl LevelList {
    /// Build from pre-allocated per-level slices.
    pub fn from_levels(levels: &[&[f64]], dim: Dim) -> Self {
        let m = levels.len();
        let total: usize = levels.iter().map(|l| l.len()).sum();
        let mut data = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(m + 1);
        offsets.push(0);
        for &level in levels {
            data.extend_from_slice(level);
            offsets.push(data.len());
        }
        Self { data, offsets, dim }
    }

    /// Build from owned Vec per level.
    pub fn from_level_vecs(levels: &[Vec<f64>], dim: Dim) -> Self {
        let m = levels.len();
        let total: usize = levels.iter().map(Vec::len).sum();
        let mut data = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(m + 1);
        offsets.push(0);
        for level in levels {
            data.extend_from_slice(level);
            offsets.push(data.len());
        }
        Self { data, offsets, dim }
    }

    /// Build from a single flat data vec + precomputed offsets.
    pub fn from_flat_with_offsets(data: Vec<f64>, offsets: Vec<usize>, dim: Dim) -> Self {
        Self { data, offsets, dim }
    }

    pub fn zeros(dim: Dim, depth: Depth) -> Self {
        let m = depth.value();
        let mut offsets = Vec::with_capacity(m + 1);
        offsets.push(0);
        let mut total = 0;
        for k in 1..=m {
            total += dim.pow(k);
            offsets.push(total);
        }
        Self {
            data: vec![0.0; total],
            offsets,
            dim,
        }
    }

    pub fn dim(&self) -> Dim {
        self.dim
    }

    pub fn depth(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Access level k (0-indexed) as a slice.
    pub fn level(&self, k: usize) -> &[f64] {
        &self.data[self.offsets[k]..self.offsets[k + 1]]
    }

    /// Mutable access to level k (0-indexed).
    pub fn level_mut(&mut self, k: usize) -> &mut [f64] {
        let start = self.offsets[k];
        let end = self.offsets[k + 1];
        &mut self.data[start..end]
    }

    /// Length of level k.
    pub fn level_len(&self, k: usize) -> usize {
        self.offsets[k + 1] - self.offsets[k]
    }

    /// The entire flat data.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Mutable access to entire flat data.
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// The offsets array (for constructing from parts).
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Split a flat signature array into a `LevelList`.
    pub fn from_flat(flat: &[f64], dim: Dim, depth: Depth) -> Self {
        let m = depth.value();
        let mut offsets = Vec::with_capacity(m + 1);
        offsets.push(0);
        let mut offset = 0;
        for k in 1..=m {
            offset += dim.pow(k);
            offsets.push(offset);
        }
        Self {
            data: flat[..offset].to_vec(),
            offsets,
            dim,
        }
    }

    /// Return the flat data as a Vec (near-zero cost clone).
    pub fn to_flat(&self) -> Vec<f64> {
        self.data.clone()
    }

    /// Consume self and return the flat data (zero-cost).
    pub fn into_flat(self) -> Vec<f64> {
        self.data
    }

    /// Add another `LevelList` in-place: `self += other`.
    pub fn add_assign(&mut self, other: &LevelList) {
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
    }

    /// Scaled add: self += scale * other.
    pub fn scaled_add(&mut self, scale: f64, other: &LevelList) {
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += scale * b;
        }
    }

    /// Copy data from another `LevelList` into self.
    pub fn copy_from(&mut self, other: &LevelList) {
        self.data.copy_from_slice(&other.data);
    }

    /// Set self = a + b (element-wise).
    pub fn set_sum(&mut self, a: &LevelList, b: &LevelList) {
        for ((dst, &av), &bv) in self.data.iter_mut().zip(a.data.iter()).zip(b.data.iter()) {
            *dst = av + bv;
        }
    }

    /// Set all data to zero.
    pub fn fill_zero(&mut self) {
        self.data.fill(0.0);
    }

    /// Negate even-indexed levels (0, 2, 4, ...) to compute S(-h) from S(h).
    ///
    /// For a segment signature where level index k stores `h^{tensor(k+1)}/(k+1)!`,
    /// negating the displacement gives `(-1)^{k+1}` at each level. Even indices
    /// (k=0,2,4,...) correspond to odd tensor powers and get negated.
    pub fn negate_even_indexed_levels(&mut self) {
        for k in (0..self.depth()).step_by(2) {
            for val in self.level_mut(k) {
                *val = -*val;
            }
        }
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
        let m = self.levels.len();
        let mut offsets = Vec::with_capacity(m + 1);
        offsets.push(0);
        let total: usize = self.levels.iter().map(Array2::ncols).sum();
        let mut data = Vec::with_capacity(total);
        for lev in &self.levels {
            let row = lev.row(i);
            data.extend_from_slice(row.as_slice().expect("contiguous"));
            offsets.push(data.len());
        }
        LevelList::from_flat_with_offsets(data, offsets, self.dim)
    }

    /// Split batched levels into left (even indices) and right (odd indices).
    pub fn split_even_odd(&self) -> (Self, Self) {
        let half = self.batch_size / 2;
        let mut left_levels = Vec::with_capacity(self.levels.len());
        let mut right_levels = Vec::with_capacity(self.levels.len());

        for lev in &self.levels {
            let cols = lev.ncols();
            let mut left = Array2::zeros((half, cols));
            let mut right = Array2::zeros((half, cols));
            for i in 0..half {
                left.row_mut(i).assign(&lev.row(2 * i));
                right.row_mut(i).assign(&lev.row(2 * i + 1));
            }
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
