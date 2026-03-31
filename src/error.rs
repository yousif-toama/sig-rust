use thiserror::Error;

#[derive(Debug, Error)]
pub enum SigError {
    #[error("path must have at least 2 dimensions with n >= 1 and d >= 1, got shape {shape:?}")]
    InvalidPathShape { shape: Vec<usize> },

    #[error("depth must be positive, got {0}")]
    InvalidDepth(usize),

    #[error("dimension must be positive, got {0}")]
    InvalidDim(usize),

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("rotinv2d requires 2D paths, got d={0}")]
    Not2DPath(usize),

    #[error("unsupported inv_type: {0}")]
    UnsupportedInvType(String),

    #[error("no standard factorization found for word {0:?}")]
    NoFactorization(Vec<u8>),

    #[error("SVD failed for projection matrix at level {0}")]
    SvdFailed(usize),

    #[error("signature length mismatch: expected {expected}, got {actual}")]
    SignatureLengthMismatch { expected: usize, actual: usize },
}
