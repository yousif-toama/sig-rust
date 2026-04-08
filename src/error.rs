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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_path_shape_display() {
        let e = SigError::InvalidPathShape { shape: vec![1] };
        let msg = e.to_string();
        assert!(msg.contains("at least 2 dimensions"));
        assert!(msg.contains("[1]"));
    }

    #[test]
    fn test_invalid_depth_display() {
        let e = SigError::InvalidDepth(0);
        assert!(e.to_string().contains("depth must be positive"));
        assert!(e.to_string().contains('0'));
    }

    #[test]
    fn test_invalid_dim_display() {
        let e = SigError::InvalidDim(0);
        assert!(e.to_string().contains("dimension must be positive"));
        assert!(e.to_string().contains('0'));
    }

    #[test]
    fn test_dimension_mismatch_display() {
        let e = SigError::DimensionMismatch {
            expected: 3,
            actual: 5,
        };
        assert!(e.to_string().contains("expected 3"));
        assert!(e.to_string().contains("got 5"));
    }

    #[test]
    fn test_all_variants_are_debug() {
        let variants: Vec<SigError> = vec![
            SigError::InvalidPathShape { shape: vec![1, 2] },
            SigError::InvalidDepth(0),
            SigError::InvalidDim(0),
            SigError::DimensionMismatch {
                expected: 1,
                actual: 2,
            },
            SigError::Not2DPath(3),
            SigError::UnsupportedInvType("b".to_string()),
            SigError::NoFactorization(vec![0, 1]),
            SigError::SvdFailed(2),
            SigError::SignatureLengthMismatch {
                expected: 6,
                actual: 10,
            },
        ];
        for v in &variants {
            let _ = format!("{v:?}");
            let _ = format!("{v}");
        }
    }
}
