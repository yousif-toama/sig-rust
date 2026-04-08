use ndarray::Array1;

use crate::algebra::{
    concat_levels, sig_of_segment, sig_of_segment_adjoint, split_signature, tensor_multiply_adjoint,
};
use crate::signature::sigcombine;
use crate::types::{Depth, Dim};

/// Extend a signature by appending a linear segment.
pub fn sigjoin(
    sig_flat: &Array1<f64>,
    segment: &Array1<f64>,
    dim: Dim,
    depth: Depth,
    fixed_last: Option<f64>,
) -> Array1<f64> {
    let full_segment = match fixed_last {
        Some(val) => {
            let mut s = segment.to_vec();
            s.push(val);
            Array1::from_vec(s)
        }
        None => segment.clone(),
    };
    let seg_sig_flat = concat_levels(&sig_of_segment(&full_segment, depth));
    sigcombine(sig_flat, &seg_sig_flat, dim, depth)
}

/// Gradient for `sigjoin`.
pub fn sigjoinbackprop(
    deriv: &Array1<f64>,
    sig_flat: &Array1<f64>,
    segment: &Array1<f64>,
    dim: Dim,
    depth: Depth,
    fixed_last: Option<f64>,
) -> SigjoinGradient {
    let full_segment = match fixed_last {
        Some(val) => {
            let mut s = segment.to_vec();
            s.push(val);
            Array1::from_vec(s)
        }
        None => segment.clone(),
    };

    let sig_levels = split_signature(sig_flat, dim, depth);
    let seg_levels = sig_of_segment(&full_segment, depth);

    let deriv_levels = split_signature(deriv, dim, depth);
    let (dsig_levels, dseg_levels) =
        tensor_multiply_adjoint(&deriv_levels, &sig_levels, &seg_levels);

    let dsig = concat_levels(&dsig_levels);
    let dfull_segment = sig_of_segment_adjoint(&dseg_levels, &full_segment, depth);

    match fixed_last {
        Some(_) => {
            let d_seg = dfull_segment.slice(ndarray::s![..segment.len()]).to_owned();
            let d_fixed = dfull_segment[segment.len()];
            SigjoinGradient::WithFixed {
                dsig,
                dsegment: d_seg,
                dfixed_last: d_fixed,
            }
        }
        None => SigjoinGradient::WithoutFixed {
            dsig,
            dsegment: dfull_segment,
        },
    }
}

/// Result of `sigjoinbackprop`.
pub enum SigjoinGradient {
    WithoutFixed {
        dsig: Array1<f64>,
        dsegment: Array1<f64>,
    },
    WithFixed {
        dsig: Array1<f64>,
        dsegment: Array1<f64>,
        dfixed_last: f64,
    },
}

/// Rescale a signature by per-dimension scale factors.
pub fn sigscale(
    sig_flat: &Array1<f64>,
    scales: &Array1<f64>,
    dim: Dim,
    depth: Depth,
) -> Array1<f64> {
    let d = dim.value();
    let m = depth.value();
    let levels = split_signature(sig_flat, dim, depth);

    let mut result_flat: Vec<f64> = Vec::with_capacity(sig_flat.len());

    for k in 1..=m {
        let level = levels.level(k - 1);
        let scale_k = build_scale_tensor(scales.as_slice().expect("c"), d, k);

        for (i, &lv) in level.iter().enumerate() {
            result_flat.push(lv * scale_k[i]);
        }
    }

    Array1::from_vec(result_flat)
}

/// Gradient for `sigscale`.
pub fn sigscalebackprop(
    deriv: &Array1<f64>,
    sig_flat: &Array1<f64>,
    scales: &Array1<f64>,
    dim: Dim,
    depth: Depth,
) -> (Array1<f64>, Array1<f64>) {
    let d = dim.value();
    let m = depth.value();

    // dsig: sigscale is its own adjoint w.r.t. sig
    let dsig = sigscale(deriv, scales, dim, depth);

    let deriv_levels = split_signature(deriv, dim, depth);
    let sig_levels = split_signature(sig_flat, dim, depth);
    let scales_slice = scales.as_slice().expect("contiguous");

    let mut dscales = Array1::zeros(d);

    for k in 1..=m {
        let deriv_level = deriv_levels.level(k - 1);
        let sig_level = sig_levels.level(k - 1);
        let scale_k = build_scale_tensor(scales_slice, d, k);

        // Element-wise product * scale
        let scaled_product: Vec<f64> = deriv_level
            .iter()
            .zip(sig_level.iter())
            .zip(scale_k.iter())
            .map(|((&dl, &sl), &sk)| dl * sl * sk)
            .collect();

        // For each position pos in the k-fold index
        for pos in 0..k {
            let stride_before: usize = d.pow(pos as u32);
            let stride_after: usize = d.pow((k - 1 - pos) as u32);

            for dim_idx in 0..d {
                let mut contribution = 0.0;
                for before in 0..stride_before {
                    for after in 0..stride_after {
                        let idx = before * (d * stride_after) + dim_idx * stride_after + after;
                        contribution += scaled_product[idx];
                    }
                }
                if scales_slice[dim_idx].abs() > 0.0 {
                    dscales[dim_idx] += contribution / scales_slice[dim_idx];
                }
            }
        }
    }

    (dsig, dscales)
}

/// Build scale tensor: outer product of scales with itself k times.
fn build_scale_tensor(scales: &[f64], d: usize, k: usize) -> Vec<f64> {
    let mut tensor = scales.to_vec();
    for _ in 1..k {
        let prev_len = tensor.len();
        let mut new_tensor = Vec::with_capacity(prev_len * d);
        for &t in &tensor {
            for &s in scales {
                new_tensor.push(t * s);
            }
        }
        tensor = new_tensor;
    }
    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{concat_levels, sig_of_segment};
    use crate::signature::sig_levels;
    use approx::assert_relative_eq;

    fn dim_val(d: usize) -> Dim {
        Dim::new(d).expect("valid dim")
    }

    fn depth_val(m: usize) -> Depth {
        Depth::new(m).expect("valid depth")
    }

    // --- sigjoin ---

    #[test]
    fn test_sigjoin_matches_sigcombine() {
        let path = ndarray::Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.5, 1.5, 1.0])
            .expect("valid");
        let d = dim_val(2);
        let m = depth_val(2);
        let sig_flat = concat_levels(&sig_levels(&path, m));
        let segment = Array1::from_vec(vec![0.3, -0.7]);

        let joined = sigjoin(&sig_flat, &segment, d, m, None);
        let seg_sig_flat = concat_levels(&sig_of_segment(&segment, m));
        let combined = sigcombine(&sig_flat, &seg_sig_flat, d, m);

        for (a, b) in joined.iter().zip(combined.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_sigjoin_with_fixed_last() {
        let path = ndarray::Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.5, 1.0, 1.0],
        )
        .expect("valid");
        let d = dim_val(3);
        let m = depth_val(2);
        let sig_flat = concat_levels(&sig_levels(&path, m));

        let segment = Array1::from_vec(vec![0.3, -0.7]);
        let joined = sigjoin(&sig_flat, &segment, d, m, Some(0.5));

        // Should use [0.3, -0.7, 0.5] as the full segment
        let full_seg = Array1::from_vec(vec![0.3, -0.7, 0.5]);
        let expected = sigcombine(
            &sig_flat,
            &concat_levels(&sig_of_segment(&full_seg, m)),
            d,
            m,
        );

        for (a, b) in joined.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_sigjoin_zero_segment() {
        let path = ndarray::Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.5, 1.5, 1.0])
            .expect("valid");
        let d = dim_val(2);
        let m = depth_val(2);
        let sig_flat = concat_levels(&sig_levels(&path, m));
        let zero_seg = Array1::zeros(2);
        let joined = sigjoin(&sig_flat, &zero_seg, d, m, None);
        for (a, b) in sig_flat.iter().zip(joined.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    // --- sigjoinbackprop ---

    #[test]
    fn test_sigjoinbackprop_finite_diff_without_fixed() {
        let path = ndarray::Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.5, 1.5, 1.0])
            .expect("valid");
        let d = dim_val(2);
        let m = depth_val(2);
        let sig_flat = concat_levels(&sig_levels(&path, m));
        let segment = Array1::from_vec(vec![0.3, -0.7]);
        let deriv = Array1::ones(sig_flat.len());

        let grad = sigjoinbackprop(&deriv, &sig_flat, &segment, d, m, None);
        assert!(matches!(grad, SigjoinGradient::WithoutFixed { .. }));
        if let SigjoinGradient::WithoutFixed { dsig: _, dsegment } = grad {
            let eps = 1e-6;
            for i in 0..segment.len() {
                let mut s_plus = segment.clone();
                let mut s_minus = segment.clone();
                s_plus[i] += eps;
                s_minus[i] -= eps;
                let f_plus: f64 = sigjoin(&sig_flat, &s_plus, d, m, None).sum();
                let f_minus: f64 = sigjoin(&sig_flat, &s_minus, d, m, None).sum();
                let numerical = (f_plus - f_minus) / (2.0 * eps);
                assert_relative_eq!(dsegment[i], numerical, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_sigjoinbackprop_variant_types() {
        let sig_flat = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let segment = Array1::from_vec(vec![0.1, 0.2]);
        let deriv = Array1::ones(6);
        let d = dim_val(2);
        let m = depth_val(2);

        let g_no_fixed = sigjoinbackprop(&deriv, &sig_flat, &segment, d, m, None);
        assert!(matches!(g_no_fixed, SigjoinGradient::WithoutFixed { .. }));

        let g_fixed = sigjoinbackprop(&deriv, &sig_flat, &segment, d, m, Some(1.0));
        assert!(matches!(g_fixed, SigjoinGradient::WithFixed { .. }));
    }

    // --- sigscale ---

    #[test]
    fn test_sigscale_identity() {
        let path = ndarray::Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.5, 1.5, 1.0])
            .expect("valid");
        let d = dim_val(2);
        let m = depth_val(2);
        let sig_flat = concat_levels(&sig_levels(&path, m));
        let scales = Array1::ones(2);
        let scaled = sigscale(&sig_flat, &scales, d, m);
        for (a, b) in sig_flat.iter().zip(scaled.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_sigscale_known_values() {
        // dim=2, depth=2, scales=[2,3]
        // Level 1 scaled by [2,3], level 2 scaled by outer([2,3],[2,3]) = [4,6,6,9]
        let sig_flat = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let d = dim_val(2);
        let m = depth_val(2);
        let scales = Array1::from_vec(vec![2.0, 3.0]);
        let scaled = sigscale(&sig_flat, &scales, d, m);
        assert_relative_eq!(scaled[0], 2.0, epsilon = 1e-14);
        assert_relative_eq!(scaled[1], 3.0, epsilon = 1e-14);
        assert_relative_eq!(scaled[2], 4.0, epsilon = 1e-14);
        assert_relative_eq!(scaled[3], 6.0, epsilon = 1e-14);
        assert_relative_eq!(scaled[4], 6.0, epsilon = 1e-14);
        assert_relative_eq!(scaled[5], 9.0, epsilon = 1e-14);
    }

    // --- sigscalebackprop ---

    #[test]
    fn test_sigscalebackprop_finite_diff() {
        let path = ndarray::Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.5, 1.5, 1.0])
            .expect("valid");
        let d = dim_val(2);
        let m = depth_val(2);
        let sig_flat = concat_levels(&sig_levels(&path, m));
        let scales = Array1::from_vec(vec![2.0, 3.0]);
        let deriv = Array1::ones(sig_flat.len());

        let (_dsig, dscales) = sigscalebackprop(&deriv, &sig_flat, &scales, d, m);
        let eps = 1e-6;

        for i in 0..scales.len() {
            let mut s_plus = scales.clone();
            let mut s_minus = scales.clone();
            s_plus[i] += eps;
            s_minus[i] -= eps;
            let f_plus: f64 = sigscale(&sig_flat, &s_plus, d, m).sum();
            let f_minus: f64 = sigscale(&sig_flat, &s_minus, d, m).sum();
            let numerical = (f_plus - f_minus) / (2.0 * eps);
            assert_relative_eq!(dscales[i], numerical, epsilon = 1e-4);
        }
    }

    // --- build_scale_tensor ---

    #[test]
    fn test_build_scale_tensor() {
        assert_eq!(build_scale_tensor(&[2.0, 3.0], 2, 1), vec![2.0, 3.0]);
        assert_eq!(
            build_scale_tensor(&[2.0, 3.0], 2, 2),
            vec![4.0, 6.0, 6.0, 9.0]
        );
    }
}
