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
        let level = &levels.levels[k - 1];
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
        let product: Array1<f64> = &deriv_levels.levels[k - 1] * &sig_levels.levels[k - 1];
        let scale_k = build_scale_tensor(scales_slice, d, k);

        // Product * scale_k, then for each position, compute contribution
        let scaled_product: Vec<f64> = product
            .iter()
            .zip(scale_k.iter())
            .map(|(&p, &s)| p * s)
            .collect();

        // For each position pos in the k-fold index
        for pos in 0..k {
            let stride_before: usize = d.pow(pos as u32);
            let stride_after: usize = d.pow((k - 1 - pos) as u32);
            let block_size = stride_before * d * stride_after;

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
                let _ = block_size;
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
