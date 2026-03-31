use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayD, Axis, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::backprop;
use crate::logsignature;
use crate::rotational;
use crate::signature::{self, SigFormat, SigResult};
use crate::transforms::{self, SigjoinGradient};
use crate::types::{Depth, Dim};

// ---------------------------------------------------------------------------
// PyO3 wrappers for PreparedData
// ---------------------------------------------------------------------------

#[pyclass(name = "PreparedData", frozen)]
pub struct PyPreparedData {
    inner: Arc<logsignature::PreparedData>,
}

#[pymethods]
impl PyPreparedData {
    #[getter]
    fn d(&self) -> usize {
        self.inner.dim.value()
    }

    #[getter]
    fn m(&self) -> usize {
        self.inner.depth.value()
    }
}

#[pyclass(name = "RotInv2DPreparedData", frozen)]
pub struct PyRotInv2DPreparedData {
    inner: Arc<rotational::RotInv2DPreparedData>,
}

#[pymethods]
impl PyRotInv2DPreparedData {
    #[getter]
    fn m(&self) -> usize {
        self.inner.depth.value()
    }
}

// ---------------------------------------------------------------------------
// Module functions
// ---------------------------------------------------------------------------

#[pyfunction(name = "sig")]
#[pyo3(signature = (path, m, format=0))]
#[allow(clippy::needless_pass_by_value)]
fn py_sig<'py>(
    py: Python<'py>,
    path: PyReadonlyArrayDyn<'py, f64>,
    m: usize,
    format: usize,
) -> PyResult<PyObject> {
    let shape = path.shape().to_vec();

    if shape.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "path must be at least 2D",
        ));
    }

    let depth =
        Depth::new(m).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let fmt = SigFormat::from_int(format)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    if shape.len() > 2 && fmt != SigFormat::Levels {
        // Batched — copy once into owned array
        let path_array = path.as_array().to_owned();
        let n = shape[shape.len() - 2];
        let d = shape[shape.len() - 1];
        let batch_shape: Vec<usize> = shape[..shape.len() - 2].to_vec();
        let batch_size: usize = batch_shape.iter().product();
        let flat_batch = path_array
            .into_shape_with_order(IxDyn(&[batch_size, n, d]))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let dim =
            Dim::new(d).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let results: Vec<Array1<f64>> = py.allow_threads(|| {
            (0..batch_size)
                .into_par_iter()
                .map(|i| {
                    let path_i: Array2<f64> = flat_batch
                        .index_axis(Axis(0), i)
                        .into_dimensionality::<ndarray::Ix2>()
                        .expect("2d")
                        .to_owned();
                    if fmt == SigFormat::Cumulative {
                        let cum = signature::sig_cumulative(&path_i, depth);
                        Array1::from_iter(cum.iter().copied())
                    } else {
                        let levels = signature::sig_levels(&path_i, depth);
                        crate::algebra::concat_levels(&levels)
                    }
                })
                .collect()
        });

        if fmt == SigFormat::Cumulative {
            let sl = signature::siglength(dim, depth);
            let mut out_shape = batch_shape.clone();
            out_shape.push(n - 1);
            out_shape.push(sl);
            let flat: Vec<f64> = results.into_iter().flat_map(|r| r.to_vec()).collect();
            let arr = ArrayD::from_shape_vec(IxDyn(&out_shape), flat)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            return Ok(arr.into_pyarray(py).into_any().unbind());
        }

        let sl = signature::siglength(dim, depth);
        let mut out_shape = batch_shape;
        out_shape.push(sl);
        let flat: Vec<f64> = results.into_iter().flat_map(|r| r.to_vec()).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&out_shape), flat)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        return Ok(arr.into_pyarray(py).into_any().unbind());
    }

    // Single path — copy directly to Array2
    let path_2d: Array2<f64> = path
        .as_array()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .to_owned();

    let result = py.allow_threads(|| signature::sig(&path_2d, depth, fmt));

    match result {
        SigResult::Flat(arr) => {
            let dyn_arr = arr.into_dyn();
            Ok(dyn_arr.into_pyarray(py).into_any().unbind())
        }
        SigResult::Levels(levels) => {
            let py_list = PyList::new(
                py,
                levels
                    .levels
                    .into_iter()
                    .map(|l| l.into_dyn().into_pyarray(py).into_any().unbind()),
            )?;
            Ok(py_list.into_any().unbind())
        }
        SigResult::Cumulative(arr) => {
            let dyn_arr = arr.into_dyn();
            Ok(dyn_arr.into_pyarray(py).into_any().unbind())
        }
    }
}

#[pyfunction(name = "siglength")]
fn py_siglength(d: usize, m: usize) -> PyResult<usize> {
    let dim = Dim::new(d).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let depth =
        Depth::new(m).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(signature::siglength(dim, depth))
}

#[pyfunction(name = "sigcombine")]
#[allow(clippy::needless_pass_by_value)]
fn py_sigcombine<'py>(
    py: Python<'py>,
    sig1: PyReadonlyArrayDyn<'py, f64>,
    sig2: PyReadonlyArrayDyn<'py, f64>,
    d: usize,
    m: usize,
) -> PyResult<PyObject> {
    let dim = Dim::new(d).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let depth =
        Depth::new(m).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let s1 = sig1.as_array().to_owned();
    let s2 = sig2.as_array().to_owned();
    let shape = s1.shape().to_vec();

    if shape.len() > 1 {
        // Batched
        let batch_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
        let batch_size: usize = batch_shape.iter().product();
        let sl = shape[shape.len() - 1];
        let flat1 = s1
            .into_shape_with_order(IxDyn(&[batch_size, sl]))
            .expect("reshape");
        let flat2 = s2
            .into_shape_with_order(IxDyn(&[batch_size, sl]))
            .expect("reshape");

        let results: Vec<Array1<f64>> = py.allow_threads(|| {
            (0..batch_size)
                .into_par_iter()
                .map(|i| {
                    let a = flat1
                        .index_axis(Axis(0), i)
                        .to_owned()
                        .into_dimensionality::<ndarray::Ix1>()
                        .expect("1d");
                    let b = flat2
                        .index_axis(Axis(0), i)
                        .to_owned()
                        .into_dimensionality::<ndarray::Ix1>()
                        .expect("1d");
                    signature::sigcombine(&a, &b, dim, depth)
                })
                .collect()
        });

        let mut out_shape = batch_shape;
        out_shape.push(sl);
        let flat: Vec<f64> = results.into_iter().flat_map(|r| r.to_vec()).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&out_shape), flat).expect("shape");
        return Ok(arr.into_pyarray(py).into_any().unbind());
    }

    let a = s1.into_dimensionality::<ndarray::Ix1>().expect("1d");
    let b = s2.into_dimensionality::<ndarray::Ix1>().expect("1d");
    let result = py.allow_threads(|| signature::sigcombine(&a, &b, dim, depth));
    Ok(result.into_dyn().into_pyarray(py).into_any().unbind())
}

#[pyfunction(name = "prepare")]
fn py_prepare(d: usize, m: usize) -> PyResult<PyPreparedData> {
    let dim = Dim::new(d).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let depth =
        Depth::new(m).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let inner = logsignature::prepare(dim, depth);
    Ok(PyPreparedData {
        inner: Arc::new(inner),
    })
}

#[pyfunction(name = "basis")]
fn py_basis(s: &PyPreparedData) -> Vec<String> {
    logsignature::basis(&s.inner)
}

#[pyfunction(name = "logsig")]
#[allow(clippy::needless_pass_by_value, clippy::unnecessary_wraps)]
fn py_logsig<'py>(
    py: Python<'py>,
    path: PyReadonlyArrayDyn<'py, f64>,
    s: &PyPreparedData,
) -> PyResult<PyObject> {
    let shape = path.shape().to_vec();
    let prepared = s.inner.clone();

    if shape.len() > 2 {
        let path_array = path.as_array().to_owned();
        let n = shape[shape.len() - 2];
        let d = shape[shape.len() - 1];
        let batch_shape: Vec<usize> = shape[..shape.len() - 2].to_vec();
        let batch_size: usize = batch_shape.iter().product();
        let flat_batch = path_array
            .into_shape_with_order(IxDyn(&[batch_size, n, d]))
            .expect("reshape");

        let results: Vec<Array1<f64>> = py.allow_threads(|| {
            (0..batch_size)
                .into_par_iter()
                .map(|i| {
                    let path_i: Array2<f64> = flat_batch
                        .index_axis(Axis(0), i)
                        .into_dimensionality::<ndarray::Ix2>()
                        .expect("2d")
                        .to_owned();
                    logsignature::logsig(&path_i, &prepared)
                })
                .collect()
        });

        let lsl = logsignature::logsiglength(prepared.dim, prepared.depth);
        let mut out_shape = batch_shape;
        out_shape.push(lsl);
        let flat: Vec<f64> = results.into_iter().flat_map(|r| r.to_vec()).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&out_shape), flat).expect("shape");
        return Ok(arr.into_pyarray(py).into_any().unbind());
    }

    // Single path — copy directly to Array2
    let path_2d: Array2<f64> = path
        .as_array()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d")
        .to_owned();
    let result = py.allow_threads(|| logsignature::logsig(&path_2d, &prepared));
    Ok(result.into_dyn().into_pyarray(py).into_any().unbind())
}

#[pyfunction(name = "logsig_expanded")]
#[allow(clippy::needless_pass_by_value, clippy::unnecessary_wraps)]
fn py_logsig_expanded<'py>(
    py: Python<'py>,
    path: PyReadonlyArrayDyn<'py, f64>,
    s: &PyPreparedData,
) -> PyResult<PyObject> {
    let shape = path.shape().to_vec();
    let prepared = s.inner.clone();

    if shape.len() > 2 {
        let path_array = path.as_array().to_owned();
        let n = shape[shape.len() - 2];
        let d = shape[shape.len() - 1];
        let batch_shape: Vec<usize> = shape[..shape.len() - 2].to_vec();
        let batch_size: usize = batch_shape.iter().product();
        let flat_batch = path_array
            .into_shape_with_order(IxDyn(&[batch_size, n, d]))
            .expect("reshape");

        let results: Vec<Array1<f64>> = py.allow_threads(|| {
            (0..batch_size)
                .into_par_iter()
                .map(|i| {
                    let path_i: Array2<f64> = flat_batch
                        .index_axis(Axis(0), i)
                        .into_dimensionality::<ndarray::Ix2>()
                        .expect("2d")
                        .to_owned();
                    logsignature::logsig_expanded(&path_i, &prepared)
                })
                .collect()
        });

        let sl = signature::siglength(prepared.dim, prepared.depth);
        let mut out_shape = batch_shape;
        out_shape.push(sl);
        let flat: Vec<f64> = results.into_iter().flat_map(|r| r.to_vec()).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&out_shape), flat).expect("shape");
        return Ok(arr.into_pyarray(py).into_any().unbind());
    }

    let path_2d: Array2<f64> = path
        .as_array()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d")
        .to_owned();
    let result = py.allow_threads(|| logsignature::logsig_expanded(&path_2d, &prepared));
    Ok(result.into_dyn().into_pyarray(py).into_any().unbind())
}

#[pyfunction(name = "logsiglength")]
fn py_logsiglength(d: usize, m: usize) -> PyResult<usize> {
    let dim = Dim::new(d).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let depth =
        Depth::new(m).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(logsignature::logsiglength(dim, depth))
}

#[pyfunction(name = "sigbackprop")]
#[allow(clippy::needless_pass_by_value)]
fn py_sigbackprop<'py>(
    py: Python<'py>,
    deriv: PyReadonlyArrayDyn<'py, f64>,
    path: PyReadonlyArrayDyn<'py, f64>,
    m: usize,
) -> PyResult<PyObject> {
    let depth =
        Depth::new(m).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let path_shape = path.shape().to_vec();
    let deriv_shape = deriv.shape().to_vec();

    if path_shape.len() > 2 {
        let path_array = path.as_array().to_owned();
        let deriv_array = deriv.as_array().to_owned();
        let n = path_shape[path_shape.len() - 2];
        let d = path_shape[path_shape.len() - 1];
        let batch_shape: Vec<usize> = path_shape[..path_shape.len() - 2].to_vec();
        let batch_size: usize = batch_shape.iter().product();
        let deriv_last = deriv_shape[deriv_shape.len() - 1];
        let flat_path = path_array
            .into_shape_with_order(IxDyn(&[batch_size, n, d]))
            .expect("reshape");
        let flat_deriv = deriv_array
            .into_shape_with_order(IxDyn(&[batch_size, deriv_last]))
            .expect("reshape");

        let results: Vec<Array2<f64>> = py.allow_threads(|| {
            (0..batch_size)
                .into_par_iter()
                .map(|i| {
                    let p: Array2<f64> = flat_path
                        .index_axis(Axis(0), i)
                        .into_dimensionality::<ndarray::Ix2>()
                        .expect("2d")
                        .to_owned();
                    let dv: Array1<f64> = flat_deriv
                        .index_axis(Axis(0), i)
                        .into_dimensionality::<ndarray::Ix1>()
                        .expect("1d")
                        .to_owned();
                    backprop::sigbackprop(&dv, &p, depth)
                })
                .collect()
        });

        let mut out_shape = batch_shape;
        out_shape.push(n);
        out_shape.push(d);
        let flat: Vec<f64> = results
            .into_iter()
            .flat_map(|r| r.iter().copied().collect::<Vec<_>>())
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&out_shape), flat).expect("shape");
        return Ok(arr.into_pyarray(py).into_any().unbind());
    }

    let path_2d: Array2<f64> = path
        .as_array()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d")
        .to_owned();
    let deriv_1d: Array1<f64> = deriv
        .as_array()
        .into_dimensionality::<ndarray::Ix1>()
        .expect("1d")
        .to_owned();
    let result = py.allow_threads(|| backprop::sigbackprop(&deriv_1d, &path_2d, depth));
    Ok(result.into_dyn().into_pyarray(py).into_any().unbind())
}

#[pyfunction(name = "sigjacobian")]
#[allow(clippy::needless_pass_by_value)]
fn py_sigjacobian<'py>(
    py: Python<'py>,
    path: PyReadonlyArrayDyn<'py, f64>,
    m: usize,
) -> PyResult<PyObject> {
    let depth =
        Depth::new(m).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let path_2d: Array2<f64> = path
        .as_array()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .to_owned();
    let result = py.allow_threads(|| backprop::sigjacobian(&path_2d, depth));
    Ok(result.into_dyn().into_pyarray(py).into_any().unbind())
}

#[pyfunction(name = "logsigbackprop")]
#[allow(clippy::needless_pass_by_value, clippy::unnecessary_wraps)]
fn py_logsigbackprop<'py>(
    py: Python<'py>,
    deriv: PyReadonlyArrayDyn<'py, f64>,
    path: PyReadonlyArrayDyn<'py, f64>,
    s: &PyPreparedData,
) -> PyResult<PyObject> {
    let path_shape = path.shape().to_vec();
    let deriv_shape = deriv.shape().to_vec();
    let prepared = s.inner.clone();

    if path_shape.len() > 2 {
        let path_array = path.as_array().to_owned();
        let deriv_array = deriv.as_array().to_owned();
        let n = path_shape[path_shape.len() - 2];
        let d = path_shape[path_shape.len() - 1];
        let batch_shape: Vec<usize> = path_shape[..path_shape.len() - 2].to_vec();
        let batch_size: usize = batch_shape.iter().product();
        let deriv_last = deriv_shape[deriv_shape.len() - 1];
        let flat_path = path_array
            .into_shape_with_order(IxDyn(&[batch_size, n, d]))
            .expect("reshape");
        let flat_deriv = deriv_array
            .into_shape_with_order(IxDyn(&[batch_size, deriv_last]))
            .expect("reshape");

        let results: Vec<Array2<f64>> = py.allow_threads(|| {
            (0..batch_size)
                .into_par_iter()
                .map(|i| {
                    let p: Array2<f64> = flat_path
                        .index_axis(Axis(0), i)
                        .into_dimensionality::<ndarray::Ix2>()
                        .expect("2d")
                        .to_owned();
                    let dv: Array1<f64> = flat_deriv
                        .index_axis(Axis(0), i)
                        .into_dimensionality::<ndarray::Ix1>()
                        .expect("1d")
                        .to_owned();
                    backprop::logsigbackprop(&dv, &p, &prepared)
                })
                .collect()
        });

        let mut out_shape = batch_shape;
        out_shape.push(n);
        out_shape.push(d);
        let flat: Vec<f64> = results
            .into_iter()
            .flat_map(|r| r.iter().copied().collect::<Vec<_>>())
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&out_shape), flat).expect("shape");
        return Ok(arr.into_pyarray(py).into_any().unbind());
    }

    let path_2d: Array2<f64> = path
        .as_array()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d")
        .to_owned();
    let deriv_1d: Array1<f64> = deriv
        .as_array()
        .into_dimensionality::<ndarray::Ix1>()
        .expect("1d")
        .to_owned();
    let result = py.allow_threads(|| backprop::logsigbackprop(&deriv_1d, &path_2d, &prepared));
    Ok(result.into_dyn().into_pyarray(py).into_any().unbind())
}

#[pyfunction(name = "sigjoin")]
#[pyo3(signature = (sig_flat, segment, d, m, fixedLast=f64::NAN))]
#[allow(clippy::needless_pass_by_value)]
fn py_sigjoin<'py>(
    py: Python<'py>,
    sig_flat: PyReadonlyArrayDyn<'py, f64>,
    segment: PyReadonlyArrayDyn<'py, f64>,
    d: usize,
    m: usize,
    #[allow(non_snake_case)] fixedLast: f64,
) -> PyResult<PyObject> {
    let dim = Dim::new(d).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let depth =
        Depth::new(m).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let sig = sig_flat
        .as_array()
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()
        .expect("1d");
    let seg = segment
        .as_array()
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()
        .expect("1d");
    let fixed = if fixedLast.is_nan() {
        None
    } else {
        Some(fixedLast)
    };

    let result = py.allow_threads(|| transforms::sigjoin(&sig, &seg, dim, depth, fixed));
    Ok(result.into_dyn().into_pyarray(py).into_any().unbind())
}

#[pyfunction(name = "sigjoinbackprop")]
#[pyo3(signature = (deriv, sig_flat, segment, d, m, fixedLast=f64::NAN))]
#[allow(clippy::needless_pass_by_value)]
fn py_sigjoinbackprop<'py>(
    py: Python<'py>,
    deriv: PyReadonlyArrayDyn<'py, f64>,
    sig_flat: PyReadonlyArrayDyn<'py, f64>,
    segment: PyReadonlyArrayDyn<'py, f64>,
    d: usize,
    m: usize,
    #[allow(non_snake_case)] fixedLast: f64,
) -> PyResult<PyObject> {
    let dim = Dim::new(d).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let depth =
        Depth::new(m).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let dv = deriv
        .as_array()
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()
        .expect("1d");
    let sig = sig_flat
        .as_array()
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()
        .expect("1d");
    let seg = segment
        .as_array()
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()
        .expect("1d");
    let fixed = if fixedLast.is_nan() {
        None
    } else {
        Some(fixedLast)
    };

    let result =
        py.allow_threads(|| transforms::sigjoinbackprop(&dv, &sig, &seg, dim, depth, fixed));

    match result {
        SigjoinGradient::WithoutFixed { dsig, dsegment } => {
            let tuple = pyo3::types::PyTuple::new(
                py,
                [
                    dsig.into_dyn().into_pyarray(py).into_any().unbind(),
                    dsegment.into_dyn().into_pyarray(py).into_any().unbind(),
                ],
            )?;
            Ok(tuple.into_any().unbind())
        }
        SigjoinGradient::WithFixed {
            dsig,
            dsegment,
            dfixed_last,
        } => {
            let dsig_py = dsig.into_dyn().into_pyarray(py).into_any().unbind();
            let dseg_py = dsegment.into_dyn().into_pyarray(py).into_any().unbind();
            let dfixed_py = dfixed_last.into_pyobject(py)?.into_any().unbind();
            let tuple = pyo3::types::PyTuple::new(py, [dsig_py, dseg_py, dfixed_py])?;
            Ok(tuple.into_any().unbind())
        }
    }
}

#[pyfunction(name = "sigscale")]
#[allow(clippy::needless_pass_by_value)]
fn py_sigscale<'py>(
    py: Python<'py>,
    sig_flat: PyReadonlyArrayDyn<'py, f64>,
    scales: PyReadonlyArrayDyn<'py, f64>,
    d: usize,
    m: usize,
) -> PyResult<PyObject> {
    let dim = Dim::new(d).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let depth =
        Depth::new(m).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let sig = sig_flat
        .as_array()
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()
        .expect("1d");
    let sc = scales
        .as_array()
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()
        .expect("1d");

    let result = py.allow_threads(|| transforms::sigscale(&sig, &sc, dim, depth));
    Ok(result.into_dyn().into_pyarray(py).into_any().unbind())
}

#[pyfunction(name = "sigscalebackprop")]
#[allow(clippy::needless_pass_by_value)]
fn py_sigscalebackprop<'py>(
    py: Python<'py>,
    deriv: PyReadonlyArrayDyn<'py, f64>,
    sig_flat: PyReadonlyArrayDyn<'py, f64>,
    scales: PyReadonlyArrayDyn<'py, f64>,
    d: usize,
    m: usize,
) -> PyResult<PyObject> {
    let dim = Dim::new(d).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let depth =
        Depth::new(m).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let dv = deriv
        .as_array()
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()
        .expect("1d");
    let sig = sig_flat
        .as_array()
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()
        .expect("1d");
    let sc = scales
        .as_array()
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()
        .expect("1d");

    let (dsig, dscales) =
        py.allow_threads(|| transforms::sigscalebackprop(&dv, &sig, &sc, dim, depth));

    let tuple = pyo3::types::PyTuple::new(
        py,
        [
            dsig.into_dyn().into_pyarray(py).into_any().unbind(),
            dscales.into_dyn().into_pyarray(py).into_any().unbind(),
        ],
    )?;
    Ok(tuple.into_any().unbind())
}

#[pyfunction(name = "rotinv2dprepare")]
#[pyo3(signature = (m, inv_type="a"))]
fn py_rotinv2dprepare(m: usize, inv_type: &str) -> PyResult<PyRotInv2DPreparedData> {
    let depth =
        Depth::new(m).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let inner = rotational::rotinv2dprepare(depth, inv_type)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyRotInv2DPreparedData {
        inner: Arc::new(inner),
    })
}

#[pyfunction(name = "rotinv2d")]
#[allow(clippy::needless_pass_by_value)]
fn py_rotinv2d<'py>(
    py: Python<'py>,
    path: PyReadonlyArrayDyn<'py, f64>,
    s: &PyRotInv2DPreparedData,
) -> PyResult<PyObject> {
    let path_2d = path
        .as_array()
        .to_owned()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let result = py
        .allow_threads(|| rotational::rotinv2d(&path_2d, &s.inner))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(result.into_dyn().into_pyarray(py).into_any().unbind())
}

#[pyfunction(name = "rotinv2dlength")]
fn py_rotinv2dlength(s: &PyRotInv2DPreparedData) -> usize {
    rotational::rotinv2dlength(&s.inner)
}

#[pyfunction(name = "rotinv2dcoeffs")]
fn py_rotinv2dcoeffs(py: Python<'_>, s: &PyRotInv2DPreparedData) -> PyResult<PyObject> {
    let coeffs = rotational::rotinv2dcoeffs(&s.inner);
    let py_list = PyList::new(
        py,
        coeffs
            .into_iter()
            .map(|c| c.into_dyn().into_pyarray(py).into_any().unbind()),
    )?;
    Ok(py_list.into_any().unbind())
}

#[pyfunction(name = "version")]
fn py_version() -> String {
    "0.1.0".to_string()
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_sig, m)?)?;
    m.add_function(wrap_pyfunction!(py_siglength, m)?)?;
    m.add_function(wrap_pyfunction!(py_sigcombine, m)?)?;
    m.add_function(wrap_pyfunction!(py_prepare, m)?)?;
    m.add_function(wrap_pyfunction!(py_basis, m)?)?;
    m.add_function(wrap_pyfunction!(py_logsig, m)?)?;
    m.add_function(wrap_pyfunction!(py_logsig_expanded, m)?)?;
    m.add_function(wrap_pyfunction!(py_logsiglength, m)?)?;
    m.add_function(wrap_pyfunction!(py_sigbackprop, m)?)?;
    m.add_function(wrap_pyfunction!(py_sigjacobian, m)?)?;
    m.add_function(wrap_pyfunction!(py_logsigbackprop, m)?)?;
    m.add_function(wrap_pyfunction!(py_sigjoin, m)?)?;
    m.add_function(wrap_pyfunction!(py_sigjoinbackprop, m)?)?;
    m.add_function(wrap_pyfunction!(py_sigscale, m)?)?;
    m.add_function(wrap_pyfunction!(py_sigscalebackprop, m)?)?;
    m.add_function(wrap_pyfunction!(py_rotinv2dprepare, m)?)?;
    m.add_function(wrap_pyfunction!(py_rotinv2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_rotinv2dlength, m)?)?;
    m.add_function(wrap_pyfunction!(py_rotinv2dcoeffs, m)?)?;
    m.add_function(wrap_pyfunction!(py_version, m)?)?;
    m.add_class::<PyPreparedData>()?;
    m.add_class::<PyRotInv2DPreparedData>()?;
    Ok(())
}
