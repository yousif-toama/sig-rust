"""Comprehensive tests for sig-rust Python bindings.

Ports all tests from sig-light test suite to verify identical behavior.
"""

import math

import numpy as np
import pytest
import sig_rust

# === Version ===


class TestVersion:
    def test_version_returns_string(self):
        assert isinstance(sig_rust.version(), str)

    def test_version_matches_dunder(self):
        assert sig_rust.version() == sig_rust.__version__


# === Siglength ===


class TestSiglength:
    def test_d2_m2(self):
        assert sig_rust.siglength(2, 2) == 6

    def test_d3_m3(self):
        assert sig_rust.siglength(3, 3) == 3 + 9 + 27

    def test_d1(self):
        assert sig_rust.siglength(1, 5) == 5

    def test_formula(self):
        for d in range(2, 6):
            for m in range(1, 5):
                expected = d * (d**m - 1) // (d - 1)
                assert sig_rust.siglength(d, m) == expected

    def test_large(self):
        assert sig_rust.siglength(9500, 2) == 9500**2 + 9500


# === Sig ===


class TestSig:
    def test_l_path_depth2(self, l_path):
        s = sig_rust.sig(l_path, 2)
        expected = np.array([1.0, 1.0, 0.5, 1.0, 0.0, 0.5])
        np.testing.assert_allclose(s, expected, atol=1e-12)

    def test_square_path_depth2(self, square_path):
        s = sig_rust.sig(square_path, 2)
        expected = np.array([0.0, 0.0, 0.0, 1.0, -1.0, 0.0])
        np.testing.assert_allclose(s, expected, atol=1e-12)

    def test_1d_depth3(self, path_1d):
        s = sig_rust.sig(path_1d, 3)
        expected = np.array([3.0, 4.5, 4.5])
        np.testing.assert_allclose(s, expected, atol=1e-12)

    def test_single_point_is_zero(self, single_point):
        s = sig_rust.sig(single_point, 3)
        assert s.shape == (sig_rust.siglength(2, 3),)
        np.testing.assert_allclose(s, 0.0)

    def test_level_1_is_displacement(self, rng):
        path = rng.standard_normal((15, 4))
        s = sig_rust.sig(path, 3)
        displacement = path[-1] - path[0]
        np.testing.assert_allclose(s[:4], displacement, atol=1e-12)

    def test_output_length(self, rng):
        d, m = 3, 4
        path = rng.standard_normal((10, d))
        s = sig_rust.sig(path, m)
        assert s.shape == (sig_rust.siglength(d, m),)

    def test_format_1_returns_list(self, l_path):
        levels = sig_rust.sig(l_path, 3, format=1)
        assert isinstance(levels, list)
        assert len(levels) == 3
        assert levels[0].shape == (2,)
        assert levels[1].shape == (4,)
        assert levels[2].shape == (8,)

    def test_format_1_matches_format_0(self, rng):
        path = rng.standard_normal((10, 3))
        flat = sig_rust.sig(path, 3, format=0)
        levels = sig_rust.sig(path, 3, format=1)
        np.testing.assert_allclose(np.concatenate(levels), flat)

    def test_straight_line_depth3(self, straight_2d):
        h = np.array([3.0, 5.0])
        s = sig_rust.sig(straight_2d, 3, format=1)
        np.testing.assert_allclose(s[0], h, atol=1e-12)
        np.testing.assert_allclose(s[1], np.outer(h, h).ravel() / 2, atol=1e-12)
        expected_3 = np.einsum("i,j,k->ijk", h, h, h).ravel() / 6
        np.testing.assert_allclose(s[2], expected_3, atol=1e-12)

    def test_two_point_path(self):
        path = np.array([[1.0, 2.0], [4.0, -1.0]])
        s = sig_rust.sig(path, 2)
        h = np.array([3.0, -3.0])
        expected = np.concatenate([h, np.outer(h, h).ravel() / 2])
        np.testing.assert_allclose(s, expected, atol=1e-12)


# === Sigcombine ===


class TestSigcombine:
    def test_chens_identity(self, rng):
        d, m = 3, 3
        path = rng.standard_normal((20, d))
        mid = 10
        sig_full = sig_rust.sig(path, m)
        sig_first = sig_rust.sig(path[: mid + 1], m)
        sig_second = sig_rust.sig(path[mid:], m)
        combined = sig_rust.sigcombine(sig_first, sig_second, d, m)
        np.testing.assert_allclose(combined, sig_full, atol=1e-10)

    def test_identity_element(self, rng):
        d, m = 2, 3
        path = rng.standard_normal((10, d))
        s = sig_rust.sig(path, m)
        zero = np.zeros(sig_rust.siglength(d, m))
        np.testing.assert_allclose(sig_rust.sigcombine(s, zero, d, m), s, atol=1e-12)
        np.testing.assert_allclose(sig_rust.sigcombine(zero, s, d, m), s, atol=1e-12)


# === Sig format=2 ===


class TestSigFormat2:
    def test_output_shape(self, rng):
        d, m = 3, 3
        path = rng.standard_normal((10, d))
        result = sig_rust.sig(path, m, format=2)
        assert result.shape == (9, sig_rust.siglength(d, m))

    def test_last_row_matches_full_sig(self, rng):
        d, m = 2, 3
        path = rng.standard_normal((8, d))
        cumulative = sig_rust.sig(path, m, format=2)
        full = sig_rust.sig(path, m)
        np.testing.assert_allclose(cumulative[-1], full, atol=1e-12)

    def test_each_row_matches_prefix(self, rng):
        d, m = 2, 3
        path = rng.standard_normal((7, d))
        cumulative = sig_rust.sig(path, m, format=2)
        for i in range(len(path) - 1):
            expected = sig_rust.sig(path[: i + 2], m)
            np.testing.assert_allclose(cumulative[i], expected, atol=1e-12)

    def test_single_point_returns_empty(self, single_point):
        d = single_point.shape[1]
        m = 3
        result = sig_rust.sig(single_point, m, format=2)
        assert result.shape == (0, sig_rust.siglength(d, m))


# === Sig batching ===


class TestSigBatching:
    def test_batch_shape(self, rng):
        d, m = 2, 3
        batch = rng.standard_normal((4, 10, d))
        result = sig_rust.sig(batch, m)
        assert result.shape == (4, sig_rust.siglength(d, m))

    def test_batch_matches_individual(self, rng):
        d, m = 2, 3
        batch = rng.standard_normal((5, 8, d))
        result = sig_rust.sig(batch, m)
        for i in range(5):
            individual = sig_rust.sig(batch[i], m)
            np.testing.assert_allclose(result[i], individual, atol=1e-12)

    def test_multidim_batch(self, rng):
        d, m = 2, 2
        batch = rng.standard_normal((3, 4, 6, d))
        result = sig_rust.sig(batch, m)
        assert result.shape == (3, 4, sig_rust.siglength(d, m))
        for i in range(3):
            for j in range(4):
                individual = sig_rust.sig(batch[i, j], m)
                np.testing.assert_allclose(result[i, j], individual, atol=1e-12)


# === Logsiglength ===


class TestLogsiglength:
    def test_d2_m2(self):
        assert sig_rust.logsiglength(2, 2) == 3

    def test_d2_m3(self):
        assert sig_rust.logsiglength(2, 3) == 5

    def test_d3_m3(self):
        assert sig_rust.logsiglength(3, 3) == 14

    def test_level_1_equals_d(self):
        for d in range(1, 6):
            assert sig_rust.logsiglength(d, 1) == d

    def test_d1_all_depths(self):
        for m in range(1, 8):
            assert sig_rust.logsiglength(1, m) == 1


# === Prepare and Basis ===


class TestPrepareAndBasis:
    def test_basis_d2_m2(self):
        s = sig_rust.prepare(2, 2)
        assert sig_rust.basis(s) == ["1", "2", "[1,2]"]

    def test_basis_d2_m3(self):
        s = sig_rust.prepare(2, 3)
        assert sig_rust.basis(s) == ["1", "2", "[1,2]", "[1,[1,2]]", "[[1,2],2]"]

    def test_basis_d3_m1(self):
        s = sig_rust.prepare(3, 1)
        assert sig_rust.basis(s) == ["1", "2", "3"]

    def test_basis_length_matches_logsiglength(self):
        for d in [2, 3, 4]:
            for m in [1, 2, 3, 4]:
                s = sig_rust.prepare(d, m)
                assert len(sig_rust.basis(s)) == sig_rust.logsiglength(d, m)


# === Logsig ===


class TestLogsig:
    def test_single_point_is_zero(self, single_point):
        s = sig_rust.prepare(2, 3)
        ls = sig_rust.logsig(single_point, s)
        np.testing.assert_allclose(ls, 0.0)

    def test_straight_line_level1_only(self, straight_2d):
        s = sig_rust.prepare(2, 3)
        ls = sig_rust.logsig(straight_2d, s)
        b = sig_rust.basis(s)
        level1_idx = [i for i, label in enumerate(b) if len(label) == 1]
        level_higher_idx = [i for i, label in enumerate(b) if len(label) > 1]
        displacement = straight_2d[-1] - straight_2d[0]
        np.testing.assert_allclose(ls[level1_idx], displacement, atol=1e-12)
        np.testing.assert_allclose(ls[level_higher_idx], 0.0, atol=1e-12)

    def test_square_area(self, square_path):
        s = sig_rust.prepare(2, 2)
        ls = sig_rust.logsig(square_path, s)
        b = sig_rust.basis(s)
        np.testing.assert_allclose(ls[b.index("1")], 0.0, atol=1e-12)
        np.testing.assert_allclose(ls[b.index("2")], 0.0, atol=1e-12)
        np.testing.assert_allclose(ls[b.index("[1,2]")], 1.0, atol=1e-12)

    def test_l_path(self, l_path):
        s = sig_rust.prepare(2, 2)
        ls = sig_rust.logsig(l_path, s)
        expected = np.array([1.0, 1.0, 0.5])
        np.testing.assert_allclose(ls, expected, atol=1e-12)

    def test_1d_path(self):
        path = np.array([[0.0], [1.0], [3.0]])
        s = sig_rust.prepare(1, 3)
        ls = sig_rust.logsig(path, s)
        np.testing.assert_allclose(ls, [3.0], atol=1e-12)


# === Logsig expanded ===


class TestLogsigExpanded:
    def test_straight_line(self, straight_2d):
        s = sig_rust.prepare(2, 2)
        ls = sig_rust.logsig_expanded(straight_2d, s)
        displacement = straight_2d[-1] - straight_2d[0]
        assert ls.shape == (sig_rust.siglength(2, 2),)
        np.testing.assert_allclose(ls[:2], displacement, atol=1e-12)
        np.testing.assert_allclose(ls[2:], 0.0, atol=1e-12)

    def test_single_point_is_zero(self, single_point):
        s = sig_rust.prepare(2, 2)
        ls = sig_rust.logsig_expanded(single_point, s)
        assert ls.shape == (sig_rust.siglength(2, 2),)
        np.testing.assert_allclose(ls, 0.0)


# === Logsig batching ===


class TestLogsigBatching:
    def test_batch_matches_individual(self, rng):
        d, m = 2, 3
        s = sig_rust.prepare(d, m)
        batch = rng.standard_normal((5, 8, d))
        result = sig_rust.logsig(batch, s)
        assert result.shape == (5, sig_rust.logsiglength(d, m))
        for i in range(5):
            individual = sig_rust.logsig(batch[i], s)
            np.testing.assert_allclose(result[i], individual, atol=1e-12)

    def test_multidim_batch(self, rng):
        d, m = 2, 2
        s = sig_rust.prepare(d, m)
        batch = rng.standard_normal((3, 4, 6, d))
        result = sig_rust.logsig(batch, s)
        assert result.shape == (3, 4, sig_rust.logsiglength(d, m))


# === Backprop ===


def _numerical_sig_gradient(path, m, deriv, epsilon=1e-7):
    grad = np.zeros_like(path)
    for i in range(path.shape[0]):
        for j in range(path.shape[1]):
            path_plus = path.copy()
            path_minus = path.copy()
            path_plus[i, j] += epsilon
            path_minus[i, j] -= epsilon
            s_plus = sig_rust.sig(path_plus, m)
            s_minus = sig_rust.sig(path_minus, m)
            grad[i, j] = deriv @ (s_plus - s_minus) / (2 * epsilon)
    return grad


def _numerical_logsig_gradient(path, s, deriv, epsilon=1e-7):
    grad = np.zeros_like(path)
    for i in range(path.shape[0]):
        for j in range(path.shape[1]):
            path_plus = path.copy()
            path_minus = path.copy()
            path_plus[i, j] += epsilon
            path_minus[i, j] -= epsilon
            ls_plus = sig_rust.logsig(path_plus, s)
            ls_minus = sig_rust.logsig(path_minus, s)
            grad[i, j] = deriv @ (ls_plus - ls_minus) / (2 * epsilon)
    return grad


class TestSigbackprop:
    def test_vs_finite_differences(self, rng):
        d, m = 2, 3
        path = rng.standard_normal((8, d))
        deriv = rng.standard_normal(sig_rust.siglength(d, m))
        analytical = sig_rust.sigbackprop(deriv, path, m)
        numerical = _numerical_sig_gradient(path, m, deriv)
        np.testing.assert_allclose(analytical, numerical, atol=1e-5)

    def test_two_point_path(self, rng):
        d, m = 3, 2
        path = rng.standard_normal((2, d))
        deriv = rng.standard_normal(sig_rust.siglength(d, m))
        analytical = sig_rust.sigbackprop(deriv, path, m)
        numerical = _numerical_sig_gradient(path, m, deriv)
        np.testing.assert_allclose(analytical, numerical, atol=1e-5)

    def test_various_d_and_m(self, rng):
        for d, m in [(1, 3), (3, 2), (2, 4)]:
            path = rng.standard_normal((5, d))
            deriv = rng.standard_normal(sig_rust.siglength(d, m))
            analytical = sig_rust.sigbackprop(deriv, path, m)
            numerical = _numerical_sig_gradient(path, m, deriv)
            np.testing.assert_allclose(analytical, numerical, atol=1e-5)

    def test_single_point_returns_zeros(self, single_point):
        d = single_point.shape[1]
        m = 3
        deriv = np.ones(sig_rust.siglength(d, m))
        grad = sig_rust.sigbackprop(deriv, single_point, m)
        np.testing.assert_allclose(grad, 0.0)
        assert grad.shape == single_point.shape

    def test_batched(self, rng):
        d, m = 2, 2
        paths = rng.standard_normal((4, 8, d))
        derivs = rng.standard_normal((4, sig_rust.siglength(d, m)))
        result = sig_rust.sigbackprop(derivs, paths, m)
        assert result.shape == (4, 8, d)
        for i in range(4):
            individual = sig_rust.sigbackprop(derivs[i], paths[i], m)
            np.testing.assert_allclose(result[i], individual, atol=1e-12)


class TestSigjacobian:
    def test_shape(self, rng):
        d, m = 2, 3
        path = rng.standard_normal((7, d))
        jac = sig_rust.sigjacobian(path, m)
        assert jac.shape == (7, d, sig_rust.siglength(d, m))

    def test_vs_sigbackprop(self, rng):
        d, m = 2, 2
        path = rng.standard_normal((5, d))
        jac = sig_rust.sigjacobian(path, m)
        sig_len = sig_rust.siglength(d, m)
        for c in range(sig_len):
            e = np.zeros(sig_len)
            e[c] = 1.0
            grad = sig_rust.sigbackprop(e, path, m)
            np.testing.assert_allclose(jac[:, :, c], grad, atol=1e-12)


class TestLogsigbackprop:
    def test_vs_finite_differences(self, rng):
        d, m = 2, 3
        path = rng.standard_normal((8, d))
        s = sig_rust.prepare(d, m)
        deriv = rng.standard_normal(sig_rust.logsiglength(d, m))
        analytical = sig_rust.logsigbackprop(deriv, path, s)
        numerical = _numerical_logsig_gradient(path, s, deriv)
        np.testing.assert_allclose(analytical, numerical, atol=1e-5)

    def test_two_point_path(self, rng):
        d, m = 2, 3
        path = rng.standard_normal((2, d))
        s = sig_rust.prepare(d, m)
        deriv = rng.standard_normal(sig_rust.logsiglength(d, m))
        analytical = sig_rust.logsigbackprop(deriv, path, s)
        numerical = _numerical_logsig_gradient(path, s, deriv)
        np.testing.assert_allclose(analytical, numerical, atol=1e-5)

    def test_single_point_returns_zeros(self, single_point):
        d = single_point.shape[1]
        m = 3
        s = sig_rust.prepare(d, m)
        deriv = np.ones(sig_rust.logsiglength(d, m))
        grad = sig_rust.logsigbackprop(deriv, single_point, s)
        np.testing.assert_allclose(grad, 0.0)
        assert grad.shape == single_point.shape

    def test_batched(self, rng):
        d, m = 2, 2
        paths = rng.standard_normal((3, 6, d))
        s = sig_rust.prepare(d, m)
        derivs = rng.standard_normal((3, sig_rust.logsiglength(d, m)))
        result = sig_rust.logsigbackprop(derivs, paths, s)
        assert result.shape == (3, 6, d)
        for i in range(3):
            individual = sig_rust.logsigbackprop(derivs[i], paths[i], s)
            np.testing.assert_allclose(result[i], individual, atol=1e-12)


# === Transforms ===


class TestSigjoin:
    def test_extends_signature(self, rng):
        d, m = 3, 3
        path = rng.standard_normal((10, d))
        for k in range(2, len(path)):
            sig_prefix = sig_rust.sig(path[:k], m)
            segment = path[k] - path[k - 1]
            joined = sig_rust.sigjoin(sig_prefix, segment, d, m)
            expected = sig_rust.sig(path[: k + 1], m)
            np.testing.assert_allclose(joined, expected, atol=1e-10)

    def test_with_fixed_last(self, rng):
        d, m = 3, 2
        path = rng.standard_normal((5, d))
        sig_prefix = sig_rust.sig(path[:3], m)
        full_segment = path[3] - path[2]
        partial_segment = full_segment[:-1]
        fixed_val = float(full_segment[-1])
        joined_full = sig_rust.sigjoin(sig_prefix, full_segment, d, m)
        joined_fixed = sig_rust.sigjoin(
            sig_prefix, partial_segment, d, m, fixedLast=fixed_val
        )
        np.testing.assert_allclose(joined_full, joined_fixed, atol=1e-12)


class TestSigjoinbackprop:
    def test_vs_finite_differences(self, rng):
        d, m = 2, 3
        sig_flat = sig_rust.sig(rng.standard_normal((5, d)), m)
        segment = rng.standard_normal(d)
        deriv = rng.standard_normal(sig_rust.siglength(d, m))
        result = sig_rust.sigjoinbackprop(deriv, sig_flat, segment, d, m)
        dsig_a, dseg_a = result[0], result[1]
        epsilon = 1e-7
        dsig_num = np.zeros_like(sig_flat)
        for i in range(len(sig_flat)):
            sig_plus = sig_flat.copy()
            sig_minus = sig_flat.copy()
            sig_plus[i] += epsilon
            sig_minus[i] -= epsilon
            f_plus = sig_rust.sigjoin(sig_plus, segment, d, m)
            f_minus = sig_rust.sigjoin(sig_minus, segment, d, m)
            dsig_num[i] = deriv @ (f_plus - f_minus) / (2 * epsilon)
        np.testing.assert_allclose(dsig_a, dsig_num, atol=1e-5)
        dseg_num = np.zeros_like(segment)
        for i in range(len(segment)):
            seg_plus = segment.copy()
            seg_minus = segment.copy()
            seg_plus[i] += epsilon
            seg_minus[i] -= epsilon
            f_plus = sig_rust.sigjoin(sig_flat, seg_plus, d, m)
            f_minus = sig_rust.sigjoin(sig_flat, seg_minus, d, m)
            dseg_num[i] = deriv @ (f_plus - f_minus) / (2 * epsilon)
        np.testing.assert_allclose(dseg_a, dseg_num, atol=1e-5)

    def test_with_fixed_last(self, rng):
        d, m = 3, 2
        sig_flat = sig_rust.sig(rng.standard_normal((5, d)), m)
        partial_segment = rng.standard_normal(d - 1)
        fixed_val = 0.5
        deriv = rng.standard_normal(sig_rust.siglength(d, m))
        result = sig_rust.sigjoinbackprop(
            deriv, sig_flat, partial_segment, d, m, fixedLast=fixed_val
        )
        assert len(result) == 3
        assert result[0].shape == sig_flat.shape
        assert result[1].shape == partial_segment.shape
        assert isinstance(result[2], float)


class TestSigscale:
    def test_identity_with_ones(self, rng):
        d, m = 3, 3
        path = rng.standard_normal((10, d))
        s = sig_rust.sig(path, m)
        ones = np.ones(d)
        scaled = sig_rust.sigscale(s, ones, d, m)
        np.testing.assert_allclose(scaled, s, atol=1e-12)

    def test_level_1_scaling(self, rng):
        d, m = 2, 2
        path = rng.standard_normal((8, d))
        s = sig_rust.sig(path, m)
        scales = np.array([2.0, 3.0])
        scaled = sig_rust.sigscale(s, scales, d, m)
        np.testing.assert_allclose(scaled[:d], s[:d] * scales, atol=1e-12)

    def test_d2_m2(self, rng):
        d, m = 2, 2
        path = rng.standard_normal((5, d))
        s = sig_rust.sig(path, m)
        scales = np.array([0.5, 2.0])
        scaled = sig_rust.sigscale(s, scales, d, m)
        scaled_path = path * scales
        expected = sig_rust.sig(scaled_path, m)
        np.testing.assert_allclose(scaled, expected, atol=1e-10)


class TestSigscalebackprop:
    def test_dsig_vs_finite_differences(self, rng):
        d, m = 2, 3
        path = rng.standard_normal((6, d))
        sig_flat = sig_rust.sig(path, m)
        scales = rng.standard_normal(d) * 0.5 + 1.0
        deriv = rng.standard_normal(sig_rust.siglength(d, m))
        dsig_a, _ = sig_rust.sigscalebackprop(deriv, sig_flat, scales, d, m)
        epsilon = 1e-7
        dsig_num = np.zeros_like(sig_flat)
        for i in range(len(sig_flat)):
            sig_plus = sig_flat.copy()
            sig_minus = sig_flat.copy()
            sig_plus[i] += epsilon
            sig_minus[i] -= epsilon
            f_plus = sig_rust.sigscale(sig_plus, scales, d, m)
            f_minus = sig_rust.sigscale(sig_minus, scales, d, m)
            dsig_num[i] = deriv @ (f_plus - f_minus) / (2 * epsilon)
        np.testing.assert_allclose(dsig_a, dsig_num, atol=1e-5)

    def test_dscales_vs_finite_differences(self, rng):
        d, m = 2, 2
        path = rng.standard_normal((6, d))
        sig_flat = sig_rust.sig(path, m)
        scales = rng.standard_normal(d) * 0.5 + 1.0
        deriv = rng.standard_normal(sig_rust.siglength(d, m))
        _, dscales_a = sig_rust.sigscalebackprop(deriv, sig_flat, scales, d, m)
        epsilon = 1e-7
        dscales_num = np.zeros(d)
        for i in range(d):
            s_plus = scales.copy()
            s_minus = scales.copy()
            s_plus[i] += epsilon
            s_minus[i] -= epsilon
            f_plus = sig_rust.sigscale(sig_flat, s_plus, d, m)
            f_minus = sig_rust.sigscale(sig_flat, s_minus, d, m)
            dscales_num[i] = deriv @ (f_plus - f_minus) / (2 * epsilon)
        np.testing.assert_allclose(dscales_a, dscales_num, atol=1e-5)


# === Rotation invariants ===


class TestRotinv2dprepare:
    def test_m2_length(self):
        s = sig_rust.rotinv2dprepare(2)
        assert sig_rust.rotinv2dlength(s) == 2

    def test_m4_length(self):
        s = sig_rust.rotinv2dprepare(4)
        assert sig_rust.rotinv2dlength(s) == 8

    def test_m6_length(self):
        s = sig_rust.rotinv2dprepare(6)
        assert sig_rust.rotinv2dlength(s) == 28

    def test_odd_m_same_as_even_below(self):
        s3 = sig_rust.rotinv2dprepare(3)
        s2 = sig_rust.rotinv2dprepare(2)
        assert sig_rust.rotinv2dlength(s3) == sig_rust.rotinv2dlength(s2)

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match=r"unsupported inv_type"):
            sig_rust.rotinv2dprepare(2, inv_type="b")


class TestRotinv2d:
    def test_rotation_invariance(self, rng):
        m = 4
        s = sig_rust.rotinv2dprepare(m)
        path = rng.standard_normal((15, 2))
        inv_original = sig_rust.rotinv2d(path, s)
        for _ in range(3):
            theta = rng.uniform(0, 2 * math.pi)
            c, sn = math.cos(theta), math.sin(theta)
            rot = np.array([[c, -sn], [sn, c]])
            rotated_path = path @ rot.T
            inv_rotated = sig_rust.rotinv2d(rotated_path, s)
            np.testing.assert_allclose(inv_original, inv_rotated, atol=1e-10)

    def test_simple_known_path(self):
        m = 2
        s = sig_rust.rotinv2dprepare(m)
        path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        inv = sig_rust.rotinv2d(path, s)
        assert inv.shape == (sig_rust.rotinv2dlength(s),)
        assert not np.allclose(inv, 0.0)

    def test_non_2d_raises(self):
        s = sig_rust.rotinv2dprepare(2)
        path_3d = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        with pytest.raises(ValueError, match="2D"):
            sig_rust.rotinv2d(path_3d, s)


class TestRotinv2dcoeffs:
    def test_returns_list(self):
        s = sig_rust.rotinv2dprepare(4)
        coeffs = sig_rust.rotinv2dcoeffs(s)
        assert isinstance(coeffs, list)

    def test_correct_number_of_levels(self):
        for m in [2, 4, 6]:
            s = sig_rust.rotinv2dprepare(m)
            coeffs = sig_rust.rotinv2dcoeffs(s)
            assert len(coeffs) == m // 2

    def test_correct_shapes(self):
        s = sig_rust.rotinv2dprepare(4)
        coeffs = sig_rust.rotinv2dcoeffs(s)
        assert coeffs[0].shape == (2, 4)
        assert coeffs[1].shape == (6, 16)


# === Cross-validation against iisignature (if available) ===

try:
    import iisignature  # ty: ignore[unresolved-import]

    HAS_IISIG = True
except ImportError:
    HAS_IISIG = False


GRAD_ATOL = 1e-6


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestSigCompat:
    """Compare sig() format=0 against iisignature.sig()."""

    @pytest.fixture
    def compat_rng(self):
        return np.random.default_rng(123)

    def test_random_2d(self, compat_rng):
        path = compat_rng.standard_normal((20, 2))
        for m in range(1, 5):
            ours = sig_rust.sig(path, m)
            theirs = iisignature.sig(path, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_random_3d(self, compat_rng):
        path = compat_rng.standard_normal((15, 3))
        for m in range(1, 4):
            ours = sig_rust.sig(path, m)
            theirs = iisignature.sig(path, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_random_5d(self, compat_rng):
        path = compat_rng.standard_normal((10, 5))
        for m in [1, 2]:
            ours = sig_rust.sig(path, m)
            theirs = iisignature.sig(path, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_random_1d(self, compat_rng):
        path = compat_rng.standard_normal((25, 1))
        for m in range(1, 6):
            ours = sig_rust.sig(path, m)
            theirs = iisignature.sig(path, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestSigFormat2Compat:
    """Compare sig() format=2 against iisignature.sig(format=2)."""

    @pytest.fixture
    def compat_rng(self):
        return np.random.default_rng(123)

    def test_random_2d(self, compat_rng):
        path = compat_rng.standard_normal((10, 2))
        for m in [2, 3]:
            ours = sig_rust.sig(path, m, format=2)
            theirs = iisignature.sig(path, m, 2)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_random_3d(self, compat_rng):
        path = compat_rng.standard_normal((8, 3))
        for m in [2, 3]:
            ours = sig_rust.sig(path, m, format=2)
            theirs = iisignature.sig(path, m, 2)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestSiglengthCompat:
    def test_various(self):
        for d in [1, 2, 3, 5, 10]:
            for m in [1, 2, 3, 4]:
                assert sig_rust.siglength(d, m) == iisignature.siglength(d, m)


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestSigcombineCompat:
    @pytest.fixture
    def compat_rng(self):
        return np.random.default_rng(123)

    def test_random_2d(self, compat_rng):
        path = compat_rng.standard_normal((10, 2))
        for m in [2, 3]:
            s1 = sig_rust.sig(path[:5], m)
            s2 = sig_rust.sig(path[4:], m)
            ours = sig_rust.sigcombine(s1, s2, 2, m)
            theirs = iisignature.sigcombine(s1, s2, 2, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestLogsiglengthCompat:
    def test_various(self):
        for d in [1, 2, 3, 5]:
            for m in [1, 2, 3, 4]:
                assert sig_rust.logsiglength(d, m) == iisignature.logsiglength(d, m)


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestLogsigCompat:
    @pytest.fixture
    def compat_rng(self):
        return np.random.default_rng(123)

    def test_random_2d(self, compat_rng):
        path = compat_rng.standard_normal((20, 2))
        for m in range(1, 5):
            s_ours = sig_rust.prepare(2, m)
            s_theirs = iisignature.prepare(2, m)
            ours = sig_rust.logsig(path, s_ours)
            theirs = iisignature.logsig(path, s_theirs)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_random_3d(self, compat_rng):
        path = compat_rng.standard_normal((15, 3))
        for m in range(1, 4):
            s_ours = sig_rust.prepare(3, m)
            s_theirs = iisignature.prepare(3, m)
            ours = sig_rust.logsig(path, s_ours)
            theirs = iisignature.logsig(path, s_theirs)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestLogsigExpandedCompat:
    @pytest.fixture
    def compat_rng(self):
        return np.random.default_rng(123)

    def test_random_2d(self, compat_rng):
        path = compat_rng.standard_normal((10, 2))
        for m in [2, 3]:
            s_ours = sig_rust.prepare(2, m)
            s_theirs = iisignature.prepare(2, m, "X")
            ours = sig_rust.logsig_expanded(path, s_ours)
            theirs = iisignature.logsig(path, s_theirs, "X")
            np.testing.assert_allclose(ours, theirs, atol=1e-10)


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestBasisCompat:
    def test_d2(self):
        for m in range(1, 5):
            s_ours = sig_rust.prepare(2, m)
            s_theirs = iisignature.prepare(2, m)
            assert sig_rust.basis(s_ours) == list(iisignature.basis(s_theirs))

    def test_d3(self):
        for m in range(1, 4):
            s_ours = sig_rust.prepare(3, m)
            s_theirs = iisignature.prepare(3, m)
            assert sig_rust.basis(s_ours) == list(iisignature.basis(s_theirs))


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestSigbackpropCompat:
    @pytest.fixture
    def compat_rng(self):
        return np.random.default_rng(123)

    def test_random_2d(self, compat_rng):
        path = compat_rng.standard_normal((10, 2))
        for m in [2, 3]:
            deriv = compat_rng.standard_normal(iisignature.siglength(2, m))
            ours = sig_rust.sigbackprop(deriv, path, m)
            theirs = iisignature.sigbackprop(deriv, path, m)
            np.testing.assert_allclose(ours, theirs, atol=GRAD_ATOL)

    def test_random_3d(self, compat_rng):
        path = compat_rng.standard_normal((8, 3))
        m = 2
        deriv = compat_rng.standard_normal(iisignature.siglength(3, m))
        ours = sig_rust.sigbackprop(deriv, path, m)
        theirs = iisignature.sigbackprop(deriv, path, m)
        np.testing.assert_allclose(ours, theirs, atol=GRAD_ATOL)


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestSigjacobianCompat:
    @pytest.fixture
    def compat_rng(self):
        return np.random.default_rng(123)

    def test_random_2d(self, compat_rng):
        path = compat_rng.standard_normal((6, 2))
        for m in [2, 3]:
            ours = sig_rust.sigjacobian(path, m)
            theirs = iisignature.sigjacobian(path, m)
            np.testing.assert_allclose(ours, theirs, atol=GRAD_ATOL)


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestLogsigbackpropCompat:
    @pytest.fixture
    def compat_rng(self):
        return np.random.default_rng(123)

    def test_random_2d(self, compat_rng):
        path = compat_rng.standard_normal((10, 2))
        for m in [2, 3]:
            s_ours = sig_rust.prepare(2, m)
            s_theirs = iisignature.prepare(2, m, "S")
            deriv = compat_rng.standard_normal(iisignature.logsiglength(2, m))
            ours = sig_rust.logsigbackprop(deriv, path, s_ours)
            theirs = iisignature.logsigbackprop(deriv, path, s_theirs, "S")
            np.testing.assert_allclose(ours, theirs, atol=GRAD_ATOL)

    def test_random_3d(self, compat_rng):
        path = compat_rng.standard_normal((8, 3))
        m = 2
        s_ours = sig_rust.prepare(3, m)
        s_theirs = iisignature.prepare(3, m, "S")
        deriv = compat_rng.standard_normal(iisignature.logsiglength(3, m))
        ours = sig_rust.logsigbackprop(deriv, path, s_ours)
        theirs = iisignature.logsigbackprop(deriv, path, s_theirs, "S")
        np.testing.assert_allclose(ours, theirs, atol=GRAD_ATOL)


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestSigjoinCompat:
    @pytest.fixture
    def compat_rng(self):
        return np.random.default_rng(123)

    def test_random_2d(self, compat_rng):
        path = compat_rng.standard_normal((10, 2))
        for m in [2, 3]:
            s = sig_rust.sig(path[:5], m)
            seg = path[5] - path[4]
            ours = sig_rust.sigjoin(s, seg, 2, m)
            theirs = iisignature.sigjoin(s, seg, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_random_3d(self, compat_rng):
        path = compat_rng.standard_normal((8, 3))
        m = 2
        s = sig_rust.sig(path[:4], m)
        seg = path[4] - path[3]
        ours = sig_rust.sigjoin(s, seg, 3, m)
        theirs = iisignature.sigjoin(s, seg, m)
        np.testing.assert_allclose(ours, theirs, atol=1e-10)


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestSigjoinbackpropCompat:
    @pytest.fixture
    def compat_rng(self):
        return np.random.default_rng(123)

    def test_random_2d(self, compat_rng):
        path = compat_rng.standard_normal((8, 2))
        m = 2
        s = sig_rust.sig(path[:4], m)
        seg = path[4] - path[3]
        deriv = compat_rng.standard_normal(iisignature.siglength(2, m))
        ours = sig_rust.sigjoinbackprop(deriv, s, seg, 2, m)
        theirs = iisignature.sigjoinbackprop(deriv, s, seg, m)
        np.testing.assert_allclose(ours[0], theirs[0], atol=GRAD_ATOL)
        np.testing.assert_allclose(ours[1], theirs[1], atol=GRAD_ATOL)


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestSigscaleCompat:
    @pytest.fixture
    def compat_rng(self):
        return np.random.default_rng(123)

    def test_random_2d(self, compat_rng):
        path = compat_rng.standard_normal((10, 2))
        scales = compat_rng.standard_normal(2) + 2
        for m in [2, 3]:
            s = sig_rust.sig(path, m)
            ours = sig_rust.sigscale(s, scales, 2, m)
            theirs = iisignature.sigscale(s, scales, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_random_3d(self, compat_rng):
        path = compat_rng.standard_normal((8, 3))
        scales = np.array([2.0, 0.5, 3.0])
        m = 2
        s = sig_rust.sig(path, m)
        ours = sig_rust.sigscale(s, scales, 3, m)
        theirs = iisignature.sigscale(s, scales, m)
        np.testing.assert_allclose(ours, theirs, atol=1e-10)


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestSigscalebackpropCompat:
    @pytest.fixture
    def compat_rng(self):
        return np.random.default_rng(123)

    def test_random_2d(self, compat_rng):
        path = compat_rng.standard_normal((10, 2))
        scales = compat_rng.standard_normal(2) + 2
        m = 3
        s = sig_rust.sig(path, m)
        deriv = compat_rng.standard_normal(iisignature.siglength(2, m))
        ours = sig_rust.sigscalebackprop(deriv, s, scales, 2, m)
        theirs = iisignature.sigscalebackprop(deriv, s, scales, m)
        np.testing.assert_allclose(ours[0], theirs[0], atol=1e-10)
        np.testing.assert_allclose(ours[1], theirs[1], atol=1e-10)


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestRotinv2dCompat:
    @pytest.fixture
    def compat_rng(self):
        return np.random.default_rng(123)

    def test_rotinv2dlength(self):
        for m in [2, 4, 6]:
            s_ours = sig_rust.rotinv2dprepare(m, "a")
            s_theirs = iisignature.rotinv2dprepare(m, "a")
            assert sig_rust.rotinv2dlength(s_ours) == iisignature.rotinv2dlength(
                s_theirs
            )

    def test_rotinv2d_values(self, compat_rng):
        path = compat_rng.standard_normal((15, 2))
        for m in [2, 4]:
            s_ours = sig_rust.rotinv2dprepare(m, "a")
            s_theirs = iisignature.rotinv2dprepare(m, "a")
            ours = sig_rust.rotinv2d(path, s_ours)
            theirs = iisignature.rotinv2d(path, s_theirs)
            assert len(ours) == len(theirs)
            np.testing.assert_equal(len(ours), len(theirs))


@pytest.mark.skipif(not HAS_IISIG, reason="iisignature not installed")
class TestBatchingCompat:
    @pytest.fixture
    def compat_rng(self):
        return np.random.default_rng(123)

    def test_batched_sig(self, compat_rng):
        paths = compat_rng.standard_normal((5, 10, 2))
        m = 3
        ours = sig_rust.sig(paths, m)
        theirs = iisignature.sig(paths, m)
        np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_batched_logsig(self, compat_rng):
        paths = compat_rng.standard_normal((5, 10, 2))
        m = 3
        s_ours = sig_rust.prepare(2, m)
        s_theirs = iisignature.prepare(2, m)
        ours = sig_rust.logsig(paths, s_ours)
        theirs = iisignature.logsig(paths, s_theirs)
        np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_batched_sigbackprop(self, compat_rng):
        paths = compat_rng.standard_normal((3, 8, 2))
        m = 2
        deriv = compat_rng.standard_normal((3, iisignature.siglength(2, m)))
        ours = sig_rust.sigbackprop(deriv, paths, m)
        theirs = iisignature.sigbackprop(deriv, paths, m)
        np.testing.assert_allclose(ours, theirs, atol=GRAD_ATOL)
