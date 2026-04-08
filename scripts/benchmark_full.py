"""Comprehensive benchmarks: sig-rust vs sig-light (Python) vs iisignature (C++).

Tests short paths, long paths, and batched operations to evaluate
single-threaded performance and multithreading characteristics.

Run with: uv run python scripts/benchmark_full.py
"""

import timeit
from collections.abc import Callable
from typing import Any

import numpy as np
import sig_rust

try:
    import iisignature  # type: ignore[import-untyped]
except ImportError:
    iisignature = None

try:
    import sig_light
except ImportError:
    sig_light = None


REPEATS = 30


def _time(fn: Callable[[], Any], repeats: int = REPEATS) -> float:
    """Return average time in seconds."""
    return timeit.timeit(fn, number=repeats) / repeats


def _fmt_ms(t: float | None) -> str:
    if t is None:
        return "-"
    return f"{t * 1000:.3f}ms"


def _fmt_speedup(t_this: float, t_ref: float | None) -> str:
    if t_ref is None or t_ref <= 0:
        return "-"
    ratio = t_ref / t_this
    marker = "+" if ratio >= 1 else ""
    return f"{marker}{ratio:.2f}x"


def _print_header_3() -> None:
    print(
        f"  {'Config':<30} {'sig-rust':>12} {'sig-light':>12}"
        f" {'C++ (iisig)':>12} {'rust/light':>11}"
        f" {'rust/C++':>11}"
    )
    print(f"  {'-' * 92}")


def _print_row_3(
    label: str,
    t_rust: float,
    t_light: float | None,
    t_cpp: float | None,
) -> None:
    print(
        f"  {label:<30}"
        f" {_fmt_ms(t_rust):>12}"
        f" {_fmt_ms(t_light):>12}"
        f" {_fmt_ms(t_cpp):>12}"
        f" {_fmt_speedup(t_rust, t_light):>11}"
        f" {_fmt_speedup(t_rust, t_cpp):>11}"
    )


# ── Single-path configs ────────────────────────────────────────────

SHORT_CONFIGS = [
    ("d=2, m=3, n=10", 2, 3, 10),
    ("d=2, m=5, n=10", 2, 5, 10),
    ("d=3, m=3, n=20", 3, 3, 20),
    ("d=3, m=4, n=20", 3, 4, 20),
    ("d=5, m=3, n=20", 5, 3, 20),
]

MEDIUM_CONFIGS = [
    ("d=2, m=5, n=100", 2, 5, 100),
    ("d=3, m=4, n=100", 3, 4, 100),
    ("d=5, m=3, n=100", 5, 3, 100),
    ("d=3, m=5, n=100", 3, 5, 100),
    ("d=2, m=6, n=100", 2, 6, 100),
]

LONG_CONFIGS = [
    ("d=2, m=5, n=1000", 2, 5, 1000),
    ("d=3, m=4, n=1000", 3, 4, 1000),
    ("d=5, m=3, n=1000", 5, 3, 1000),
    ("d=3, m=3, n=5000", 3, 3, 5000),
    ("d=2, m=4, n=10000", 2, 4, 10000),
]

# ── Batch configs ──────────────────────────────────────────────────

BATCH_CONFIGS = [
    ("B=10, d=3, m=4, n=50", 10, 3, 4, 50),
    ("B=50, d=3, m=4, n=50", 50, 3, 4, 50),
    ("B=100, d=3, m=4, n=50", 100, 3, 4, 50),
    ("B=200, d=3, m=4, n=50", 200, 3, 4, 50),
    ("B=10, d=2, m=5, n=100", 10, 2, 5, 100),
    ("B=50, d=2, m=5, n=100", 50, 2, 5, 100),
    ("B=100, d=2, m=5, n=100", 100, 2, 5, 100),
    ("B=10, d=5, m=3, n=100", 10, 5, 3, 100),
    ("B=50, d=5, m=3, n=100", 50, 5, 3, 100),
]


def _loop_batch(fn: Callable[..., Any], items: np.ndarray, *args: Any) -> None:
    for item in items:
        fn(item, *args)


def _loop_batch2(
    fn: Callable[..., Any],
    a: np.ndarray,
    b: np.ndarray,
    *args: Any,
) -> None:
    for x, y in zip(a, b):
        fn(x, y, *args)


# ── Benchmark functions ───────────────────────────────────────────


def bench_sig_single() -> None:
    """sig() on single paths: short, medium, long."""
    rng = np.random.default_rng(42)

    for section, configs in [
        ("Short paths", SHORT_CONFIGS),
        ("Medium paths", MEDIUM_CONFIGS),
        ("Long paths", LONG_CONFIGS),
    ]:
        print(f"\n  --- {section} ---")
        _print_header_3()
        for label, d, m, n in configs:
            path = rng.standard_normal((n, d))
            t_rust = _time(lambda p=path, m_=m: sig_rust.sig(p, m_))
            t_light = None
            if sig_light is not None:
                t_light = _time(lambda p=path, m_=m: sig_light.sig(p, m_))
            t_cpp = None
            if iisignature is not None:
                t_cpp = _time(lambda p=path, m_=m: iisignature.sig(p, m_))
            _print_row_3(label, t_rust, t_light, t_cpp)


def bench_logsig_single() -> None:
    """logsig() on single paths."""
    rng = np.random.default_rng(42)

    for section, configs in [
        ("Short paths", SHORT_CONFIGS),
        ("Medium paths", MEDIUM_CONFIGS),
        ("Long paths", LONG_CONFIGS[:3]),
    ]:
        print(f"\n  --- {section} ---")
        _print_header_3()
        for label, d, m, n in configs:
            path = rng.standard_normal((n, d))
            s_rust = sig_rust.prepare(d, m)
            t_rust = _time(lambda p=path, s=s_rust: sig_rust.logsig(p, s))
            t_light = None
            if sig_light is not None:
                s_light = sig_light.prepare(d, m)
                t_light = _time(lambda p=path, s=s_light: sig_light.logsig(p, s))
            t_cpp = None
            if iisignature is not None:
                s_cpp = iisignature.prepare(d, m)
                t_cpp = _time(lambda p=path, s=s_cpp: iisignature.logsig(p, s))
            _print_row_3(label, t_rust, t_light, t_cpp)


def bench_sigbackprop_single() -> None:
    """sigbackprop() on single paths."""
    rng = np.random.default_rng(42)

    for section, configs in [
        ("Short paths", SHORT_CONFIGS),
        ("Medium paths", MEDIUM_CONFIGS),
        ("Long paths", LONG_CONFIGS[:3]),
    ]:
        print(f"\n  --- {section} ---")
        _print_header_3()
        for label, d, m, n in configs:
            path = rng.standard_normal((n, d))
            deriv = rng.standard_normal(sig_rust.siglength(d, m))
            t_rust = _time(
                lambda dr=deriv, p=path, m_=m: sig_rust.sigbackprop(dr, p, m_)
            )
            t_light = None
            if sig_light is not None:
                t_light = _time(
                    lambda dr=deriv, p=path, m_=m: sig_light.sigbackprop(dr, p, m_)
                )
            t_cpp = None
            if iisignature is not None:
                t_cpp = _time(
                    lambda dr=deriv, p=path, m_=m: iisignature.sigbackprop(dr, p, m_)
                )
            _print_row_3(label, t_rust, t_light, t_cpp)


def bench_logsigbackprop_single() -> None:
    """logsigbackprop() on single paths."""
    rng = np.random.default_rng(42)

    for section, configs in [
        ("Short paths", SHORT_CONFIGS),
        ("Medium paths", MEDIUM_CONFIGS[:3]),
    ]:
        print(f"\n  --- {section} ---")
        _print_header_3()
        for label, d, m, n in configs:
            path = rng.standard_normal((n, d))
            s_rust = sig_rust.prepare(d, m)
            ls_len = sig_rust.logsiglength(d, m)
            deriv = rng.standard_normal(ls_len)
            t_rust = _time(
                lambda dr=deriv, p=path, s=s_rust: sig_rust.logsigbackprop(dr, p, s)
            )
            t_light = None
            if sig_light is not None:
                s_light = sig_light.prepare(d, m)
                t_light = _time(
                    lambda dr=deriv, p=path, s=s_light: sig_light.logsigbackprop(
                        dr, p, s
                    )
                )
            t_cpp = None
            if iisignature is not None:
                s_cpp = iisignature.prepare(d, m)
                t_cpp = _time(
                    lambda dr=deriv, p=path, s=s_cpp: iisignature.logsigbackprop(
                        dr, p, s
                    )
                )
            _print_row_3(label, t_rust, t_light, t_cpp)


def bench_transforms_single() -> None:
    """sigjoin() and sigscale() on single paths."""
    rng = np.random.default_rng(42)

    configs = MEDIUM_CONFIGS[:4]

    print("\n  --- sigscale ---")
    _print_header_3()
    for label, d, m, n in configs:
        path = rng.standard_normal((n, d))
        s = sig_rust.sig(path, m)
        scales = rng.standard_normal(d) + 2
        t_rust = _time(
            lambda s_=s, sc=scales, d_=d, m_=m: sig_rust.sigscale(s_, sc, d_, m_)
        )
        t_light = None
        if sig_light is not None:
            t_light = _time(
                lambda s_=s, sc=scales, d_=d, m_=m: sig_light.sigscale(s_, sc, d_, m_)
            )
        t_cpp = None
        if iisignature is not None:
            t_cpp = _time(
                lambda s_=s, sc=scales, m_=m: iisignature.sigscale(s_, sc, m_)
            )
        _print_row_3(label, t_rust, t_light, t_cpp)

    print("\n  --- sigjoin ---")
    _print_header_3()
    for label, d, m, n in configs:
        path = rng.standard_normal((n, d))
        s = sig_rust.sig(path[: n // 2], m)
        seg = path[n // 2] - path[n // 2 - 1]
        t_rust = _time(
            lambda s_=s, sg=seg, d_=d, m_=m: sig_rust.sigjoin(s_, sg, d_, m_)
        )
        t_light = None
        if sig_light is not None:
            t_light = _time(
                lambda s_=s, sg=seg, d_=d, m_=m: sig_light.sigjoin(s_, sg, d_, m_)
            )
        t_cpp = None
        if iisignature is not None:
            t_cpp = _time(lambda s_=s, sg=seg, m_=m: iisignature.sigjoin(s_, sg, m_))
        _print_row_3(label, t_rust, t_light, t_cpp)


def bench_sig_batched() -> None:
    """sig() on batched paths (tests multithreading)."""
    rng = np.random.default_rng(42)

    _print_header_3()
    for label, batch, d, m, n in BATCH_CONFIGS:
        paths = rng.standard_normal((batch, n, d))
        t_rust = _time(lambda p=paths, m_=m: sig_rust.sig(p, m_), repeats=10)
        t_light = None
        if sig_light is not None:
            t_light = _time(lambda p=paths, m_=m: sig_light.sig(p, m_), repeats=10)
        t_cpp = None
        if iisignature is not None:
            t_cpp = _time(
                lambda p=paths, m_=m: _loop_batch(iisignature.sig, p, m_),
                repeats=10,
            )
        _print_row_3(label, t_rust, t_light, t_cpp)


def bench_logsig_batched() -> None:
    """logsig() on batched paths (tests multithreading)."""
    rng = np.random.default_rng(42)

    configs = BATCH_CONFIGS[:4]

    _print_header_3()
    for label, batch, d, m, n in configs:
        paths = rng.standard_normal((batch, n, d))
        s_rust = sig_rust.prepare(d, m)
        t_rust = _time(
            lambda p=paths, s=s_rust: sig_rust.logsig(p, s),
            repeats=10,
        )
        t_light = None
        if sig_light is not None:
            s_light = sig_light.prepare(d, m)
            t_light = _time(
                lambda p=paths, s=s_light: sig_light.logsig(p, s),
                repeats=10,
            )
        t_cpp = None
        if iisignature is not None:
            s_cpp = iisignature.prepare(d, m)
            t_cpp = _time(
                lambda p=paths, s=s_cpp: _loop_batch(iisignature.logsig, p, s),
                repeats=10,
            )
        _print_row_3(label, t_rust, t_light, t_cpp)


def bench_sigbackprop_batched() -> None:
    """sigbackprop() on batched paths (tests multithreading)."""
    rng = np.random.default_rng(42)

    configs = BATCH_CONFIGS[:4]

    _print_header_3()
    for label, batch, d, m, n in configs:
        paths = rng.standard_normal((batch, n, d))
        sl = sig_rust.siglength(d, m)
        derivs = rng.standard_normal((batch, sl))
        t_rust = _time(
            lambda dr=derivs, p=paths, m_=m: sig_rust.sigbackprop(dr, p, m_),
            repeats=10,
        )
        t_light = None
        if sig_light is not None:
            t_light = _time(
                lambda dr=derivs, p=paths, m_=m: sig_light.sigbackprop(dr, p, m_),
                repeats=10,
            )
        t_cpp = None
        if iisignature is not None:
            t_cpp = _time(
                lambda dr=derivs, p=paths, m_=m: _loop_batch2(
                    iisignature.sigbackprop, dr, p, m_
                ),
                repeats=10,
            )
        _print_row_3(label, t_rust, t_light, t_cpp)


def bench_logsigbackprop_batched() -> None:
    """logsigbackprop() on batched paths (tests multithreading)."""
    rng = np.random.default_rng(42)

    configs = [
        ("B=10, d=3, m=4, n=50", 10, 3, 4, 50),
        ("B=50, d=3, m=4, n=50", 50, 3, 4, 50),
        ("B=100, d=3, m=4, n=50", 100, 3, 4, 50),
    ]

    _print_header_3()
    for label, batch, d, m, n in configs:
        paths = rng.standard_normal((batch, n, d))
        s_rust = sig_rust.prepare(d, m)
        ls_len = sig_rust.logsiglength(d, m)
        derivs = rng.standard_normal((batch, ls_len))
        t_rust = _time(
            lambda dr=derivs, p=paths, s=s_rust: sig_rust.logsigbackprop(dr, p, s),
            repeats=10,
        )
        t_light = None
        if sig_light is not None:
            s_light = sig_light.prepare(d, m)
            t_light = _time(
                lambda dr=derivs, p=paths, s=s_light: sig_light.logsigbackprop(
                    dr, p, s
                ),
                repeats=10,
            )
        t_cpp = None
        if iisignature is not None:
            s_cpp = iisignature.prepare(d, m)
            t_cpp = _time(
                lambda dr=derivs, p=paths, s=s_cpp: _loop_batch2(
                    iisignature.logsigbackprop, dr, p, s
                ),
                repeats=10,
            )
        _print_row_3(label, t_rust, t_light, t_cpp)


if __name__ == "__main__":
    print("=" * 96)
    print("  sig-rust vs sig-light (Python) vs iisignature (C++)")
    print("=" * 96)

    libs = []
    print(f"\n  sig-rust:     {sig_rust.version()}")
    if sig_light is not None:
        print(f"  sig-light:    {sig_light.__version__}")
        libs.append("sig-light")
    else:
        print("  sig-light:    NOT INSTALLED")
    if iisignature is not None:
        print(f"  iisignature:  {iisignature.version()}")
        libs.append("iisignature")
    else:
        print("  iisignature:  NOT INSTALLED")
    print()

    # ── Single-path benchmarks ─────────────────────────────────────
    print("=" * 96)
    print("  1. SIGNATURE (sig)")
    print("=" * 96)
    bench_sig_single()

    print()
    print("=" * 96)
    print("  2. LOG SIGNATURE (logsig)")
    print("=" * 96)
    bench_logsig_single()

    print()
    print("=" * 96)
    print("  3. SIGNATURE BACKPROP (sigbackprop)")
    print("=" * 96)
    bench_sigbackprop_single()

    print()
    print("=" * 96)
    print("  4. LOG SIGNATURE BACKPROP (logsigbackprop)")
    print("=" * 96)
    bench_logsigbackprop_single()

    print()
    print("=" * 96)
    print("  5. TRANSFORMS (sigscale, sigjoin)")
    print("=" * 96)
    bench_transforms_single()

    # ── Batched benchmarks ─────────────────────────────────────────
    print()
    print("=" * 96)
    print("  6. BATCHED sig() -- multithreading test")
    print("=" * 96)
    print()
    bench_sig_batched()

    print()
    print("=" * 96)
    print("  7. BATCHED logsig() -- multithreading test")
    print("=" * 96)
    print()
    bench_logsig_batched()

    print()
    print("=" * 96)
    print("  8. BATCHED sigbackprop() -- multithreading test")
    print("=" * 96)
    print()
    bench_sigbackprop_batched()

    print()
    print("=" * 96)
    print("  9. BATCHED logsigbackprop() -- multithreading test")
    print("=" * 96)
    print()
    bench_logsigbackprop_batched()

    print()
    print("=" * 96)
    print("  Benchmark complete.")
    print("=" * 96)
