# sig-rust development commands

# Run all checks (Rust + Python)
check: rust-check python-check

# Run all Rust checks
rust-check: rust-fmt-check rust-lint rust-test

# Run all Python checks
python-check: lint format-check typecheck test

# --- Rust ---

# Run Rust tests
rust-test:
    cargo test

# Run Rust linter
rust-lint:
    cargo clippy --all-targets --all-features -- -D warnings

# Check Rust formatting
rust-fmt-check:
    cargo fmt --check

# Auto-format Rust code
rust-fmt:
    cargo fmt

# Run Rust tests with coverage (excludes python.rs PyO3 wrappers)
rust-test-cov:
    cargo llvm-cov --lcov --output-path rust-lcov.info --ignore-filename-regex python

# --- Python ---

# Build the native extension
build:
    uv run maturin develop --release

# Run Python linter
lint:
    uv run ruff check .

# Check Python formatting
format-check:
    uv run ruff format --check .

# Auto-format Python code
format:
    uv run ruff format .

# Run Python type checker
typecheck:
    uv run ty check

# Run Python tests
test:
    uv run pytest

# Run Python tests with coverage
test-cov:
    uv run pytest --cov=python/sig_rust --cov-report=term-missing

# --- Combined ---

# Auto-format everything
format-all: rust-fmt format

# Run full test suite with combined coverage (Rust + Python → single LCOV)
# Instruments Rust so pytest exercising PyO3 bindings also generates coverage data.
test-cov-all:
    #!/usr/bin/env bash
    set -euo pipefail
    eval "$(cargo llvm-cov show-env --export-prefix)"
    export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR"
    cargo llvm-cov clean --workspace
    cargo test
    uv run maturin develop
    uv run pytest --cov=python/sig_rust --cov-report=term-missing --cov-report=xml:python-coverage.xml
    cargo llvm-cov report --lcov --output-path combined-lcov.info

# Run quick benchmark
bench:
    uv run python scripts/benchmark.py

# Run full benchmark suite
bench-full:
    uv run python scripts/benchmark_full.py

# Install all dependencies
sync:
    uv sync --all-extras --group dev --group test
