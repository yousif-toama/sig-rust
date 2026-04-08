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

# Run Rust tests with coverage
rust-test-cov:
    cargo llvm-cov --lcov --output-path rust-lcov.info

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

# Run full test suite with coverage (Rust + Python)
test-cov-all: rust-test-cov test-cov

# Run quick benchmark
bench:
    uv run python scripts/benchmark.py

# Run full benchmark suite
bench-full:
    uv run python scripts/benchmark_full.py

# Install all dependencies
sync:
    uv sync --all-extras --group dev --group test
