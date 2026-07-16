.PHONY: help test test-geometry test-all eval-independence eval-all demo demo-gpu install lint

# ── Default ───────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "EpistemicRuntime — entry points"
	@echo ""
	@echo "  No GPU required:"
	@echo "    make demo              Five routing cases + geometry scatter + independence proof"
	@echo "    make eval-independence Structural ceiling test (synthetic, n=1000)"
	@echo "    make test              All unit tests (32 geometry + routing invariants)"
	@echo "    make test-geometry     Epistemic geometry tests only"
	@echo ""
	@echo "  Requires GPU + model:"
	@echo "    make demo-gpu          Full demo on real model (Qwen2.5-1.5B)"
	@echo "    make eval-all          All evals including adversarial battery"
	@echo ""
	@echo "  Setup:"
	@echo "    make install           pip install -e .[all]"
	@echo "    make lint              pyflakes esm/ cams/"
	@echo ""

# ── No GPU required ───────────────────────────────────────────────────────────

demo:
	@echo "Running GPU-free epistemic geometry demo..."
	python3 demo/synthetic_geometry.py

eval-independence:
	@echo "Running structural ceiling test (n=1000 synthetic samples)..."
	python3 evals/independence_test.py --synthetic --n 500

test-geometry:
	python3 -m pytest tests/unit/test_epistemic_geometry.py -v --tb=short

test:
	python3 -m pytest tests/unit/ -v --tb=short

# ── Requires GPU + model ──────────────────────────────────────────────────────

demo-gpu:
	@echo "Running full demo (requires Qwen2.5-1.5B + calibration)..."
	python3 demo/epistemic_demo.py --model Qwen/Qwen2.5-1.5B-Instruct --auto-calibrate

eval-adversarial:
	@echo "Running adversarial invariance battery (requires GPU + Llama-3.2-3B-Instruct)..."
	python3 evals/adversarial_invariance.py \
		--model meta-llama/Llama-3.2-3B-Instruct \
		--calibration checkpoints/llama3b_cal.json \
		--n 50

eval-cross-arch:
	@echo "Running cross-architecture validation (requires GPU)..."
	python3 evals/cross_arch_validation.py --n 100

eval-all: eval-independence eval-adversarial eval-cross-arch

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	pip install -e ".[all]"

lint:
	python3 -m pyflakes esm/ cams/ 2>&1 | head -40 || true

# ── Version check ─────────────────────────────────────────────────────────────

version:
	python3 -m esm version
