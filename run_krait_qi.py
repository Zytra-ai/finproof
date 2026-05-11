#!/usr/bin/env python3
"""
run_krait_qi.py — KRAIT-QI Orchestrator

Runs all three QCBM-based adversarial generation pipelines and produces
the final KRAIT-QI benchmark corpus with full provenance.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python3 assay/run_krait_qi.py [--approach boundary|joint|difficulty|all]

Output: assay/krait_qi/
    boundary_adversarial.jsonl   — Option 1: decision-boundary prompts
    joint_distribution.jsonl     — Option 2: cross-pattern multi-vector
    difficulty_gradient.jsonl    — Option 3: intermediate difficulty
    krait_qi_full.jsonl          — All three merged
    _KRAIT_QI_MANIFEST.json      — Full provenance + circuit params SHAs

Research claim (paper-ready):
    "KRAIT-QI adversarial prompts are generated via Quantum Circuit Born Machine
    sampling: (1) boundary prompts sample from Semalith's decision-boundary
    feature distribution, (2) joint prompts sample from the entangled cross-pattern
    distribution using hardware-efficient 8-qubit circuits with cross-partition
    CNOT gates, (3) gradient prompts interpolate between direct/advanced QCBM
    distributions to probe the difficulty gap. This provides broader distributional
    coverage than classical template expansion, which remains in-distribution."
"""

import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from assay.qcbm_core import KRAIT_DIR
import pennylane as qml


def check_env():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("NOTE: ANTHROPIC_API_KEY not set — using file-based handoff generation.")
        print("      Claude Code session will fill pending_requests/ as prompts are needed.")


def merge_outputs():
    """Merge all three JSONL outputs into krait_qi_full.jsonl."""
    all_records = []
    for fname in ["boundary_adversarial.jsonl", "joint_distribution.jsonl", "difficulty_gradient.jsonl"]:
        path = KRAIT_DIR / fname
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    all_records.append(json.loads(line))

    out = KRAIT_DIR / "krait_qi_full.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in all_records) + "\n")
    return all_records


def build_manifest(boundary_audit, joint_audit, difficulty_audit, elapsed_total):
    from collections import Counter
    import pennylane as qml

    full = merge_outputs()
    approach_counts = Counter(r.get("krait_approach") for r in full)
    cat_counts = Counter(r.get("category") for r in full)

    manifest = {
        "benchmark": "KRAIT-QI",
        "version": "1.0",
        "description": (
            "Quantum Circuit Born Machine adversarial prompts for BFSI AI safety evaluation. "
            "Three generation approaches: decision-boundary, cross-pattern joint distribution, "
            "difficulty gradient interpolation."
        ),
        "total_records": len(full),
        "approach_breakdown": dict(approach_counts),
        "category_breakdown": dict(cat_counts),
        "qcbm_config": {
            "n_qubits": 8,
            "n_layers": 4,
            "optimizer": "L-BFGS-B",
            "max_iter": 200,
            "n_restarts": 3,
            "kernel": "gaussian_mmd",
            "pennylane_version": qml.__version__,
        },
        "semalith_version": "v1.5",
        "boundary_p_range": [0.35, 0.65],
        "elapsed_total_s": round(elapsed_total, 1),
        "pipelines": {
            "boundary_adversarial": boundary_audit,
            "joint_distribution": joint_audit,
            "difficulty_gradient": difficulty_audit,
        },
    }

    (KRAIT_DIR / "_KRAIT_QI_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def print_summary(manifest):
    print(f"\n{'='*70}")
    print(f"KRAIT-QI Generation Complete")
    print(f"{'='*70}")
    print(f"  Total records:     {manifest['total_records']}")
    print(f"  Approach breakdown:")
    for k, v in manifest["approach_breakdown"].items():
        print(f"    {k:<30} {v:>5}")
    print(f"  Category breakdown:")
    for k, v in sorted(manifest["category_breakdown"].items()):
        print(f"    {k:<10} {v:>5}")
    print(f"  Output: {KRAIT_DIR}")
    print(f"  Elapsed: {manifest['elapsed_total_s']:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="KRAIT-QI adversarial benchmark generation")
    parser.add_argument(
        "--approach", default="all",
        choices=["boundary", "joint", "difficulty", "all"],
        help="Which pipeline(s) to run (default: all)"
    )
    args = parser.parse_args()

    check_env()
    t0 = time.time()

    boundary_audit = joint_audit = difficulty_audit = {}

    if args.approach in ("boundary", "all"):
        from assay.qcbm_boundary import run as run_boundary
        _, boundary_audit = run_boundary(n_generate_per_category=30)

    if args.approach in ("joint", "all"):
        from assay.qcbm_joint import run as run_joint
        _, joint_audit = run_joint(n_generate_per_group=40)

    if args.approach in ("difficulty", "all"):
        from assay.qcbm_difficulty import run as run_difficulty
        _, difficulty_audit = run_difficulty(n_generate_per_category=20)

    manifest = build_manifest(boundary_audit, joint_audit, difficulty_audit,
                              elapsed_total=time.time() - t0)
    print_summary(manifest)


if __name__ == "__main__":
    main()
