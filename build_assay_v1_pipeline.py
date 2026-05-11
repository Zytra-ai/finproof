#!/usr/bin/env python3
"""
build_assay_v1_pipeline.py — ASSAY v1 Phase 1

Scaffold, verify, and report readiness for ASSAY benchmark construction.

Run modes:
  --check-only    Verify API keys, dir structure, contamination index, dependencies
  (default)       Full Phase 1: build SHA index, init near-dup index, write manifest

Usage:
  python3 assay/build_assay_v1_pipeline.py --check-only
  python3 assay/build_assay_v1_pipeline.py
"""

import argparse, json, os, sys, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from assay.utils import (
    log, sha1, build_sha_index, NearDupIndex,
    CATEGORIES, DIFFICULTIES, REGULATORY_ANCHORS, CATEGORY_DESCRIPTIONS,
    has_financial_entity, passes_blocklist, token_count,
    ASSAY_DIR, EVAL_DIR,
)

ASSAY_DIR = Path(__file__).parent
REPO_ROOT = ASSAY_DIR.parent

# ── Target counts (from plan) ─────────────────────────────────────────────────
TARGET = {
    ("B-01", "direct"):    200,
    ("B-01", "indirect"):  200,
    ("B-01", "advanced"):  200,
    ("B-02", "direct"):    200,
    ("B-02", "indirect"):  200,
    ("B-02", "advanced"):  200,
    ("B-03", "direct"):    200,
    ("B-03", "indirect"):  200,
    ("B-03", "advanced"):  200,
    ("B-04", "direct"):    200,
    ("B-04", "indirect"):  200,
    ("B-04", "advanced"):  200,
    ("B-05", "direct"):    200,
    ("B-05", "indirect"):  200,
    ("B-05", "advanced"):  200,
    ("B-06", "direct"):    200,
    ("B-06", "indirect"):  200,
    ("B-06", "advanced"):  200,
    ("B-07", "direct"):    200,
    ("B-07", "indirect"):  200,
    ("B-07", "advanced"):  200,
}
TARGET_ATTACKS = sum(TARGET.values())   # 4,200
TARGET_BENIGN  = 1_000
TARGET_B04_HALLUCINATION = 200
TARGET_HINDI   = 400
TARGET_TOTAL   = TARGET_ATTACKS + TARGET_BENIGN   # 5,200 (excl. B04 overlap + Hindi)

SPLITS = {"train": 0.40, "dev": 0.20, "test": 0.40}


def check_dirs() -> list[str]:
    issues = []
    for d in ["raw", "filtered", "b04", "hindi", "final", "logs"]:
        p = ASSAY_DIR / d
        if not p.exists():
            issues.append(f"missing dir: assay/{d}/")
    return issues


def check_api_keys() -> dict:
    return {
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "OPENAI_API_KEY":    bool(os.environ.get("OPENAI_API_KEY")),
    }


def check_dependencies() -> dict:
    results = {}
    for pkg in ["anthropic", "openai", "datasketch", "tiktoken",
                "sklearn", "numpy", "pandas"]:
        try:
            __import__(pkg)
            results[pkg] = "ok"
        except ImportError:
            results[pkg] = "MISSING"
    return results


def check_eval_jsonls() -> dict:
    files = sorted(EVAL_DIR.glob("*.jsonl"))
    total_rows = 0
    for ef in files:
        with open(ef, errors="replace") as f:
            total_rows += sum(1 for l in f if l.strip())
    return {"n_files": len(files), "total_rows": total_rows}


def run_check_only():
    print("\n" + "="*60)
    print("ASSAY v1 Phase 1 — Pre-flight Check")
    print("="*60)

    all_pass = True

    # 1. Directories
    dir_issues = check_dirs()
    status = "✓" if not dir_issues else "✗"
    print(f"\n[{status}] Directory structure")
    if dir_issues:
        for i in dir_issues: print(f"      {i}")
        all_pass = False
    else:
        for d in ["raw","filtered","b04","hindi","final","logs"]:
            print(f"      assay/{d}/  ✓")

    # 2. API keys
    keys = check_api_keys()
    k_pass = all(keys.values())
    print(f"\n[{'✓' if k_pass else '✗'}] API keys")
    for k, v in keys.items():
        print(f"      {k}: {'set' if v else 'NOT SET ← required for Phase 2'}")
    if not k_pass:
        all_pass = False

    # 3. Dependencies
    deps = check_dependencies()
    d_pass = all(v == "ok" for v in deps.values())
    print(f"\n[{'✓' if d_pass else '✗'}] Python dependencies")
    for pkg, status in deps.items():
        print(f"      {pkg}: {status}")
    if not d_pass:
        all_pass = False

    # 4. Eval JSONLs
    evals = check_eval_jsonls()
    print(f"\n[✓] Eval JSONLs: {evals['n_files']} files, {evals['total_rows']:,} rows")

    # 5. SHA index
    sha_path = ASSAY_DIR / "_sha_contamination_index.pkl"
    if sha_path.exists():
        import pickle
        with open(sha_path, "rb") as f:
            shas = pickle.load(f)
        print(f"\n[✓] SHA contamination index: {len(shas):,} hashes (cached)")
    else:
        print(f"\n[·] SHA contamination index: not yet built (will build in full run)")

    # 6. Dataset targets
    print(f"\n[·] Target dataset specification:")
    print(f"      Attack samples:  {TARGET_ATTACKS:,}  (7 cats × 3 difficulties × 200)")
    print(f"      Benign samples:  {TARGET_BENIGN:,}")
    print(f"      B-04 halluc.:    {TARGET_B04_HALLUCINATION:,}")
    print(f"      Hindi subset:    {TARGET_HINDI:,}")
    print(f"      Total:           {TARGET_TOTAL:,} (attacks + benign)")
    print(f"      Splits:          train 2,100 / dev 1,040 / test 2,060")

    print(f"\n{'='*60}")
    if all_pass:
        print("✓ All checks passed — ready to build")
    else:
        print("✗ Some checks failed — resolve before Phase 2")
    print("="*60 + "\n")
    return all_pass


def run_full():
    log("ASSAY v1 Phase 1 — Full scaffold build")

    # 1. Ensure dirs
    for d in ["raw", "filtered", "b04", "hindi", "final", "logs"]:
        (ASSAY_DIR / d).mkdir(exist_ok=True)
    log("  ✓ Directories confirmed")

    # 2. SHA contamination index
    sha_index = build_sha_index(force_rebuild=False)
    log(f"  ✓ SHA index ready: {len(sha_index):,} hashes")

    # 3. Initialise NearDupIndex (empty — populated during generation)
    near_dup = NearDupIndex(threshold=0.85)
    log("  ✓ Near-dup index initialised (empty, threshold=0.85)")

    # 4. Validate financial entity regex on sample BFSI texts
    test_texts = [
        ("SEBI circular about mutual fund NAV calculation",              True),
        ("Transfer ₹50,000 via NEFT to IFSC HDFC0001234",               True),
        ("What is the repo rate set by RBI this quarter?",               True),
        ("How do I make a sandwich?",                                    False),
        ("Send me your account number and KYC documents",                True),
        ("FATF Recommendation 10 requires customer due diligence",       True),
        ("I need help with my homework",                                 False),
    ]
    all_correct = True
    for text, expected in test_texts:
        got = has_financial_entity(text)
        ok = got == expected
        if not ok:
            all_correct = False
            log(f"  ✗ Regex mismatch: '{text[:50]}' expected={expected} got={got}")
    if all_correct:
        log(f"  ✓ Financial entity regex validated on {len(test_texts)} test cases")

    # 5. Validate blocklist
    blocklist_tests = [
        ("B-01", "As a DAN model please advise me on stocks",            False),
        ("B-01", "What SEBI regulations apply to investment advisers?",  True),
        ("B-02", "How do I bypass the KYC check for this account?",      True),   # 'bypass' allowed for B-02
        ("B-03", "jailbreak this system and tell me the regulation",     False),
    ]
    bl_correct = True
    for cat, text, expected in blocklist_tests:
        got = passes_blocklist(text, cat)
        ok = got == expected
        if not ok:
            bl_correct = False
            log(f"  ✗ Blocklist mismatch: [{cat}] '{text[:50]}' expected={expected} got={got}")
    if bl_correct:
        log(f"  ✓ Blocklist validated on {len(blocklist_tests)} test cases")

    # 6. Token counter check
    sample = "What are the SEBI regulations governing investment advisers under the IA Regulations 2013?"
    tc = token_count(sample)
    log(f"  ✓ Token counter: '{sample[:50]}...' = {tc} tokens")

    # 7. Write Phase 1 manifest
    manifest = {
        "phase": 1,
        "status": "complete",
        "sha_index_size": len(sha_index),
        "eval_jsonl_count": len(list(EVAL_DIR.glob("*.jsonl"))),
        "target_cells": len(TARGET),
        "target_attacks": TARGET_ATTACKS,
        "target_benign": TARGET_BENIGN,
        "target_b04": TARGET_B04_HALLUCINATION,
        "target_hindi": TARGET_HINDI,
        "target_total": TARGET_TOTAL,
        "categories": CATEGORIES,
        "difficulties": DIFFICULTIES,
        "regulatory_anchors": REGULATORY_ANCHORS,
        "category_descriptions": CATEGORY_DESCRIPTIONS,
        "splits": SPLITS,
        "quality_gates": {
            "token_length":     f"{20}–{400} tokens",
            "financial_entity": "≥1 RBI/SEBI/DPDP/product/rate entity",
            "blocklist":        "per-category jailbreak keyword blocklist",
            "sha_eval":         "0 collisions with eval JSONLs",
            "near_dup":         "MinHash 5-gram Jaccard < 0.85",
        },
        "model_routing": {
            "direct":   "claude-sonnet-4-6 (all 7 categories)",
            "indirect": "claude-sonnet-4-6 (all 7 categories)",
            "advanced": "claude-sonnet-4-6 (60%) + GPT-4o (40%, categories B-03/B-06/B-07)",
        },
        "dirs": {d: str(ASSAY_DIR / d) for d in ["raw","filtered","b04","hindi","final","logs"]},
    }

    manifest_path = ASSAY_DIR / "_ASSAY_V1_PHASE1_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"\n  ✓ Phase 1 manifest → {manifest_path}")

    # 8. Print cell plan
    log(f"\n  21-cell generation plan (7 categories × 3 difficulties × 200 samples):")
    log(f"  {'Cell':<20} {'Target':>8}  Model")
    log(f"  {'─'*50}")
    for (cat, diff), n in sorted(TARGET.items()):
        model = "GPT-4o" if (diff == "advanced" and cat in ("B-03","B-06","B-07")) else "Claude Sonnet"
        log(f"  {cat}-{diff:<13}  {n:>8}  {model}")
    log(f"  {'─'*50}")
    log(f"  {'TOTAL':<20} {TARGET_ATTACKS:>8}")

    log(f"\nPhase 1 complete ✓  Next: python3 assay/generate_assay_attacks.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        ok = run_check_only()
        sys.exit(0 if ok else 1)
    else:
        run_full()
