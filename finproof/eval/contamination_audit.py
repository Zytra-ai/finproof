"""
FinProof v1 — Contamination audit
Apache 2.0 — https://github.com/zytra-ai/finproof

Verifies that a training corpus has not been contaminated with
FinProof evaluation examples. Uses SHA-1 exact match and
MinHash 5-gram Jaccard near-duplicate detection.

Usage:
    python contamination_audit.py \
        --finproof-tier2 path/to/finproof_v1_tier2_public.jsonl \
        --finproof-tier3 path/to/finproof_v1_tier3_research.jsonl \
        --training-corpus path/to/your_training_corpus.jsonl \
        [--text-field text]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> set[str]:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    return {t[i : i + k] for i in range(max(0, len(t) - k + 1))}


def minhash_sig(text: str, n_hash: int = 128) -> list[int]:
    """Simple minhash signature."""
    s = shingles(text)
    import hashlib
    sig = []
    for i in range(n_hash):
        seed = i.to_bytes(4, "big")
        sig.append(min(int(hashlib.md5(seed + w.encode()).hexdigest(), 16) for w in s) if s else 0)
    return sig


def jaccard_minhash(sig_a: list[int], sig_b: list[int]) -> float:
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)


def load_texts(path: str, text_field: str = "input") -> dict[str, str]:
    rows = [json.loads(l) for l in Path(path).open()]
    return {r.get("id", sha1(r[text_field])): r[text_field] for r in rows if text_field in r}


def audit(
    finproof_paths: list[str],
    training_path: str,
    text_field: str = "input",
    minhash_threshold: float = 0.85,
    verbose: bool = True,
) -> dict:
    # Build FinProof SHA-1 index
    fp_texts: dict[str, str] = {}
    for p in finproof_paths:
        fp_texts.update(load_texts(p, text_field="input"))

    fp_sha_index = {sha1(t): fid for fid, t in fp_texts.items()}
    fp_sigs = {fid: minhash_sig(t) for fid, t in fp_texts.items()}

    # Load training corpus
    train_texts = load_texts(training_path, text_field)
    if verbose:
        print(f"FinProof examples indexed : {len(fp_texts):,}")
        print(f"Training corpus rows      : {len(train_texts):,}")

    exact_collisions = []
    near_dup_collisions = []

    for tid, text in train_texts.items():
        # Exact match
        h = sha1(text)
        if h in fp_sha_index:
            exact_collisions.append({"train_id": tid, "finproof_id": fp_sha_index[h]})
            continue

        # Near-duplicate
        sig = minhash_sig(text)
        for fid, fsig in fp_sigs.items():
            j = jaccard_minhash(sig, fsig)
            if j >= minhash_threshold:
                near_dup_collisions.append(
                    {"train_id": tid, "finproof_id": fid, "jaccard": round(j, 3)}
                )
                break

    total = len(train_texts)
    result = {
        "training_corpus_rows": total,
        "finproof_examples_checked": len(fp_texts),
        "exact_collisions": len(exact_collisions),
        "exact_collision_pct": round(100 * len(exact_collisions) / total, 4) if total else 0,
        "near_dup_collisions": len(near_dup_collisions),
        "near_dup_collision_pct": round(100 * len(near_dup_collisions) / total, 4) if total else 0,
        "contaminated": len(exact_collisions) > 0 or len(near_dup_collisions) > 0,
        "verdict": "CLEAN" if (len(exact_collisions) == 0 and len(near_dup_collisions) == 0)
                   else "CONTAMINATED",
        "exact_collision_examples": exact_collisions[:5],
        "near_dup_examples": near_dup_collisions[:5],
    }

    if verbose:
        print(f"\nContamination audit result: {result['verdict']}")
        print(f"  Exact matches     : {result['exact_collisions']} ({result['exact_collision_pct']:.4f}%)")
        print(f"  Near-duplicates   : {result['near_dup_collisions']} ({result['near_dup_collision_pct']:.4f}%)")
        if result["contaminated"]:
            print("\n  ⚠️  TRAINING CORPUS IS CONTAMINATED WITH FINPROOF EXAMPLES")
            print("     Scores on FinProof will be inflated and invalid.")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finproof-tier2", required=True)
    ap.add_argument("--finproof-tier3", default=None)
    ap.add_argument("--training-corpus", required=True)
    ap.add_argument("--text-field", default="input")
    ap.add_argument("--minhash-threshold", type=float, default=0.85)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    fps = [args.finproof_tier2]
    if args.finproof_tier3:
        fps.append(args.finproof_tier3)

    result = audit(fps, args.training_corpus, args.text_field, args.minhash_threshold)

    if args.output_json:
        import json
        Path(args.output_json).write_text(json.dumps(result, indent=2))
        print(f"\nAudit report saved to {args.output_json}")


if __name__ == "__main__":
    main()
