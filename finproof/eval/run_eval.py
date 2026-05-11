"""
FinProof v1 — Local evaluation runner
Apache 2.0 — https://github.com/zytra-ai/finproof

Runs a model's guardian function against FinProof Tier 2 and/or Tier 3
examples, computes scores, and optionally prepares a submission package
for the official leaderboard.

Usage — one command from HuggingFace:
    python run_eval.py \
        --hf-dataset Zytra/finproof-bench \
        --model-module my_guardian \
        --output scores.json

Usage — local files:
    python run_eval.py \
        --finproof-tier2 finproof_v1_tier2_public.jsonl \
        [--finproof-tier3 finproof_v1_tier3_research.jsonl] \
        --model-module my_guardian \
        --output scores.json

Model module interface (my_guardian.py):
    def predict(text: str) -> int:
        # Returns 0 (benign) or 1 (attack)
        ...

    Or batch interface:
    def predict_batch(texts: list[str]) -> list[int]:
        # Faster; preferred if available
        ...
"""
from __future__ import annotations
import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Callable

from scoring import compute_metrics, print_report, load_jsonl


def run_model(
    rows: list[dict],
    predict_fn: Callable[[str], int] | None = None,
    predict_batch_fn: Callable[[list[str]], list[int]] | None = None,
    batch_size: int = 64,
    verbose: bool = True,
) -> dict[str, int]:
    """Returns {id: prediction} dict."""
    if predict_batch_fn:
        predictions = {}
        total = len(rows)
        for i in range(0, total, batch_size):
            batch = rows[i : i + batch_size]
            preds = predict_batch_fn([r["input"] for r in batch])
            for r, p in zip(batch, preds):
                predictions[r["id"]] = p
            if verbose and (i + batch_size) % 200 == 0:
                print(f"  {min(i + batch_size, total)}/{total} processed...")
        return predictions

    if predict_fn:
        predictions = {}
        for i, r in enumerate(rows):
            predictions[r["id"]] = predict_fn(r["input"])
            if verbose and (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(rows)} processed...")
        return predictions

    raise ValueError("Provide either predict_fn or predict_batch_fn")


def download_from_hf(hf_dataset: str, local_dir: str = ".finproof_cache") -> tuple[str, str | None]:
    """Download Tier 2 (and optionally Tier 3) from HuggingFace. Returns (tier2_path, tier3_path)."""
    from huggingface_hub import hf_hub_download
    import os
    os.makedirs(local_dir, exist_ok=True)
    print(f"[FinProof] Downloading from {hf_dataset}...")
    t2 = hf_hub_download(repo_id=hf_dataset, filename="data/finproof_v1_tier2_public.jsonl",
                         repo_type="dataset", local_dir=local_dir)
    print(f"[FinProof] Tier 2 cached at {t2}")
    t3 = None
    try:
        research_repo = hf_dataset.replace("finproof-bench", "finproof-research")
        t3 = hf_hub_download(repo_id=research_repo, filename="data/finproof_v1_tier3_research.jsonl",
                             repo_type="dataset", local_dir=local_dir)
        print(f"[FinProof] Tier 3 cached at {t3}")
    except Exception:
        print("[FinProof] Tier 3 not available (requires research agreement). Skipping.")
    return t2, t3


def main():
    ap = argparse.ArgumentParser(
        description="FinProof v1 evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One-command from HuggingFace:
  python run_eval.py --hf-dataset Zytra/finproof-bench --model-module my_guardian

  # Local files:
  python run_eval.py --finproof-tier2 finproof_v1_tier2_public.jsonl --model-module my_guardian
        """
    )
    ap.add_argument("--hf-dataset", default=None,
                    help="HuggingFace dataset ID (e.g. Zytra/finproof-bench). Auto-downloads data.")
    ap.add_argument("--finproof-tier2", default=None,
                    help="Path to local Tier 2 JSONL (required if --hf-dataset not set)")
    ap.add_argument("--finproof-tier3", default=None)
    ap.add_argument("--model-module", required=True,
                    help="Python module with predict() or predict_batch()")
    ap.add_argument("--output", default="finproof_scores.json")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--prepare-submission", action="store_true",
                    help="Package results for finproof.zytra.ai submission")
    args = ap.parse_args()

    # Auto-download from HF if requested
    if args.hf_dataset:
        args.finproof_tier2, args.finproof_tier3 = download_from_hf(args.hf_dataset)
    elif not args.finproof_tier2:
        ap.error("Provide either --hf-dataset or --finproof-tier2")

    # Load model
    mod = importlib.import_module(args.model_module)
    predict_fn = getattr(mod, "predict", None)
    predict_batch_fn = getattr(mod, "predict_batch", None)
    if not predict_fn and not predict_batch_fn:
        raise ValueError(f"{args.model_module} must define predict() or predict_batch()")

    all_results = {}

    for tier_path, tier_label in [
        (args.finproof_tier2, "T2"),
        (args.finproof_tier3, "T3"),
    ]:
        if not tier_path:
            continue
        print(f"\n[FinProof] Running on Tier {tier_label[-1]}: {tier_path}")
        rows = load_jsonl(tier_path)
        # Skip withheld inputs
        rows = [r for r in rows if r.get("input") != "[WITHHELD]"]

        t0 = time.perf_counter()
        preds = run_model(rows, predict_fn=predict_fn,
                          predict_batch_fn=predict_batch_fn,
                          batch_size=args.batch_size)
        elapsed = time.perf_counter() - t0

        metrics = compute_metrics(rows, preds)
        metrics["eval_seconds"] = round(elapsed, 2)
        metrics["prompts_per_second"] = round(len(rows) / elapsed, 1) if elapsed else 0

        print_report(metrics, tier=tier_label[-1])
        all_results[f"tier_{tier_label[-1].lower()}"] = metrics

    Path(args.output).write_text(json.dumps(all_results, indent=2))
    print(f"[FinProof] Scores saved to {args.output}")

    if args.prepare_submission:
        import submit
        submit.prepare(all_results, args.output.replace(".json", "_submission.zip"))


if __name__ == "__main__":
    main()
