"""
FinProof v1 — Submission packager for finproof.zytra.ai
Apache 2.0 — https://github.com/zytra-ai/finproof

Packages local Tier 2 + Tier 3 scores for official leaderboard submission.
Official Tier 4 (withheld test set) evaluation happens server-side at
finproof.zytra.ai — you cannot self-score on Tier 4.

Usage:
    python submit.py \
        --scores finproof_scores.json \
        --model-name "MyModel v2.0" \
        --model-org "MyOrg" \
        --hf-model-id "myorg/mymodel-v2" \
        [--output submission.zip]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


SUBMISSION_API = "https://finproof.zytra.ai/api/v1/submit"
FINPROOF_TIER4_SHA256 = "bf35df2e5a3f08c9202555db1a5bd825"  # partial — for integrity check


def prepare(scores: dict, output_path: str, model_name: str = "", model_org: str = "",
            hf_model_id: str = "") -> str:
    """Create submission zip. Returns path."""
    submission = {
        "finproof_version": "v1",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": model_name,
            "org": model_org,
            "hf_model_id": hf_model_id,
        },
        "local_scores": scores,
        "tier4_sha256_prefix": FINPROOF_TIER4_SHA256[:16],
        "submission_sha256": hashlib.sha256(
            json.dumps(scores, sort_keys=True).encode()
        ).hexdigest(),
        "note": (
            "Tier 4 evaluation performed server-side by Zytra only. "
            "Official leaderboard scores reflect Tier 4 results exclusively."
        ),
    }

    Path(output_path).write_text(json.dumps(submission, indent=2))
    print(f"[FinProof] Submission prepared: {output_path}")
    print(f"  Upload at: {SUBMISSION_API}")
    print(f"  Or email:  finproof@zytra.ai")
    return output_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--model-name", default="")
    ap.add_argument("--model-org", default="")
    ap.add_argument("--hf-model-id", default="")
    ap.add_argument("--output", default="finproof_submission.json")
    args = ap.parse_args()

    scores = json.loads(Path(args.scores).read_text())
    prepare(scores, args.output, args.model_name, args.model_org, args.hf_model_id)


if __name__ == "__main__":
    main()
