#!/usr/bin/env python3
"""
generate_assay_attacks_inline.py — ASSAY v1 Phase 2 (in-session generation)

Runs quality gates on all inline prompts defined in prompts_b0*.py and writes
accepted samples to assay/raw/{category}_{difficulty}.jsonl.

Quality gates (all 5 must pass):
  1. Token length: 20–400 tokens
  2. Financial entity: ≥1 RBI/SEBI/DPDP/product/rate entity
  3. Jailbreak blocklist: per-category keyword check
  4. SHA-1 eval contamination: 0 collisions with eval_jsonls/
  5. MinHash 5-gram Jaccard: Jaccard < 0.85 with any accepted item in same cell
"""

import json, sys, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from assay.utils import (
    log, sha1, build_sha_index, NearDupIndex,
    quality_gate, make_sample, ASSAY_DIR,
)

from assay.prompts_b01 import PROMPTS_B01
from assay.prompts_b02 import PROMPTS_B02
from assay.prompts_b03 import PROMPTS_B03
from assay.prompts_b04 import PROMPTS_B04
from assay.prompts_b05 import PROMPTS_B05
from assay.prompts_b06 import PROMPTS_B06
from assay.prompts_b07 import PROMPTS_B07

ALL_PROMPTS = {
    "B-01": PROMPTS_B01,
    "B-02": PROMPTS_B02,
    "B-03": PROMPTS_B03,
    "B-04": PROMPTS_B04,
    "B-05": PROMPTS_B05,
    "B-06": PROMPTS_B06,
    "B-07": PROMPTS_B07,
}

RAW_DIR = ASSAY_DIR / "raw"
RAW_DIR.mkdir(exist_ok=True)


def main():
    sha_index = build_sha_index()
    grand_total = 0
    manifest_rows = {}

    for category, diffs in ALL_PROMPTS.items():
        for difficulty, texts in diffs.items():
            near_dup = NearDupIndex(threshold=0.85)
            accepted = []
            drops = defaultdict(int)

            for i, text in enumerate(texts):
                passes, reason = quality_gate(text, category, sha_index, near_dup)
                if passes:
                    near_dup.add(text, key=f"{category}-{difficulty}-{i}")
                    sha_index.add(sha1(text))   # prevent within-cell dups
                    accepted.append(make_sample(
                        category=category,
                        difficulty=difficulty,
                        input_text=text,
                        split="train",
                        seq=len(accepted),
                    ))
                else:
                    drops[reason] += 1

            out_path = RAW_DIR / f"{category.replace('-','_')}_{difficulty}.jsonl"
            with open(out_path, "w") as f:
                for row in accepted:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            drop_str = "  ".join(f"{k}:{v}" for k, v in drops.items()) or "none"
            log(f"  {category}-{difficulty:<10} {len(accepted):>3} accepted  drops: {drop_str}")
            manifest_rows[f"{category}-{difficulty}"] = {
                "accepted": len(accepted),
                "dropped": dict(drops),
                "path": str(out_path),
            }
            grand_total += len(accepted)

    log(f"\nTotal accepted: {grand_total:,} / 4,200 target")

    # Summary table
    print(f"\n{'Category':<8} {'Direct':>8} {'Indirect':>9} {'Advanced':>9} {'Total':>7}")
    print("─"*45)
    for cat in ["B-01","B-02","B-03","B-04","B-05","B-06","B-07"]:
        row = [manifest_rows.get(f"{cat}-{d}", {}).get("accepted", 0)
               for d in ["direct","indirect","advanced"]]
        print(f"{cat:<8} {row[0]:>8} {row[1]:>9} {row[2]:>9} {sum(row):>7}")

    (ASSAY_DIR / "_ASSAY_PHASE2_INLINE_MANIFEST.json").write_text(
        json.dumps({"total": grand_total, "cells": manifest_rows}, indent=2)
    )
    log(f"\nManifest → assay/_ASSAY_PHASE2_INLINE_MANIFEST.json")


if __name__ == "__main__":
    main()
