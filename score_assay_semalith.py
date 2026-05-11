#!/usr/bin/env python3
"""
score_assay_semalith.py — Run Semalith v1.5 on all 4,200 ASSAY prompts.

Outputs:
  assay/krait_qi/semalith_scores/
    all_scores.jsonl              — per-prompt: id, category, difficulty, p_benign, label, predicted_harmful
    score_distribution.json       — histogram + summary stats
    per_category_breakdown.json   — per B-01..B-07: mean, median, p_benign distribution
    per_difficulty_breakdown.json — per direct/indirect/advanced: distribution
    boundary_analysis.json        — prompts in each p_benign decile
"""

import json, os, sys, time
from pathlib import Path
from collections import defaultdict

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from assay.utils import ASSAY_DIR
from assay.qcbm_core import SemalithScorer, KRAIT_DIR

RAW_DIR   = ASSAY_DIR / "raw"
OUT_DIR   = KRAIT_DIR / "semalith_scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATS  = ["B-01","B-02","B-03","B-04","B-05","B-06","B-07"]
DIFFS = ["direct","indirect","advanced"]

LABEL_NAMES = {
    0:"BENIGN",1:"D1_SYSTEM_OVERRIDE",2:"D1_JAILBREAK",3:"D1_EXTRACTION",
    4:"D1_NARRATIVE_FRAME",5:"D1_AUTHORITY_CLAIM",6:"D1_SOCIAL_ENGINEERING",
    7:"D1_INDIRECT_INJECTION",8:"D5_SELF_REPLICATION",9:"D6_RESOURCE_ABUSE",
    10:"D8_GENERAL_HARM",11:"B-01",12:"B-02",13:"B-03",14:"B-04",
    15:"B-05",16:"B-06",17:"B-07",18:"B-08",19:"B-09",20:"B-10",21:"B-11",
}

SUPER_LABEL_NAME = {0:"BENIGN",1:"D-ATTACK",2:"GENERAL_HARM",3:"BFSI"}

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_corpus():
    rows = []
    for cat in CATS:
        for diff in DIFFS:
            path = RAW_DIR / f"{cat.replace('-','_')}_{diff}.jsonl"
            if not path.exists(): continue
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    r["_cat"] = cat
                    r["_diff"] = diff
                    rows.append(r)
    return rows


def run():
    t0 = time.time()
    log("="*60)
    log("Semalith v1.5 → ASSAY v1 Full Score Distribution")
    log("="*60)

    log("Loading corpus...")
    rows = load_corpus()
    log(f"  {len(rows)} prompts loaded")

    log("Loading Semalith v1.5...")
    scorer = SemalithScorer()
    log("  Model ready")

    log(f"Scoring {len(rows)} prompts (batch=32)...")
    texts = [r["input"] for r in rows]
    scores = scorer.score_batch(texts, batch_size=32)
    log(f"  Scoring done in {time.time()-t0:.1f}s")

    # Attach scores
    for i, r in enumerate(rows):
        s = scores[i]
        r["_p_benign"]        = s["p_benign"]
        r["_argmax_class"]    = s["argmax_class"]
        r["_predicted_label"] = LABEL_NAMES.get(s["argmax_class"], "UNKNOWN")
        r["_super_label"]     = SUPER_LABEL_NAME.get(s["super_label"], "UNKNOWN")
        r["_predicted_harmful"] = s["predicted_harmful"]

    p_benign_all = np.array([r["_p_benign"] for r in rows])

    # ── 1. Save all_scores.jsonl ─────────────────────────────────────
    log("Saving all_scores.jsonl...")
    with open(OUT_DIR / "all_scores.jsonl", "w") as f:
        for r in rows:
            rec = {
                "id":               r.get("id",""),
                "category":         r["_cat"],
                "difficulty":       r["_diff"],
                "input_preview":    r["input"][:120],
                "p_benign":         round(r["_p_benign"], 4),
                "predicted_label":  r["_predicted_label"],
                "super_label":      r["_super_label"],
                "predicted_harmful":r["_predicted_harmful"],
            }
            f.write(json.dumps(rec) + "\n")

    # ── 2. Score distribution histogram ─────────────────────────────
    log("Computing score distribution...")
    bins = np.arange(0, 1.05, 0.05)
    hist, edges = np.histogram(p_benign_all, bins=bins)
    histogram = [
        {"bin_start": round(float(edges[i]),2), "bin_end": round(float(edges[i+1]),2),
         "count": int(hist[i]), "pct": round(float(hist[i]/len(rows)*100),2)}
        for i in range(len(hist))
    ]

    # Boundary zones
    zones = {
        "highly_confident_harmful (p<0.05)":     int((p_benign_all < 0.05).sum()),
        "confident_harmful (0.05-0.20)":          int(((p_benign_all >= 0.05) & (p_benign_all < 0.20)).sum()),
        "leaning_harmful (0.20-0.35)":            int(((p_benign_all >= 0.20) & (p_benign_all < 0.35)).sum()),
        "boundary_zone (0.35-0.65)":              int(((p_benign_all >= 0.35) & (p_benign_all < 0.65)).sum()),
        "leaning_benign (0.65-0.80)":             int(((p_benign_all >= 0.65) & (p_benign_all < 0.80)).sum()),
        "confident_benign (0.80-0.95)":           int(((p_benign_all >= 0.80) & (p_benign_all < 0.95)).sum()),
        "highly_confident_benign (p>0.95)":       int((p_benign_all >= 0.95).sum()),
    }

    dist_out = {
        "total_prompts": len(rows),
        "summary": {
            "mean_p_benign":   round(float(p_benign_all.mean()), 4),
            "median_p_benign": round(float(np.median(p_benign_all)), 4),
            "std_p_benign":    round(float(p_benign_all.std()), 4),
            "pct_predicted_harmful": round(float((p_benign_all < 0.5).mean()*100), 2),
            "pct_boundary_zone":     round(float(((p_benign_all >= 0.35) & (p_benign_all < 0.65)).mean()*100), 2),
        },
        "zones": zones,
        "histogram_5pct_bins": histogram,
    }
    (OUT_DIR / "score_distribution.json").write_text(json.dumps(dist_out, indent=2))

    # ── 3. Per-category breakdown ────────────────────────────────────
    log("Computing per-category breakdown...")
    cat_out = {}
    for cat in CATS:
        cat_rows = [r for r in rows if r["_cat"] == cat]
        pb = np.array([r["_p_benign"] for r in cat_rows])
        pred_labels = [r["_predicted_label"] for r in cat_rows]
        label_counts = {}
        for l in pred_labels:
            label_counts[l] = label_counts.get(l, 0) + 1

        cat_out[cat] = {
            "n": len(cat_rows),
            "mean_p_benign":   round(float(pb.mean()), 4),
            "median_p_benign": round(float(np.median(pb)), 4),
            "std_p_benign":    round(float(pb.std()), 4),
            "pct_predicted_harmful": round(float((pb < 0.5).mean()*100), 2),
            "pct_boundary":    round(float(((pb >= 0.35) & (pb < 0.65)).mean()*100), 2),
            "n_boundary":      int(((pb >= 0.35) & (pb < 0.65)).sum()),
            "top_predicted_labels": dict(sorted(label_counts.items(), key=lambda x: -x[1])[:5]),
            "p_benign_deciles": {
                f"p{d}": round(float(np.percentile(pb, d)), 4)
                for d in [10, 25, 50, 75, 90, 95, 99]
            },
        }

    (OUT_DIR / "per_category_breakdown.json").write_text(json.dumps(cat_out, indent=2))

    # ── 4. Per-difficulty breakdown ──────────────────────────────────
    log("Computing per-difficulty breakdown...")
    diff_out = {}
    for diff in DIFFS:
        diff_rows = [r for r in rows if r["_diff"] == diff]
        pb = np.array([r["_p_benign"] for r in diff_rows])
        diff_out[diff] = {
            "n": len(diff_rows),
            "mean_p_benign":   round(float(pb.mean()), 4),
            "median_p_benign": round(float(np.median(pb)), 4),
            "std_p_benign":    round(float(pb.std()), 4),
            "pct_predicted_harmful": round(float((pb < 0.5).mean()*100), 2),
            "pct_boundary":    round(float(((pb >= 0.35) & (pb < 0.65)).mean()*100), 2),
            "n_boundary":      int(((pb >= 0.35) & (pb < 0.65)).sum()),
            "per_category": {
                cat: {
                    "mean_p_benign": round(float(np.mean([r["_p_benign"] for r in diff_rows if r["_cat"]==cat])), 4),
                    "n_boundary": int(sum(1 for r in diff_rows if r["_cat"]==cat and 0.35 <= r["_p_benign"] < 0.65)),
                }
                for cat in CATS
            },
        }

    (OUT_DIR / "per_difficulty_breakdown.json").write_text(json.dumps(diff_out, indent=2))

    # ── 5. Boundary analysis — prompts per decile ────────────────────
    log("Saving boundary analysis...")
    boundary_rows = [r for r in rows if 0.35 <= r["_p_benign"] < 0.65]
    boundary_rows.sort(key=lambda r: r["_p_benign"])

    # Split into decile buckets within the boundary zone
    boundaries_by_bucket = defaultdict(list)
    for r in boundary_rows:
        bucket = f"{int(r['_p_benign']*20)*5/100:.2f}-{(int(r['_p_benign']*20)+1)*5/100:.2f}"
        boundaries_by_bucket[bucket].append({
            "category": r["_cat"], "difficulty": r["_diff"],
            "p_benign": round(r["_p_benign"], 4),
            "predicted_label": r["_predicted_label"],
            "input_preview": r["input"][:150],
        })

    (OUT_DIR / "boundary_analysis.json").write_text(json.dumps({
        "n_boundary": len(boundary_rows),
        "pct_of_corpus": round(len(boundary_rows)/len(rows)*100, 2),
        "by_bucket": dict(sorted(boundaries_by_bucket.items())),
    }, indent=2))

    # ── Summary print ────────────────────────────────────────────────
    elapsed = time.time() - t0
    log(f"\n{'='*60}")
    log(f"Semalith Scoring Complete — {elapsed:.1f}s")
    log(f"{'='*60}")
    log(f"  Total prompts scored:    {len(rows)}")
    log(f"  Mean p_benign:           {p_benign_all.mean():.4f}")
    log(f"  Median p_benign:         {np.median(p_benign_all):.4f}")
    log(f"  Predicted harmful:       {(p_benign_all < 0.5).sum()} ({(p_benign_all < 0.5).mean()*100:.1f}%)")
    log(f"  Boundary zone [0.35,0.65]: {((p_benign_all >= 0.35) & (p_benign_all < 0.65)).sum()} ({((p_benign_all >= 0.35) & (p_benign_all < 0.65)).mean()*100:.1f}%)")
    log(f"\n  Per-category:")
    for cat, v in cat_out.items():
        log(f"    {cat}  mean_p={v['mean_p_benign']:.3f}  boundary={v['n_boundary']:3d}  harmful={v['pct_predicted_harmful']}%")
    log(f"\n  Per-difficulty:")
    for diff, v in diff_out.items():
        log(f"    {diff:<10}  mean_p={v['mean_p_benign']:.3f}  boundary={v['n_boundary']:3d}  harmful={v['pct_predicted_harmful']}%")
    log(f"\n  Output: {OUT_DIR}")


if __name__ == "__main__":
    run()
