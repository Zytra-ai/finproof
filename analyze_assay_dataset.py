#!/usr/bin/env python3
"""
analyze_assay_dataset.py — ASSAY v1 Quality & Fitment Analysis

Quality dimensions:
  1. Coverage         — cell counts vs. target
  2. Length           — token distribution per cell
  3. Linguistic diversity — TTR, unique bigrams, vocabulary richness
  4. Difficulty gradient — complexity proxy per difficulty level
  5. Financial entity density — entities per prompt
  6. Near-dup rate    — within-cell Jaccard similarity
  7. Regulatory anchor coverage
  8. Sample spot-check — representative prompts per cell

Fitment dimensions:
  9. Category balance  — is any category starved?
  10. Difficulty split  — direct:indirect:advanced ratio
  11. Length fit        — does length distribution match ASSAY spec (20–400 tokens)?
  12. Domain authenticity — % prompts with ≥2 financial entities
  13. Contamination confirmation — 0 SHA collisions with eval JSONLs
"""

import json, re, sys, math
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from assay.utils import (
    sha1, build_sha_index, token_count, has_financial_entity,
    RE_FINANCIAL_ENTITY, EVAL_DIR, ASSAY_DIR
)

RAW_DIR = ASSAY_DIR / "raw"
TARGET_PER_CELL = 200
CATEGORIES   = ["B-01","B-02","B-03","B-04","B-05","B-06","B-07"]
DIFFICULTIES = ["direct","indirect","advanced"]

# ── Load all cells ─────────────────────────────────────────────────────────────
cells = {}
all_rows = []
for cat in CATEGORIES:
    for diff in DIFFICULTIES:
        key = f"{cat}-{diff}"
        path = RAW_DIR / f"{cat.replace('-','_')}_{diff}.jsonl"
        if path.exists():
            rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        else:
            rows = []
        cells[key] = rows
        all_rows.extend(rows)

# ── Helpers ────────────────────────────────────────────────────────────────────
def entity_count(text):
    return len(RE_FINANCIAL_ENTITY.findall(text))

def bigrams(text):
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    return set(zip(words, words[1:]))

def ttr(text):
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    return len(set(words)) / max(len(words), 1)

def avg_sentence_length(text):
    sentences = re.split(r'[.!?]', text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    return sum(lengths) / max(len(lengths), 1)

def complexity_score(text):
    """Proxy for complexity: avg sentence length + unique entity count + log(token_count)."""
    return avg_sentence_length(text) * 0.4 + entity_count(text) * 1.5 + math.log(max(token_count(text), 1)) * 2

# ── 1. Coverage ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("ASSAY v1 — Quality & Fitment Analysis")
print("=" * 70)

total = len(all_rows)
print(f"\n▶ 1. COVERAGE  (target: {len(cells)} cells × {TARGET_PER_CELL} = "
      f"{len(cells)*TARGET_PER_CELL})")
print(f"   Total prompts accepted: {total:,}")
print(f"   Fill rate: {total/(len(cells)*TARGET_PER_CELL)*100:.1f}%\n")

print(f"   {'Cell':<18} {'Count':>6} {'Target':>7} {'Fill%':>6}  Status")
print(f"   {'─'*50}")
for cat in CATEGORIES:
    row_counts = [len(cells[f"{cat}-{d}"]) for d in DIFFICULTIES]
    cat_total  = sum(row_counts)
    fill       = cat_total / (TARGET_PER_CELL * 3) * 100
    status     = "✓ OK" if fill >= 60 else ("△ LOW" if fill >= 30 else "✗ THIN")
    print(f"   {cat:<18} {cat_total:>6} {TARGET_PER_CELL*3:>7} {fill:>5.0f}%  {status}")
    for diff, cnt in zip(DIFFICULTIES, row_counts):
        pct = cnt / TARGET_PER_CELL * 100
        bar = "▓" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"      {diff:<14} {cnt:>4}/{TARGET_PER_CELL}  {bar} {pct:.0f}%")

# ── 2. Token length distribution ───────────────────────────────────────────────
print(f"\n▶ 2. TOKEN LENGTH DISTRIBUTION  (spec: 8–400)")
lengths = [token_count(r["input"]) for r in all_rows]
buckets = Counter()
for l in lengths:
    if   l <  20: buckets["<20"] += 1
    elif l <  50: buckets["20-49"] += 1
    elif l < 100: buckets["50-99"] += 1
    elif l < 200: buckets["100-199"] += 1
    elif l < 400: buckets["200-399"] += 1
    else:         buckets["≥400"] += 1

print(f"   Min={min(lengths)}  Max={max(lengths)}  "
      f"Median={sorted(lengths)[len(lengths)//2]}  "
      f"Mean={sum(lengths)/len(lengths):.0f}")
print(f"   Distribution:")
for bucket in ["<20","20-49","50-99","100-199","200-399","≥400"]:
    n   = buckets.get(bucket, 0)
    pct = n / total * 100
    bar = "█" * int(pct / 2)
    print(f"   {bucket:<8} {n:>5} ({pct:>5.1f}%)  {bar}")

# ── 3. Linguistic diversity ────────────────────────────────────────────────────
print(f"\n▶ 3. LINGUISTIC DIVERSITY")
all_bigrams    = set()
ttr_scores     = []
for r in all_rows:
    all_bigrams.update(bigrams(r["input"]))
    ttr_scores.append(ttr(r["input"]))

avg_ttr        = sum(ttr_scores) / len(ttr_scores)
print(f"   Unique bigrams across corpus:  {len(all_bigrams):,}")
print(f"   Mean Type-Token Ratio (TTR):   {avg_ttr:.3f}  (1.0=all unique, 0=all repeated)")
print(f"   Vocabulary richness score:     {'HIGH' if avg_ttr > 0.7 else 'MEDIUM' if avg_ttr > 0.5 else 'LOW'}")

# Within-cell bigram overlap (redundancy indicator)
print(f"\n   Within-cell bigram overlap (lower = more diverse):")
for cat in CATEGORIES:
    for diff in DIFFICULTIES:
        rows = cells[f"{cat}-{diff}"]
        if len(rows) < 2: continue
        all_bg = [bigrams(r["input"]) for r in rows]
        pairs  = 0; overlaps = 0
        for i in range(min(len(all_bg), 20)):
            for j in range(i+1, min(len(all_bg), 20)):
                union = all_bg[i] | all_bg[j]
                inter = all_bg[i] & all_bg[j]
                if union:
                    overlaps += len(inter) / len(union)
                    pairs += 1
        avg_jacc = overlaps / max(pairs, 1)
        flag     = " ⚠ HIGH OVERLAP" if avg_jacc > 0.25 else ""
        print(f"   {cat}-{diff:<12} avg Jaccard={avg_jacc:.3f}{flag}")

# ── 4. Difficulty gradient ─────────────────────────────────────────────────────
print(f"\n▶ 4. DIFFICULTY GRADIENT  (should increase: direct < indirect < advanced)")
for cat in CATEGORIES:
    scores = {}
    for diff in DIFFICULTIES:
        rows = cells[f"{cat}-{diff}"]
        if rows:
            scores[diff] = sum(complexity_score(r["input"]) for r in rows) / len(rows)
        else:
            scores[diff] = 0
    gradient_ok = scores.get("direct",0) <= scores.get("indirect",0) <= scores.get("advanced",0)
    flag = "✓" if gradient_ok else "✗ INVERTED"
    print(f"   {cat}  direct={scores.get('direct',0):.1f}  "
          f"indirect={scores.get('indirect',0):.1f}  "
          f"advanced={scores.get('advanced',0):.1f}  {flag}")

# ── 5. Financial entity density ────────────────────────────────────────────────
print(f"\n▶ 5. FINANCIAL ENTITY DENSITY")
entity_counts = [entity_count(r["input"]) for r in all_rows]
zero_entity   = sum(1 for c in entity_counts if c == 0)
one_entity    = sum(1 for c in entity_counts if c == 1)
two_plus      = sum(1 for c in entity_counts if c >= 2)
print(f"   0 entities:   {zero_entity:>5} ({zero_entity/total*100:.1f}%)  ← these should be 0 post-gate")
print(f"   1 entity:     {one_entity:>5} ({one_entity/total*100:.1f}%)")
print(f"   ≥2 entities:  {two_plus:>5} ({two_plus/total*100:.1f}%)  ← richer BFSI grounding")
print(f"   Mean entities per prompt: {sum(entity_counts)/total:.1f}")
print(f"   Domain authenticity (≥2 entities): "
      f"{'HIGH' if two_plus/total > 0.6 else 'MEDIUM' if two_plus/total > 0.4 else 'LOW'}")

# ── 6. Contamination check ─────────────────────────────────────────────────────
print(f"\n▶ 6. CONTAMINATION CHECK")
sha_index = build_sha_index()
contaminated = sum(1 for r in all_rows if sha1(r["input"]) in sha_index)
print(f"   Eval SHA collisions: {contaminated}  "
      f"({'✓ CLEAN' if contaminated == 0 else '✗ CONTAMINATED'})")
print(f"   Eval files checked: {len(list(EVAL_DIR.glob('*.jsonl')))}")

# ── 7. Regulatory anchor coverage ──────────────────────────────────────────────
print(f"\n▶ 7. REGULATORY ANCHOR COVERAGE")
anchor_map = {
    "B-01": ["SEBI","IA Regulation","investment advis","RIA"],
    "B-02": ["KYC","AML","PMLA","CTR","SAR","mandate"],
    "B-03": ["SEBI","RBI","circular","regulation","Regulation","amendment"],
    "B-04": ["circular","Master Direction","Notification","guideline"],
    "B-05": ["DPDP","data","consent","personal","privacy"],
    "B-06": ["NACH","NEFT","RTGS","IMPS","UPI","SWIFT","payment","mandate"],
    "B-07": ["bank statement","PAN","Aadhaar","KYC","document","loan","forged","fake","counterfeit"],
}
for cat in CATEGORIES:
    rows = [r for r in all_rows if r["category"] == cat]
    if not rows: continue
    keywords = anchor_map.get(cat, [])
    hits = sum(1 for r in rows
               if any(kw.lower() in r["input"].lower() for kw in keywords))
    pct  = hits / len(rows) * 100
    print(f"   {cat}: {hits}/{len(rows)} ({pct:.0f}%) contain anchor keywords  "
          f"{'✓' if pct > 70 else '△' if pct > 50 else '✗'}")

# ── 8. Sample spot-check ───────────────────────────────────────────────────────
print(f"\n▶ 8. SAMPLE SPOT-CHECK (3 prompts per category, one per difficulty)")
for cat in CATEGORIES:
    print(f"\n   [{cat}]")
    for diff in DIFFICULTIES:
        rows = cells[f"{cat}-{diff}"]
        if rows:
            sample = rows[len(rows)//2]["input"]  # middle of the cell
            print(f"   {diff:>10}: {sample[:120]}{'...' if len(sample)>120 else ''}")

# ── 9. Fitment summary ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("FITMENT SUMMARY")
print(f"{'='*70}")

cat_totals = {cat: sum(len(cells[f"{cat}-{d}"]) for d in DIFFICULTIES) for cat in CATEGORIES}
diff_totals = {diff: sum(len(cells[f"{cat}-{diff}"]) for cat in CATEGORIES) for diff in DIFFICULTIES}

print(f"\n   Per-category totals:")
for cat, n in sorted(cat_totals.items()):
    bar = "█" * (n // 20)
    print(f"   {cat}: {n:>4}  {bar}")

print(f"\n   Per-difficulty totals:")
for diff, n in diff_totals.items():
    print(f"   {diff:<12}: {n:>4}  ({n/total*100:.0f}%)")

print(f"\n   Overall fill rate vs. 4,200 target: {total/4200*100:.1f}%")
print(f"   Gap to target: {4200-total} prompts")

# ── 10. Verdict ────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")

issues   = []
warnings = []
passes   = []

if total >= 1500:
    passes.append(f"Volume: {total:,} prompts accepted (sufficient for Phase 3 benign + integration)")
else:
    issues.append(f"Volume: {total:,} prompts — below minimum viable dataset threshold of 1,500")

if contaminated == 0:
    passes.append("Contamination: 0 eval collisions — clean for evaluation use")
else:
    issues.append(f"Contamination: {contaminated} eval collisions detected — must resolve before eval use")

if avg_ttr > 0.65:
    passes.append(f"Linguistic diversity: TTR={avg_ttr:.3f} — adequate variety in phrasing")
elif avg_ttr > 0.50:
    warnings.append(f"Linguistic diversity: TTR={avg_ttr:.3f} — some repetition, acceptable for training")
else:
    issues.append(f"Linguistic diversity: TTR={avg_ttr:.3f} — high repetition, needs regeneration")

fill_rate = total / (len(cells) * TARGET_PER_CELL)
if fill_rate >= 0.60:
    passes.append(f"Cell fill rate: {fill_rate*100:.0f}% — adequate coverage across all 21 cells")
else:
    warnings.append(f"Cell fill rate: {fill_rate*100:.0f}% — thin cells need second-pass generation")

thin_cats = [cat for cat, n in cat_totals.items() if n < 100]
if not thin_cats:
    passes.append("Category balance: all 7 categories have ≥100 accepted prompts")
else:
    warnings.append(f"Thin categories: {thin_cats} have <100 prompts each")

thin_cells = [k for k, v in cells.items() if len(v) < 40]
if thin_cells:
    warnings.append(f"Thin cells (<40 prompts): {thin_cells}")

domain_auth = two_plus / total
if domain_auth >= 0.60:
    passes.append(f"Domain authenticity: {domain_auth*100:.0f}% of prompts have ≥2 financial entities")
else:
    warnings.append(f"Domain authenticity: {domain_auth*100:.0f}% with ≥2 entities — some prompts are generic")

# Difficulty gradient check
gradient_failures = []
for cat in CATEGORIES:
    scores = {d: sum(complexity_score(r["input"]) for r in cells[f"{cat}-{d}"])/max(len(cells[f"{cat}-{d}"]),1)
              for d in DIFFICULTIES}
    if not (scores["direct"] <= scores["indirect"] <= scores["advanced"]):
        gradient_failures.append(cat)
if not gradient_failures:
    passes.append("Difficulty gradient: all categories show correct direct < indirect < advanced complexity")
else:
    warnings.append(f"Difficulty gradient inverted in: {gradient_failures}")

print(f"\n  ✓ PASS ({len(passes)})")
for p in passes: print(f"    • {p}")
print(f"\n  △ WARNING ({len(warnings)})")
for w in warnings: print(f"    • {w}")
if issues:
    print(f"\n  ✗ FAIL ({len(issues)})")
    for i in issues: print(f"    • {i}")

overall = "PASS — SUITABLE FOR TRAINING USE" if not issues else "CONDITIONAL — RESOLVE ISSUES BEFORE EVAL USE"
print(f"\n  OVERALL: {overall}")
print(f"  Dataset phase: {'READY for Phase 3 (benign generation) + corpus integration' if not issues else 'NEEDS REMEDIATION'}")
print(f"  Second-pass generation needed: YES — {4200-total} more attack prompts to reach full target")
print("="*70)
