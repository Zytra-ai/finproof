"""
FinProof v1 — Official Scoring Service
Deploy at finproof.zytra.ai

POST /api/score   — submit predictions, receive Tier 2+3 public scores
GET  /leaderboard — public leaderboard (Tier 2+3 scores)
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json, hashlib, time, sqlite3
from pathlib import Path
from datetime import datetime, timezone

app = FastAPI(title="FinProof Scoring Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TIER2_PATH = Path("data/finproof_v1_tier2_public.jsonl")
BENIGN_PATH = Path("data/finproof_v1_tier1_benign.jsonl")
TIER4_SHA256 = "bf35df2e5a3f08c9202555db1a5bd825"  # withheld set — never evaluated here
DB_PATH = Path("leaderboard.db")

# ── Load eval data once ────────────────────────────────────────────────────────
tier2_rows = [json.loads(l) for l in TIER2_PATH.open()] if TIER2_PATH.exists() else []
benign_rows = [json.loads(l) for l in BENIGN_PATH.open()] if BENIGN_PATH.exists() else []

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS submissions (
        id TEXT PRIMARY KEY, model_name TEXT, model_org TEXT,
        hf_model_id TEXT, macro_f1 REAL, recall REAL, fpr REAL,
        miss_rate REAL, submitted_at TEXT, scores_json TEXT
    )""")
    con.commit(); con.close()

init_db()

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictionRow(BaseModel):
    id: str
    prediction: int  # 0=benign, 1=attack

class SubmissionRequest(BaseModel):
    model_name: str
    model_org: str
    hf_model_id: Optional[str] = ""
    predictions: list[PredictionRow]

# ── Scoring logic ─────────────────────────────────────────────────────────────
def score(predictions: dict[str, int]) -> dict:
    from collections import defaultdict, Counter

    cat_tp = defaultdict(int); cat_fp = defaultdict(int); cat_fn = defaultdict(int)
    all_tp = all_fp = all_fn = 0

    for row in tier2_rows:
        pred = predictions.get(row["id"], 0)
        cat = row["category"]
        if pred == 1: cat_tp[cat] += 1; all_tp += 1
        else:         cat_fn[cat] += 1; all_fn += 1

    fp_benign = sum(1 for r in benign_rows if predictions.get(r["id"], 0) == 1)
    all_fp = fp_benign
    fpr = round(fp_benign / len(benign_rows), 4) if benign_rows else 0

    def f1(tp, fp, fn):
        p = tp/(tp+fp) if (tp+fp) else 0
        r = tp/(tp+fn) if (tp+fn) else 0
        return round(2*p*r/(p+r), 4) if (p+r) else 0

    per_cat = {}
    f1s = []
    for cat in sorted(set(r["category"] for r in tier2_rows)):
        tp = cat_tp[cat]; fn = cat_fn[cat]
        f = f1(tp, 0, fn)
        per_cat[cat] = {"recall": round(tp/(tp+fn),4) if (tp+fn) else 0, "f1": f}
        f1s.append(f)

    macro_f1 = round(sum(f1s)/len(f1s), 4) if f1s else 0
    recall   = round(all_tp/(all_tp+all_fn), 4) if (all_tp+all_fn) else 0

    return {
        "macro_f1": macro_f1, "recall": recall, "fpr": fpr,
        "miss_rate_pct": round(100*(1-recall), 2),
        "per_category": per_cat,
        "note": "Scores on Tier 2 public split (1,606 attacks + 782 benign). Official Tier 4 evaluation by request only.",
        "tier4_sha256": TIER4_SHA256,
    }

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/api/score")
async def submit_score(submission: SubmissionRequest):
    preds = {p.id: p.prediction for p in submission.predictions}
    scores = score(preds)
    sub_id = hashlib.sha256(
        f"{submission.model_name}{time.time()}".encode()).hexdigest()[:12]

    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO submissions VALUES (?,?,?,?,?,?,?,?,?,?)", (
        sub_id, submission.model_name, submission.model_org,
        submission.hf_model_id, scores["macro_f1"], scores["recall"],
        scores["fpr"], scores["miss_rate_pct"],
        datetime.now(timezone.utc).isoformat(),
        json.dumps(scores)
    ))
    con.commit(); con.close()

    return {"submission_id": sub_id, "scores": scores}

@app.get("/leaderboard")
async def leaderboard():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT model_name, model_org, hf_model_id, macro_f1, recall,
               fpr, miss_rate_pct, submitted_at
        FROM submissions ORDER BY macro_f1 DESC LIMIT 50
    """).fetchall()
    con.close()

    return {"leaderboard": [
        {"rank": i+1, "model_name": r[0], "model_org": r[1],
         "hf_model_id": r[2], "macro_f1": r[3], "recall": r[4],
         "fpr": r[5], "miss_rate_pct": r[6], "submitted_at": r[7]}
        for i, r in enumerate(rows)
    ], "note": "Ranked by macro F1 on FinProof Tier 2 public split."}

@app.get("/health")
async def health():
    return {"status": "ok", "tier2_rows": len(tier2_rows), "benign_rows": len(benign_rows)}
