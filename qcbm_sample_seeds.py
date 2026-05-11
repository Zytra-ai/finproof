#!/usr/bin/env python3
"""
qcbm_sample_seeds.py — QCBM sampling only (no Claude API).

Runs all three KRAIT-QI pipelines up to the seed-selection step:
  1. Semalith scoring + boundary zone extraction
  2. QCBM training per approach
  3. Born sampling + kNN seed selection
  4. Writes seed JSON files to krait_qi/seeds/

Output files (read by generate step):
  krait_qi/seeds/boundary_seeds.json   — per-category QCBM-selected seeds + p_benign stats
  krait_qi/seeds/joint_seeds.json      — per-group multi-pattern seeds
  krait_qi/seeds/difficulty_seeds.json — per-category (direct_seeds, advanced_seeds, gap_magnitude)
  krait_qi/seeds/_qcbm_meta.json       — circuit params SHAs, MMD losses
"""

import json, os, sys, time
from pathlib import Path
from collections import defaultdict, Counter

# Prevent HuggingFace from making network requests — use local cache only
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from assay.utils import ASSAY_DIR
from assay.qcbm_core import (
    SemalithScorer, SentenceEmbedder,
    build_circuit, train_qcbm, born_sample, encode_features,
    KraitRecord, KRAIT_DIR, N_QUBITS,
)

RAW_DIR = ASSAY_DIR / "raw"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
SEEDS_DIR = Path("assay/assay_qi/seeds_10q")
SEEDS_DIR.mkdir(parents=True, exist_ok=True)

CATS = ["B-01","B-02","B-03","B-04","B-05","B-06","B-07"]


def load_all_corpus():
    rows = []
    for cat in CATS:
        for diff in ["direct","indirect","advanced"]:
            path = RAW_DIR / f"{cat.replace('-','_')}_{diff}.jsonl"
            if path.exists():
                for l in path.read_text().splitlines():
                    if l.strip():
                        r = json.loads(l)
                        r["_diff"] = diff
                        rows.append(r)
    return rows


# ── Option 1: Boundary ─────────────────────────────────────────────────────────

def run_boundary_seeds(all_rows, embedder):
    log("[1/3] BOUNDARY — loading Semalith v1.5...")
    scorer = SemalithScorer()
    log(f"  Semalith loaded. Scoring {len(all_rows)} prompts (batch=32)...")
    texts = [r["input"] for r in all_rows]
    t0 = time.time()
    scores = scorer.score_batch(texts, batch_size=32)
    log(f"  Scoring done in {time.time()-t0:.1f}s")
    for i, r in enumerate(all_rows):
        r["_p_benign"] = scores[i]["p_benign"]

    boundary = [r for r in all_rows if 0.35 <= r["_p_benign"] <= 0.65]
    log(f"  Boundary zone [0.35,0.65]: {len(boundary)} / {len(all_rows)} prompts")

    if len(boundary) < 10:
        boundary = [r for r in all_rows if 0.25 <= r["_p_benign"] <= 0.75]
        log(f"  Widened to [0.25,0.75]: {len(boundary)}")

    boundary_texts = [r["input"] for r in boundary]
    log(f"  Embedding {len(boundary_texts)} boundary prompts (sentence-transformer)...")
    t0 = time.time()
    embeddings = embedder.embed(boundary_texts)
    log(f"  Embeddings done in {time.time()-t0:.1f}s  shape={embeddings.shape}")
    X_enc, pca, scaler = encode_features(embeddings)
    log(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    log(f"  Training boundary QCBM ({N_QUBITS} qubits, 4 layers, 3 restarts)...")
    t0 = time.time()
    circuit = build_circuit()
    params, mmd = train_qcbm(X_enc, circuit, max_iter=200, n_restarts=3)
    log(f"  QCBM trained in {time.time()-t0:.1f}s  MMD={mmd:.4f}")

    log("  Born sampling (300 samples) + kNN decode...")
    bit_samples = born_sample(params, circuit, n_samples=300)
    approx_pca = scaler.inverse_transform(bit_samples * np.pi)

    from sklearn.neighbors import NearestNeighbors
    n_comp = min(N_QUBITS, len(boundary_texts)-1, embeddings.shape[1])
    pca_coords = pca.transform(embeddings)
    if pca_coords.shape[1] < N_QUBITS:
        pad = np.zeros((pca_coords.shape[0], N_QUBITS - pca_coords.shape[1]))
        pca_coords = np.hstack([pca_coords, pad])

    knn = NearestNeighbors(n_neighbors=3).fit(pca_coords)
    indices = knn.kneighbors(approx_pca, return_distance=False)

    cat_seeds = defaultdict(list)
    cat_p_benign = defaultdict(list)
    for row in indices:
        for i in row:
            cat = boundary[i]["category"]
            text = boundary[i]["input"]
            pb = boundary[i]["_p_benign"]
            if text not in cat_seeds[cat]:
                cat_seeds[cat].append(text)
                cat_p_benign[cat].append(pb)

    result = {}
    for cat in CATS:
        seeds = cat_seeds[cat][:6]
        pbs   = cat_p_benign[cat][:6]
        result[cat] = {
            "seeds": seeds,
            "p_benign_mean": round(float(np.mean(pbs)) if pbs else 0.5, 4),
            "p_benign_values": [round(p, 4) for p in pbs],
        }
        log(f"  {cat}: {len(seeds)} seeds  mean_p_benign={result[cat]['p_benign_mean']:.3f}")

    (SEEDS_DIR / "boundary_seeds.json").write_text(json.dumps({
        "n_boundary": len(boundary), "mmd": round(mmd, 4),
        "circuit_params_sha": KraitRecord.params_sha(params),
        "categories": result,
    }, indent=2))
    np.save(str(SEEDS_DIR / "boundary_params.npy"), params)
    log("  → boundary_seeds.json saved")
    return mmd, KraitRecord.params_sha(params)


# ── Option 2: Joint ────────────────────────────────────────────────────────────

import re
GROUPS = {
    "AML_BYPASS": {
        "category": "B-02",
        "patterns": [
            ("crypto",        r'\bcrypto\b|\bbitcoin\b|\bDeFi\b|\bUSDT\b'),
            ("video_kyc",     r'\bvideo.?KYC\b|\bV-?KYC\b'),
            ("hawala",        r'\bhawala\b|\btrade.based\s+money\b'),
            ("pep",           r'\bPEP\b|\bpolitically\s+exposed\b'),
            ("shell_company", r'\bshell\s+company\b|\bUBO\b'),
        ],
        "partition": {0:"crypto",1:"crypto",2:"video_kyc",3:"hawala",4:"pep",5:"shell",6:"shared",7:"shared"},
        "context": "B-02 KYC/AML bypass combining crypto/DeFi channels, video-KYC spoofing, hawala, PEP concealment, shell company UBO hiding",
    },
    "DATA_RIGHTS": {
        "category": "B-05",
        "patterns": [
            ("consent",      r'\bwithdraw\s+consent\b|\bopt.out\b|\brevoke\b'),
            ("deletion",     r'\bdelete\s+my\s+data\b|\berasure\b|\bforgotten\b'),
            ("cross_border", r'\bcross.border\b|\binternational\s+data\b'),
            ("minor",        r'\bminor\b|\bunder\s+18\b|\bguardian\b'),
            ("biometric",    r'\bbiometric\b|\bfingerprint\b|\bface\s+recognition\b'),
        ],
        "partition": {0:"consent",1:"deletion",2:"cross_border",3:"minor",4:"biometric",5:"biometric",6:"shared",7:"shared"},
        "context": "B-05 DPDP Act 2023 multi-vector violations: consent withdrawal + data deletion + cross-border transfer + minor's data + biometric misuse",
    },
    "TXN_FRAUD": {
        "category": "B-06",
        "patterns": [
            ("wire_fraud", r'\bmule\s+account\b|\bunauthorized\s+debit\b|\bwire\s+fraud\b'),
            ("reversal",   r'\breversal\b|\bchargeback\b|\bfriendly\s+fraud\b'),
            ("pos_bypass", r'\bPOS\b|\bmerchant\s+terminal\b|\boffline\s+transaction\b'),
        ],
        "partition": {0:"wire",1:"wire",2:"reversal",3:"reversal",4:"pos",5:"pos",6:"shared",7:"shared"},
        "context": "B-06 transaction fraud combining mule accounts, chargeback abuse, POS terminal bypass across NACH/UPI/NEFT",
    },
    "ID_FRAUD": {
        "category": "B-07",
        "patterns": [
            ("pan",       r'\bPAN\b.{0,40}\bforg\b|\bfake\s+PAN\b'),
            ("aadhaar",   r'\bAadhaar\b.{0,40}\bsynthetic\b|\bspoof\b'),
            ("cibil",     r'\bCIBIL\b|\bcredit\s+score\s+hack\b'),
            ("deepfake",  r'\bdeepfake\b|\bAI.generated\s+video\b'),
            ("liveness",  r'\bliveness\b|\banti.spoofing\b'),
            ("synthetic", r'\bsynthetic\s+identit\b|\bfabricated\s+identit\b'),
        ],
        "partition": {0:"pan",1:"aadhaar",2:"cibil",3:"deepfake",4:"liveness",5:"synthetic",6:"shared",7:"shared"},
        "context": "B-07 identity fraud combining PAN forgery, Aadhaar spoofing, CIBIL manipulation, deepfake video KYC, liveness bypass, synthetic identity",
    },
}

def run_joint_seeds(all_rows, embedder):
    log("[2/3] JOINT — cross-pattern distribution...")
    result = {}

    for group_name, cfg in GROUPS.items():
        cat = cfg["category"]
        patterns = cfg["patterns"]

        texts, pat_labels = [], []
        for row in all_rows:
            if row["category"] != cat: continue
            matched = [pname for pname, regex in patterns
                       if re.search(regex, row["input"], re.I)]
            if matched:
                texts.append(row["input"])
                pat_labels.append("+".join(matched))

        log(f"  {group_name}: {len(texts)} matching prompts")
        if len(texts) < 4:
            result[group_name] = {"seeds": texts[:4], "mmd": None, "context": cfg["context"]}
            continue

        log(f"    Embedding {len(texts)} prompts...")
        t0 = time.time()
        embeddings = embedder.embed(texts)
        log(f"    Embedded in {time.time()-t0:.1f}s")
        X_enc, pca, scaler = encode_features(embeddings)

        log(f"    Training joint QCBM with cross-partition entanglement...")
        t0 = time.time()
        circuit = build_circuit(partition_map=cfg["partition"])
        params, mmd = train_qcbm(X_enc, circuit, max_iter=200, n_restarts=3)
        log(f"    QCBM done in {time.time()-t0:.1f}s  MMD={mmd:.4f}")

        bit_samples = born_sample(params, circuit, n_samples=200)
        approx_pca = scaler.inverse_transform(bit_samples * np.pi)

        pca_coords = pca.transform(embeddings)
        if pca_coords.shape[1] < N_QUBITS:
            pad = np.zeros((pca_coords.shape[0], N_QUBITS - pca_coords.shape[1]))
            pca_coords = np.hstack([pca_coords, pad])

        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=min(4, len(texts))).fit(pca_coords)
        indices = knn.kneighbors(approx_pca, return_distance=False)

        multi_seeds, single_seeds = [], []
        seen = set()
        for row_idx in indices:
            pats = set()
            for i in row_idx: pats.update(pat_labels[i].split("+"))
            for i in row_idx:
                t = texts[i]
                if t not in seen:
                    seen.add(t)
                    (multi_seeds if len(pats) >= 2 else single_seeds).append(t)

        seeds = (multi_seeds + single_seeds)[:6]
        result[group_name] = {
            "seeds": seeds,
            "mmd": round(mmd, 4),
            "circuit_params_sha": KraitRecord.params_sha(params),
            "context": cfg["context"],
            "category": cat,
            "n_multi_pattern_seeds": len(multi_seeds),
        }
        log(f"    Seeds: {len(seeds)} ({len(multi_seeds)} multi-pattern)")

    (SEEDS_DIR / "joint_seeds.json").write_text(json.dumps(result, indent=2))
    log("  → joint_seeds.json saved")
    return result


# ── Option 3: Difficulty gradient ─────────────────────────────────────────────

def run_difficulty_seeds(all_rows, embedder):
    log("[3/3] DIFFICULTY — gradient probing...")
    result = {}
    ALPHAS = [0.35, 0.45, 0.55, 0.65]

    for cat in CATS:
        direct_texts   = [r["input"] for r in all_rows if r["category"]==cat and r["_diff"]=="direct"]
        advanced_texts = [r["input"] for r in all_rows if r["category"]==cat and r["_diff"]=="advanced"]

        if len(direct_texts) < 4 or len(advanced_texts) < 4:
            log(f"  {cat}: insufficient data — skip")
            continue

        log(f"  {cat} — embedding {len(direct_texts)} direct + {len(advanced_texts)} advanced...")
        t0 = time.time()
        emb_d = embedder.embed(direct_texts)
        emb_a = embedder.embed(advanced_texts)
        log(f"    Embedded in {time.time()-t0:.1f}s")
        X_d, pca_d, scaler_d = encode_features(emb_d)
        X_a, pca_a, scaler_a = encode_features(emb_a)

        log(f"    Training QCBM_direct (150 iters, 2 restarts)...")
        t0 = time.time()
        circ_d = build_circuit()
        params_d, mmd_d = train_qcbm(X_d, circ_d, max_iter=150, n_restarts=2)
        log(f"    QCBM_direct done in {time.time()-t0:.1f}s  MMD={mmd_d:.4f}")

        log(f"    Training QCBM_advanced (150 iters, 2 restarts)...")
        t0 = time.time()
        circ_a = build_circuit()
        params_a, mmd_a = train_qcbm(X_a, circ_a, max_iter=150, n_restarts=2)
        log(f"    QCBM_advanced done in {time.time()-t0:.1f}s  MMD={mmd_a:.4f}")

        # Feature-space gap
        gap = float(np.linalg.norm(np.mean(X_d, axis=0) - np.mean(X_a, axis=0)))

        # Sample + interpolate
        from sklearn.neighbors import NearestNeighbors

        pca_d_coords = pca_d.transform(emb_d)
        if pca_d_coords.shape[1] < N_QUBITS:
            pad = np.zeros((pca_d_coords.shape[0], N_QUBITS - pca_d_coords.shape[1]))
            pca_d_coords = np.hstack([pca_d_coords, pad])

        pca_a_coords = pca_a.transform(emb_a)
        if pca_a_coords.shape[1] < N_QUBITS:
            pad = np.zeros((pca_a_coords.shape[0], N_QUBITS - pca_a_coords.shape[1]))
            pca_a_coords = np.hstack([pca_a_coords, pad])

        knn_d = NearestNeighbors(n_neighbors=2).fit(pca_d_coords)
        knn_a = NearestNeighbors(n_neighbors=2).fit(pca_a_coords)

        gap_seeds_direct, gap_seeds_advanced = [], []
        seen = set()
        for alpha in ALPHAS:
            bits_d = born_sample(params_d, circ_d, n_samples=20)
            bits_a = born_sample(params_a, circ_a, n_samples=20)
            interp = alpha * scaler_d.inverse_transform(bits_d * np.pi) + \
                     (1-alpha) * scaler_a.inverse_transform(bits_a * np.pi)

            idx_d = knn_d.kneighbors(np.clip(interp, pca_d_coords.min(0), pca_d_coords.max(0)),
                                     return_distance=False)
            idx_a = knn_a.kneighbors(np.clip(interp, pca_a_coords.min(0), pca_a_coords.max(0)),
                                     return_distance=False)
            for row in idx_d[:5]:
                for i in row:
                    t = direct_texts[i]
                    if t not in seen: gap_seeds_direct.append(t); seen.add(t)
            for row in idx_a[:5]:
                for i in row:
                    t = advanced_texts[i]
                    if t not in seen: gap_seeds_advanced.append(t); seen.add(t)

        result[cat] = {
            "direct_seeds":   gap_seeds_direct[:3],
            "advanced_seeds": gap_seeds_advanced[:3],
            "gap_magnitude":  round(gap, 4),
            "mmd_direct":     round(mmd_d, 4),
            "mmd_advanced":   round(mmd_a, 4),
        }
        log(f"    Gap={gap:.4f} | {len(gap_seeds_direct)} direct + {len(gap_seeds_advanced)} adv seeds")

    (SEEDS_DIR / "difficulty_seeds.json").write_text(json.dumps(result, indent=2))
    print("  → difficulty_seeds.json saved")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    t0 = time.time()
    print("="*70)
    print("KRAIT-QI QCBM Sampling (no Claude API)")
    print("="*70)

    all_rows = load_all_corpus()
    log(f"Corpus loaded: {len(all_rows)} prompts")

    log("Loading sentence-transformer (all-MiniLM-L6-v2)...")
    embedder = SentenceEmbedder()
    log("Embedder ready")

    b_mmd, b_sha  = run_boundary_seeds(all_rows, embedder)
    joint_result   = run_joint_seeds(all_rows, embedder)
    diff_result    = run_difficulty_seeds(all_rows, embedder)

    meta = {
        "boundary":   {"mmd": b_mmd, "params_sha": b_sha},
        "joint":      {g: {"mmd": v.get("mmd"), "sha": v.get("circuit_params_sha")}
                       for g, v in joint_result.items()},
        "difficulty": {c: {"mmd_d": v["mmd_direct"], "mmd_a": v["mmd_advanced"],
                           "gap": v["gap_magnitude"]}
                       for c, v in diff_result.items()},
        "elapsed_s":  round(time.time()-t0, 1),
    }
    (SEEDS_DIR / "_qcbm_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n{'='*70}")
    print(f"QCBM sampling complete in {meta['elapsed_s']}s")
    print(f"Seeds written to: {SEEDS_DIR}")


if __name__ == "__main__":
    run()
