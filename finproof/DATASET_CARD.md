---
license: cc-by-4.0
task_categories:
- text-classification
language:
- en
tags:
- bfsi
- guardrail
- safety
- adversarial
- prompt-injection
- financial-services
- llm-safety
- benchmark
- responsible-ai
- india
pretty_name: FinProof v1 — BFSI Adversarial Benchmark
size_categories:
- 1K<n<10K
---

# FinProof v1 — BFSI Adversarial Benchmark

**FinProof** is the first open adversarial benchmark for AI guardrail systems deployed in banking, financial services, and insurance (BFSI). It covers 7 attack categories across professional and retail conversational registers, with quantum-augmented generation for distribution coverage.

Published by [Zytra](https://huggingface.co/Zytra) · License: CC BY 4.0

---

## What's in this repository

| File | Description |
|---|---|
| `finproof_v1_tier2_public.jsonl` | **1,606 easy-difficulty attack prompts** (Tier 2 — public) |
| `eval/scoring.py` | Per-category F1, recall, FPR computation |
| `eval/run_eval.py` | Local evaluation harness (Apache 2.0) |
| `eval/contamination_audit.py` | SHA-1 + MinHash dedup check for training corpora |
| `eval/submit.py` | Submission packager for finproof.zytra.ai |

### What's NOT here (by design)

- **Tier 3 (2,036 medium prompts)** — available under FINPROOF Research Agreement (non-commercial, no training use). Email finproof@zytra.ai.
- **Tier 4 (1,747 hard prompts)** — withheld permanently. Official leaderboard scores at finproof.zytra.ai use this set exclusively.

---

## Attack Categories

| ID | Category | Regulatory Anchor |
|---|---|---|
| B-01 | Investment Advice Elicitation | SEBI IA Regulations 2013 §3 |
| B-02 | KYC/AML Bypass | RBI Master Directions KYC |
| B-03 | Regulatory Misrepresentation | SEBI FPI Regulations + RBI circulars |
| B-04 | Regulatory Document Hallucination | EU AI Act Art. 9(4) |
| B-05 | Consent & Data Rights Violations | DPDP Act 2023 |
| B-06 | Transaction Integrity Violations | RBI NACH/NEFT Frameworks |
| B-07 | Account/Document Authenticity Bypass | RBI Digital Banking Security circulars |

---

## Deployment Registers

FinProof covers three conversational registers reflecting real BFSI AI deployments:

- **Professional** — compliance officer framing, regulatory citations, mean 54 words
- **Customer Mobile** — colloquial chatbot queries, 8–30 words, no regulatory language
- **RM Internal** — relationship manager to internal AI, compressed professional shorthand

---

## Generation

FinProof v1 = ASSAY v1 (4,200 prompts, Claude/GPT-4o generated) + ASSAY-QI v2.0 (1,189 non-overlapping, quantum-augmented).

**ASSAY-QI v2.0 techniques:**
- SA Annealing (344) — QUBO minimum-perturbation on Semalith's decision boundary
- QCBM 8q boundary (173) — PennyLane quantum circuit Born machine, boundary-filtered seeds
- QCBM 10q Fix1 (123) — 10-qubit QCBM with paraphrase generation, **27.6% Semalith miss rate**
- QCBM 8q gradient (125) + joint (100)
- Retail customer mobile (157) + RM internal (105) — threat-vector-embedded retail register

---

## Data Governance

This repository follows a PINT-inspired four-tier release framework:

```
Tier 1 — Fully public:  Eval harness code (Apache 2.0), taxonomy, methodology
Tier 2 — Email gate:    Easy attacks (this repo) — direct difficulty
Tier 3 — Research agmt: Medium attacks — QCBM-generated, indirect difficulty
Tier 4 — Withheld:      Hard attacks — official test set, Zytra-only evaluation
```

**Tier 4 SHA-256 prefix** (for auditor verification): `bf35df2e5a3f08c9202555db1a5bd825...`

The withheld set inputs are never released. The published hash allows future auditors to verify the same test set has been used consistently across all evaluations.

---

## Schema

```json
{
  "id": "B-01-DIR-001",
  "category": "B-01",
  "difficulty": "direct",
  "register": "professional",
  "language": "en",
  "input": "...",
  "label": "attack",
  "regulatory_anchor": "SEBI IA Regulations 2013 §3",
  "source": "assay_v1",
  "benchmark": "finproof_v1",
  "finproof_tier": 2,
  "p_benign": null,
  "threat_vector_embedded": true
}
```

---

## Guardian Model Performance on FinProof

| Model | Size | Miss Rate | Domain Fit |
|---|---|---|---|
| **Semalith v1.5** | 184M | **14.3%** | ★★★★★ BFSI-specific |
| PromptGuard-86M | 86M | 8.8% (professional only) | ★☆☆☆☆ FPR=100% on retail |
| ShieldGemma-2B | 2B | 98.3% | ★☆☆☆☆ Wrong taxonomy |
| LlamaGuard-3-1B | 1B | 100% | ★☆☆☆☆ Too small |

Key finding: No existing open-source model except Semalith v1.5 correctly discriminates BFSI retail attacks from legitimate customer queries.

---

## Quick Start

```bash
# Install
pip install huggingface_hub

# Download Tier 2 data
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Zytra/finproof-bench",
    filename="finproof_v1_tier2_public.jsonl",
    repo_type="dataset"
)

# Run local evaluation
python eval/run_eval.py \
    --finproof-tier2 finproof_v1_tier2_public.jsonl \
    --model-module my_guardian \
    --output my_scores.json
```

---

## Citation

```bibtex
@dataset{zytra2025finproof,
  title     = {FinProof v1: A BFSI Adversarial Benchmark for AI Guardrail Systems},
  author    = {Zytra},
  year      = {2025},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/Zytra/finproof-bench}
}
```

---

## Contact

- 🌐 [zytratechnologies.com](http://zytratechnologies.com)
- 📧 finproof@zytra.ai — research agreement requests
- 🏢 India · BFSI-focused AI safety
