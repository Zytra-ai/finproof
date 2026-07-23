# FINPROOF — BFSI AI Safety Benchmark

> The first open adversarial benchmark for AI guardrail systems in Banking, Financial Services, and Insurance.

**11,474 prompts · 36 attack subtypes · 0% training contamination · Full regulatory mapping**

Published by [Zytra Tech Solutions](https://zytratechnologies.com) · BKC, Mumbai, India  
Benchmark portal: [finproof.ai](https://finproof.ai)  
Dataset: [huggingface.co/datasets/Zytra/finproof-bench](https://huggingface.co/datasets/Zytra/finproof-bench)  
Contact: finproof@zytratechnologies.com  
License: CC BY 4.0

---

## Leaderboard — May 2026

### Track 1 — Guardrail Classification (B-series)

| Rank | Model | Organisation | BFSI F1 | Precision | Recall | FPR |
|------|-------|-------------|---------|-----------|--------|-----|
| 🥇 1 | **AVAL v1.4** | **Zytra Tech Solutions** | **0.968** | **0.940** | **0.998** | **—** |
| 2 | Granite Guardian 3.3 | IBM Research | 0.813 | 0.979 | 0.695 | 10.2% |
| 3 | ShieldGemma 9B | Google | 0.731 | 0.993 | 0.578 | 2.9% |
| 4 | LlamaGuard 3 | Meta AI | 0.569 | 1.000 | 0.397 | 0.0% |
| 5 | WildGuard 7B | Allen AI | 0.346 | 0.999 | 0.209 | 0.1% |

**AVAL v1.4 leads on both F1 (0.968 vs next-best 0.813) and recall (0.998).** Semalith v1.4 is the precision-optimised variant at 0.958 / 0.981 / 0.936 — a 4.1-point precision advantage over AVAL at the cost of 6.2-point recall. On a BFSI benign stream the precision gap is the operationally costly axis (false positives block real customers), so the trade-off is model-selection context dependent.

> AVAL v1.4: macro F1 / Precision / Recall from full 4-tier evaluation (n=6,183 attack rows, IT4 run). FPR available at [finproof.ai](https://finproof.ai).  
> Competitor metrics: independent benchmark run, same B-01..B-07 taxonomy, fp16, RTX 4090.

### Track 2 — Agentic Safety (D-series, LLM-as-Agent)

| Rank | Model | Organisation | Overall ASR ↓ | D-09 ASR | D-10 ASR | D-11 ASR |
|------|-------|-------------|--------------|----------|----------|----------|
| 🥇 1 | **AVAL v1.4** | **Zytra Tech Solutions** | **Eval in progress** | — | — | — |
| — | gemini-3-flash | Google | 40.3% | 33.3% | 35.8% | 51.7% |

> Lower ASR = better. ASR = Attack Success Rate on D-09/D-10/D-11 prompts sent directly to the model as agent.

### Per-domain F1 — B-01 to B-07

| Domain | AVAL v1.4 | Granite Guardian | ShieldGemma | LlamaGuard 3 | WildGuard 7B |
|--------|-----------|-----------------|-------------|--------------|-------------|
| B-01 · Investment advice | **0.954** ★ | 0.936 | 0.792 | 0.885 | 0.053 |
| B-02 · KYC / AML bypass | **0.981** ★ | 0.830 | 0.711 | 0.518 | 0.362 |
| B-03 · Regulatory misrepresentation | **0.960** ★ | 0.706 | 0.917 | 0.718 | 0.207 |
| B-04 · Regulatory hallucination | **0.957** ★ | 0.756 | 0.957 | 0.315 | 0.252 |
| B-05 · Consent & data rights | **0.986** ★ | 0.746 | 0.534 | 0.496 | 0.448 |
| B-06 · Transaction integrity | **0.975** ★ | 0.792 | 0.572 | 0.429 | 0.421 |
| B-07 · Account authenticity bypass | **0.962** ★ | 0.895 | 0.481 | 0.465 | 0.556 |
| **Weighted (n=6,183)** | **0.968** | 0.813 | 0.731 | 0.569 | 0.346 |

---

## Quick Start

```bash
pip install datasets
```

```python
from datasets import load_dataset

# Public subset — 782 benign + 1,606 attack prompts (T1+T2, no registration)
benign  = load_dataset("Zytra/finproof-bench", "benign",  split="train")  # 782
attacks = load_dataset("Zytra/finproof-bench", "attacks", split="train")  # 1,606
```

**Run the evaluation harness locally (Track 1 — guardrail endpoint):**

```bash
python finproof/eval/run_eval.py \
  --hf-dataset Zytra/finproof-bench \
  --finproof-version v1 \
  --submit-endpoint https://your-guardrail/predict \
  --model-name "MyGuard v1" --model-org "MyOrg"
```

**Track 2 — Inspect AI (LLM agent evaluation):**

```bash
pip install inspect-ai finproof-bench

# Full public benchmark (B-series + D-series)
inspect eval finproof_bench --model openai/gpt-4o

# B-series only (guardrail classification)
inspect eval finproof_bench -T layer=input --model openai/gpt-4o

# D-series requires T3 access — public subset has no D-categories
# inspect eval finproof_bench -T layer=processing ...
```

Inspect source: [`src/finproof_bench/`](src/finproof_bench/)

No HuggingFace library required — all data available as plain JSONL for restricted BFSI environments.

---

## Domain Taxonomy

### B-series — Guardrail Classification (14 categories)

| Domain | Category | Regulatory Anchor |
|--------|----------|------------------|
| B-01 | Investment advice elicitation | SEBI IA Regulations 2013 §3 |
| B-02 | KYC / AML bypass | RBI KYC Master Direction 2023 |
| B-03 | Regulatory misrepresentation | SEBI FPI Regulations + RBI circulars |
| B-04 | Regulatory document hallucination | EU AI Act Art.9(4) |
| B-05 | Consent & data rights violations | DPDP Act 2023 |
| B-06 | Transaction integrity violations | RBI NACH / NEFT frameworks |
| B-07 | Account / document authenticity bypass | RBI Digital Banking Security circulars |
| B-08 | Financial fraud / extortion | SEC Rule 10b-5 · RBI cybercrime |
| B-09 | Unlicensed financial advice | SEC §202(a)(11) · MiFID II Art.24 |
| B-10 | Regulatory compliance enquiries | EU AI Act Art.52 · SEBI CIR |
| B-11 | AML / Sanctions evasion | FATF Rec.10-16 · EU 6AMLD |
| B-12 | MNPI aggregation & market manipulation | SEBI Insider Trading Reg. 2015 |
| B-13 | Cross-border regulatory arbitrage | FATF Rec.1 · BCBS Basel III |
| B-14 | Indic-language BFSI attacks | RBI Consumer Protection Framework |

### D-series — Agentic Safety (3 categories · 15 attack subtypes)

| Domain | Category | Regulatory Anchor |
|--------|----------|------------------|
| D-09 | MCP tool-call poisoning | OWASP LLM Top 10 · EU AI Act Art.9 |
| D-10 | Multi-agent orchestration bypass | NIST AI RMF · EU AI Act Art.13 |
| D-11 | Goal-binding & alignment failures | EU AI Act Art.9(4) · IEEE 7010 |

Full taxonomy: [finproof/tiers/TAXONOMY.md](./finproof/tiers/TAXONOMY.md)

---

## Data Architecture

| Tier | Prompts | Content | Access |
|------|---------|---------|--------|
| Tier 1 | 2,127 | Benign — FPR calibration baseline (782 public on HF) | Public · No registration |
| Tier 2 | 2,857 | Direct attacks + easy agentic scenarios (1,606 public on HF) | Email registration |
| Tier 3 | 3,555 | Medium-difficulty · QCBM-generated · Indic | Research agreement required |
| Tier 4 | 2,935 | Hard-difficulty · Official test set | **Withheld permanently** |

**Total: 11,474 records** (9,347 attack · 2,127 benign)

V1/V2 breakdown:

| Version | Records | Notes |
|---------|---------|-------|
| V1 English | 6,283 | B-01 to B-11, T1–T4 |
| V1 Indic | 3,900 | Hindi/Telugu/Tamil BFSI attacks |
| V2 B-series | 625 | B-12/B-13/B-14 with 36 subtypes |
| V2 B Indic | 216 | Indic extension of B-series v2 |
| V2 D-series | 450 | D-09/D-10/D-11 agentic attacks |
| **Total** | **11,474** | |

Tier 3 access: finproof@zytratechnologies.com  
Withheld set SHA-256: `bf35df2e5a3f08c9202555db1a5bd825...`

---

## Generation Methodology

FINPROOF uses a hybrid **QCBM + Claude** pipeline:

1. 8-qubit PennyLane QCBM trained with MMD loss on DeBERTa embedding projections per domain
2. Born-rule sampling identifies underrepresented regions of each attack domain's distribution
3. Claude generates prompts conditioned on quantum-sampled feature vectors
4. SHA-1 + MinHash deduplication against 89,022 evaluated training hashes

Zero contamination with any public training dataset.

---

## Submit Your Model

```bash
# Local T2/T3 scoring
python finproof/eval/run_eval.py \
  --hf-dataset Zytra/finproof-bench \
  --finproof-version v1 \
  --submit-endpoint https://your-guardrail/predict \
  --model-name "MyGuard v1" --model-org "MyOrg"

# T4 server-side evaluation (prompts never revealed)
python finproof/eval/run_eval.py \
  --submit-endpoint https://your-guardrail/predict \
  --model-name "MyGuard v1" --model-org "MyOrg"
```

Official scores computed on withheld Tier 4 set.  
Results published at [finproof.ai](https://finproof.ai)  
Evaluation is free for all submissions.

---

## Citation

```bibtex
@article{addagada2026finproof,
  title   = {FINPROOF: The First Adversarial Benchmark for AI Guardrail
             Systems in Banking, Financial Services, and Insurance},
  author  = {Zytra Techsolutions},
  journal = {SSRN Working Paper},
  number  = {6728799},
  year    = {2026},
  url     = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6728799},
  note    = {FINPROOF™ Trademark Pending · DIPP199187}
}
```

---

## Links

- Benchmark portal: [finproof.ai](https://finproof.ai)
- Dataset: [huggingface.co/datasets/Zytra/finproof-bench](https://huggingface.co/datasets/Zytra/finproof-bench)
- Inspect eval: [github.com/Zytra-ai/finproof/tree/main/src/finproof_bench](https://github.com/Zytra-ai/finproof/tree/main/src/finproof_bench)
- SSRN paper: [papers.ssrn.com/sol3/papers.cfm?abstract_id=6728799](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6728799)
- Contact: finproof@zytratechnologies.com

---

© 2026 Zytra Tech Solutions · DIPP199187 · BKC, Mumbai, Maharashtra 400051, India  
FINPROOF™ Trademark Pending · Patent Pending
