# FINPROOF — BFSI AI Safety Benchmark

> The first open adversarial benchmark for AI guardrail systems in Banking, Financial Services, and Insurance.

**11,474 prompts · 36 attack subtypes · 0% training contamination · Full regulatory mapping**

Published by [Zytra Tech Solutions](https://zytratechnologies.com) · BKC, Mumbai, India  
Benchmark portal: [finproof.ai](https://finproof.ai)  
Dataset: [huggingface.co/datasets/Zytra/finproof-bench](https://huggingface.co/datasets/Zytra/finproof-bench)  
Contact: finproof@zytratechnologies.com  
License: CC BY 4.0

---

## What FinProof evaluates (and what it does NOT)

FinProof is a **guardrail / safety** benchmark. It evaluates the control layer around a model, not the model's financial knowledge.

| It measures | It does NOT measure |
|---|---|
| Does the guardrail correctly flag BFSI attacks vs legitimate queries? | Whether the model knows finance (FinanceBench / FinQA) |
| Regulatory alignment (RBI / SEBI / IRDAI / DPDP) | General reasoning, math, chat quality |
| Full stack: input / processing / output layers | Model capability in isolation |
| FPR on a BFSI benign set (2,127 calibration queries) | — |

---

## Leaderboard — May 2026

### Track 1 — Guardrail Classification (B-series)

| Rank | Model | Organisation | BFSI F1 | Precision | Recall | FPR |
|------|-------|-------------|---------|-----------|--------|-----|
| 🥇 1 | **AVAL v1.4** | **Zytra Tech Solutions** | **0.968** | **0.940** | **0.998** | **—†** |
| 2 | Granite Guardian 3.3 | IBM Research | 0.813 | 0.979 | 0.695 | 10.2% |
| 3 | ShieldGemma 9B | Google | 0.731 | 0.993 | 0.578 | 2.9% |
| 4 | LlamaGuard 3 | Meta AI | 0.569 | 1.000 | 0.397 | 0.0% |
| 5 | WildGuard 7B | Allen AI | 0.346 | 0.999 | 0.209 | 0.1% |

> † AVAL v1.4 FPR from official eval server; see [finproof.ai](https://finproof.ai). Competitor FPR from independent benchmark run, same B-01–B-07 taxonomy, fp16, RTX 4090.

AVAL v1.4 leads on both F1 (0.968 vs next-best 0.813) and recall (0.998). Semalith v1.4 is the precision-optimised variant at 0.958 / 0.981 / 0.936 — a 4.1-point precision advantage over AVAL at the cost of 6.2-point lower recall.

AVAL v1.5 results (F1 0.977) are available at [finproof.ai](https://finproof.ai). These figures reflect the v1.4 release (IT4 evaluation run) for reproducibility against the published Excel data.

### Track 2 — Agentic Safety (D-series, LLM-as-Agent)

| Model | Organisation | Overall ASR ↓ | D-09 ASR | D-10 ASR | D-11 ASR |
|-------|-------------|--------------|----------|----------|----------|
| gemini-3-flash | Google | 40.3% | 33.3% | 35.8% | 51.7% |

> Lower ASR = better. ASR = Attack Success Rate on D-09/D-10/D-11 prompts sent directly to the model as agent.  
> AVAL Track 2 evaluation in progress — results pending.

### Per-domain F1 — B-01 to B-07

| Domain | AVAL v1.4‡ | Granite Guardian | ShieldGemma | LlamaGuard 3 | WildGuard 7B |
|--------|-----------|-----------------|-------------|--------------|-------------|
| B-01 | **0.954** ★ | 0.936 | 0.792 | 0.885 | 0.053 |
| B-02 | **0.981** ★ | 0.830 | 0.711 | 0.518 | 0.362 |
| B-03 | **0.960** ★ | 0.706 | 0.917 | 0.718 | 0.207 |
| B-04 | 0.957 | 0.756 | 0.957 | 0.315 | 0.252 |
| B-05 | **0.986** ★ | 0.746 | 0.534 | 0.496 | 0.448 |
| B-06 | **0.975** ★ | 0.792 | 0.572 | 0.429 | 0.421 |
| B-07 | **0.962** ★ | 0.895 | 0.481 | 0.465 | 0.556 |
| AVAL weighted (n=6,183) | **0.968** | — | — | — | — |

> ‡ AVAL v1.4 per-category F1 from IT4 run (n=6,183 attack rows, all 4 tiers). Competitor aggregate F1 in Track 1 leaderboard above.  
> B-04: AVAL and ShieldGemma tied at 0.957 — no ★ awarded.  
> Category codes map to taxonomy in the Domain Taxonomy section below.

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

**Track 2 — Inspect AI (LLM agent evaluation, requires cloning this repo):**

```bash
git clone https://github.com/Zytra-ai/finproof
pip install inspect-ai
cd finproof

# B-series only (guardrail classification, works with public HF subset)
inspect eval src/finproof_bench -T layer=input --model openai/gpt-4o

# D-series requires T3 access — D-categories not in public HF subset
# inspect eval src/finproof_bench -T layer=processing ...
```

No HuggingFace library required — all data available as plain JSONL for restricted BFSI environments.

---

## Domain Taxonomy

### B-series — Guardrail Classification (14 categories)

| Domain | Category | Regulatory Anchor |
|--------|----------|------------------|
| B-01 | Investment Advice Elicitation | SEBI IA Regulations 2013 §3 |
| B-02 | KYC / AML Bypass | RBI KYC Master Direction 2023 |
| B-03 | Regulatory Misrepresentation | SEBI FPI Regulations + RBI circulars |
| B-04 | Regulatory Document Hallucination | EU AI Act Art.9(4) |
| B-05 | Consent & Data Rights Violations | DPDP Act 2023 |
| B-06 | Transaction Integrity Violations | RBI NACH / NEFT frameworks |
| B-07 | Account / Document Authenticity Bypass | RBI Digital Banking Security circulars |
| B-08 | Financial Fraud / Extortion | SEC Rule 10b-5 · RBI cybercrime |
| B-09 | Unlicensed Financial Advice | SEC §202(a)(11) · MiFID II Art.24 |
| B-10 | Regulatory Compliance Enquiries | EU AI Act Art.52 · SEBI CIR |
| B-11 | AML / Sanctions Evasion | FATF Rec.10-16 · EU 6AMLD |
| B-12 | MNPI Aggregation & Market Manipulation | SEBI Insider Trading Reg. 2015 |
| B-13 | Cross-border Regulatory Arbitrage | FATF Rec.1 · BCBS Basel III |
| B-14 | Indic-language BFSI Attacks | RBI Consumer Protection Framework |

### D-series — Agentic Safety (3 categories · 15 attack subtypes)

| Domain | Category | Regulatory Anchor |
|--------|----------|------------------|
| D-09 | MCP Tool-call Poisoning | OWASP LLM Top 10 · EU AI Act Art.9 |
| D-10 | Multi-agent Orchestration Bypass | NIST AI RMF · EU AI Act Art.13 |
| D-11 | Goal-binding & Alignment Failures | EU AI Act Art.9(4) · IEEE 7010 |

Full taxonomy: [finproof/tiers/TAXONOMY.md](./finproof/tiers/TAXONOMY.md)

---

## Structure — three layers, mirroring the guardrail stack

- **Input Layer** — prompt injection, semantic injection, jailbreak attempts, PII submissions, sensitive-credential exposure, language-restriction violations. (B-series, public.)
- **Processing Layer** — agentic workflows: tool-scope-restriction sequences, rollback of unauthorized actions, inter-agent handoff / context-fidelity validation. (D-series, gated T3+.)
- **Output Layer** — hallucination probing against source documents, bias elicitation on protected demographic categories, source-citation verification. (Roadmap.)

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

---

## Generation Methodology

FINPROOF uses a hybrid **QCBM + Claude** pipeline:

1. 8-qubit PennyLane QCBM trained with MMD loss on DeBERTa embedding projections per domain
2. Born-rule sampling identifies underrepresented regions of each attack domain's distribution
3. Claude generates prompts conditioned on quantum-sampled feature vectors
4. SHA-1 + MinHash deduplication against 89,022 evaluated training hashes

Zero contamination with any public training dataset.

---

## Comparison to existing benchmarks

- **vs FinanceBench / FinQA / ConvFinQA** — those test financial knowledge; FinProof tests financial *safety / guardrails*. Complementary, not competing.
- **vs JailbreakBench / AILuminate / HarmBench / StrongREJECT** — general-domain safety; FinProof is the BFSI-specialised counterpart with banking attack patterns and RBI/SEBI/IRDAI-aligned action criteria.
- **vs general LLM benchmarks (HellaSwag, TruthfulQA, MT-Bench)** — those evaluate a general-purpose model; financial services is not a general-purpose environment.

FinProof occupies the empty quadrant: **BFSI-domain + adversarial-safety + Indian-regulatory + agentic-aware.**

---

## Submit Your Model

### Tier 2 / Tier 3 — local scoring (you have the prompts)

```bash
python finproof/eval/run_eval.py \
  --hf-dataset Zytra/finproof-bench \
  --finproof-version v1 \
  --submit-endpoint https://your-guardrail/predict \
  --model-name "MyGuard v1" --model-org "MyOrg"
```

### Tier 4 — blind server-side evaluation (prompts never leave Zytra's server)

T4 prompts are withheld permanently. Evaluation works as a blind pull:

1. **You expose** your guardrail as an HTTP endpoint: `POST /predict` → `{"prediction": 0|1}`
2. **Zytra's eval server** calls your endpoint once per T4 prompt
3. **You never see** any T4 prompt — only your own prediction responses
4. **Official scores** are returned and published at [finproof.ai](https://finproof.ai)

To register your endpoint: email finproof@zytratechnologies.com with model name, organisation, and endpoint URL.

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

## Independence

FinProof Bench is an independently developed and openly published standard. Per the author's published statement, there is no commercial affiliation between the benchmark's author and any model appearing in the leaderboard. Institutions using it in a formal model-admission context should still check for train/eval contamination for any system that may have been exposed to FinProof data.

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
