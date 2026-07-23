# FinProof Bench

> **BFSI adversarial safety benchmark** — the first dedicated evaluation framework for AI guardrails in Banking, Financial Services and Insurance.

- Dataset: [Zytra/finproof-bench](https://huggingface.co/datasets/Zytra/finproof-bench)
- Portal + leaderboard: [finproof.ai](https://finproof.ai)
- Paper: [SSRN 6728799](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6728799)
- Contact: finproof@zytratechnologies.com
- License: CC-BY-4.0

---

**FinProof Bench is the first adversarial benchmark dataset built specifically to evaluate the GenAI guardrail stack in banking, financial services, and insurance (BFSI).** Each record carries a **binary label — `attack` or `benign`** — plus `category`, `register`, `language`, and a `regulatory_anchor` tying every prompt to a specific obligation (SEBI, RBI, DPDP, EU AI Act). It scores whether a guardrail correctly distinguishes BFSI attacks from legitimate customer queries, reporting per-category F1, recall, and false-positive rate.

**Full dataset: 11,474 prompts** across all tiers (V1 + V2). **HuggingFace public subset: 2,388 rows** — `attacks` (1,606) + `benign` (782) — across **7 BFSI attack categories (B-01–B-07)**, professional and retail registers. The wider v2 corpus — additional categories, agentic/MCP D-series, Indic languages, and the withheld Tier-4 test set — is gated separately; see access tiers below.

## What FinProof evaluates (and what it does NOT)

FinProof is a **guardrail / safety** benchmark. It evaluates the control layer around a model, not the model's knowledge.

| It measures | It does NOT measure |
|---|---|
| Does the guardrail correctly flag BFSI attacks vs benign customer queries (binary)? | Whether the model *knows* finance (that's FinanceBench / FinQA) |
| Regulatory alignment (RBI / SEBI / IRDAI / DPDP) + a safety score | General reasoning, math, chat quality |
| Full stack across Input / Processing / Output layers | Model capability in isolation |
| False-positive rate on a BFSI benign set (2,127 legitimate queries) | — |

## Structure — three layers, mirroring the guardrail stack

Distributed as **plain JSONL** by design, so institutions with restricted outbound access can run it entirely inside their own perimeter (no external API calls or package dependencies).

- **Input Layer** — prompt injection, semantic injection, jailbreak attempts, PII submissions, sensitive-credential exposure, language-restriction violations. Each case carries a binary `label` (`attack`/`benign`), a `category`, and a `regulatory_anchor`. (The Block/Redact/Warn/Clarify actions and the 21-action set are the framework's **conceptual** guardrail taxonomy — how a stack *should* respond — not fields in the dataset.)
- **Processing Layer** — agentic workflows: tool-scope-restriction sequences, rollback of unauthorized actions, inter-agent handoff / context-fidelity validation.
- **Output Layer** — hallucination probing against known source documents, bias elicitation on protected demographic categories, source-citation verification, watermark/provenance validation, and Explainability-Trace verification (EU AI Act Art. 13; SR 11-7 documentation).

## Attack categories (public tier — B-01–B-07)

| ID | Category | Regulatory anchor |
|---|---|---|
| B-01 | Investment Advice Elicitation | SEBI IA Regulations 2013 §3 |
| B-02 | KYC / AML Bypass | RBI Master Directions KYC |
| B-03 | Regulatory Misrepresentation | SEBI FPI Regulations + RBI circulars |
| B-04 | Regulatory Document Hallucination | EU AI Act Art. 9(4) |
| B-05 | Consent & Data Rights Violations | DPDP Act 2023 |
| B-06 | Transaction Integrity Violations | RBI NACH / NEFT frameworks |
| B-07 | Account / Document Authenticity Bypass | RBI Digital Banking Security circulars |

Prompts span **professional** and **customer-mobile** conversational registers; adversarial coverage augmented via QCBM generation with a contamination-audit tool (`contamination_audit.py`). Wider categories (B-08–B-14, agentic/MCP D-09–D-11) and Indic languages exist in the gated/full corpus, not the public tier.

## Key results — FinProof (ASSAY v1), per category

Eval-grounded results on the FinProof BFSI attack set (B-01–B-07). Positive class = attack; F1 / Precision / Recall are standard binary metrics on the attack categories. (The benign `general` split is excluded here — benign-class F1 is undefined; over-refusal is reported separately as pass-through accuracy.)

| Category | n† | AVAL v1.4 F1 / Pr / Rc | Semalith v1.4 F1 / Pr / Rc |
|---|---|---|---|
| B-01 Investment Advice Elicitation | 867 | 0.943 / 0.899 / 0.991 | 0.956 / 0.989 / 0.924 |
| B-02 KYC / AML Bypass | 877 | 0.952 / 0.917 / 0.990 | 0.940 / 0.974 / 0.907 |
| B-03 Regulatory Misrepresentation | 863 | 0.943 / 0.894 / 0.997 | 0.982 / 0.987 / 0.976 |
| B-04 Regulatory Document Hallucination | 874 | 0.938 / 0.895 / 0.986 | 0.980 / 0.990 / 0.970 |
| B-05 Consent & Data Rights Violations | 918 | 0.960 / 0.927 / 0.995 | 0.978 / 0.977 / 0.978 |
| B-06 Transaction Integrity Violations | 886 | 0.947 / 0.906 / 0.991 | 0.936 / 0.982 / 0.894 |
| B-07 Account / Document Authenticity Bypass | 898 | 0.947 / 0.904 / 0.994 | 0.936 / 0.968 / 0.905 |
| **Weighted overall (n=6,183)** | | **0.947 / 0.906 / 0.992** | **0.958 / 0.981 / 0.936** |

† n = full evaluation set across all 4 tiers (6,183 attack rows, including withheld T4). HuggingFace public subset has 222–239 rows per category.

**Reading it:** the two models sit at opposite ends of the precision–recall trade-off. **AVAL v1.4 is the recall leader** (0.992 — catches nearly every attack) at lower precision (0.906). **Semalith v1.4 is the precision leader** (0.981 — ~1 false alarm in 52 flags vs ~1 in 11 for AVAL) at slightly lower recall (0.936). On a BFSI benign stream, precision is the operationally costly axis (false positives block real customers), so the ~7.5-point precision gap is the key differentiator; where maximal catch-rate matters more than false-alarm cost, AVAL's recall leads. Metric definitions must travel with these numbers — a bare F1 is misread without the attack=positive convention.

## Usage

```python
from datasets import load_dataset

# Two subsets: "attacks" and "benign". Prompt text is in the `input` field.
attacks = load_dataset("Zytra/finproof-bench", "attacks", split="train")  # 1,606
benign  = load_dataset("Zytra/finproof-bench", "benign",  split="train")  # 782 (FPR calibration)

# schema: id, category, difficulty, register, language, input, label, regulatory_anchor,
#         source, benchmark, finproof_tier
```

**Run the evaluation harness locally:**

```bash
python finproof/eval/run_eval.py \
  --hf-dataset Zytra/finproof-bench \
  --finproof-version v1 \
  --submit-endpoint https://your-guardrail/predict \
  --model-name "MyGuard v1" --model-org "MyOrg"
```

## How institutions should use it

Run at three lifecycle points (per the FinProof guidance): before production deployment (SR 11-7 validation gate), after any material guardrail-config change, and on a quarterly/semi-annual monitoring cadence. Results should be reviewed by 2LoD (risk & compliance) and internal audit — not only the team that built the controls.

## Distinctive design (not in general benchmarks)

- **Two conversational registers** — Professional / Customer-Mobile. Exposes register-dependent FPR (e.g. a guardrail with ~0% FPR on professional queries but high FPR on the retail register).
- **FPR-calibrated BFSI benign set** (2,127 legitimate banking queries) — general benchmarks have no BFSI benign set, so their FPR is meaningless.
- **Four-tier access** — T1 public JSONL · T2 email-gate · T3 research agreement · T4 withheld test set (submit guardrail as HTTP endpoint; `POST {"input"} -> {"prediction":0|1}`; prompts never leave the eval server).

## Comparison to existing benchmarks

- **vs FinanceBench / FinQA / ConvFinQA** — those test financial knowledge; FinProof tests financial *safety / guardrails*. Complementary.
- **vs JailbreakBench / AILuminate / HarmBench / StrongREJECT** — general-domain safety; FinProof is the BFSI-specialised counterpart with banking attack patterns and RBI/SEBI/IRDAI-aligned action criteria.
- **vs general LLM benchmarks (HellaSwag, TruthfulQA, MT-Bench)** — those evaluate a general-purpose model; financial services is not a general-purpose environment.

FinProof occupies the empty quadrant: **BFSI-domain + adversarial-safety + Indian-regulatory + agentic-aware.**

## Access tiers

| Tier | Prompts | Content | Access |
|---|---|---|---|
| T1 | 2,127 | Benign — FPR calibration baseline | Public · HuggingFace `benign` config |
| T2 | 2,857 | Direct attacks + easy agentic scenarios | Email registration |
| T3 | 3,555 | Medium-difficulty · QCBM-generated · Indic | Research agreement |
| T4 | 2,935 | Hard-difficulty · Official withheld test set | Server-side only — prompts never revealed |
| **Total** | **11,474** | V1 + V2 | |

T3 access: finproof@zytratechnologies.com

## Citation

```bibtex
@article{addagada2026finproof,
  title   = {FINPROOF: The First Adversarial Benchmark for AI Guardrail
             Systems in Banking, Financial Services, and Insurance},
  author  = {Zytra Techsolutions},
  journal = {SSRN Working Paper},
  number  = {6728799},
  year    = {2026},
  url     = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6728799}
}
```

## Independence

FinProof Bench is an independently developed and openly published standard. Per the author's published statement, there is no commercial affiliation between the benchmark's author and any model appearing in the leaderboard. Institutions using it in a formal model-admission context should still check for train/eval contamination for any system that may have been exposed to FinProof data.

## Links

- Dataset: https://huggingface.co/datasets/Zytra/finproof-bench
- Harness (CLI): https://github.com/zytra-ai/finproof
- Portal + leaderboard: https://finproof.ai
- Paper (SSRN): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6728799
- Contact: finproof@zytratechnologies.com

## Inspect AI Evaluation (Track 2 — LLM-agent scoring)

Run FinProof against an LLM agent using [Inspect AI](https://inspect.ai-safety-institute.org.uk/):

```bash
# Install
pip install inspect-ai finproof-bench

# Full benchmark
inspect eval finproof_bench --model openai/gpt-4o

# B-series only (guardrail classification, input layer)
inspect eval finproof_bench -T layer=input --model openai/gpt-4o

# D-series only (agentic attacks, with LLM judge)
inspect eval finproof_bench -T layer=processing -T judge_model=ollama/phi3 --model openai/gpt-4o

# Smoke test
inspect eval finproof_bench -T limit=5 --model openai/gpt-4o
```

Source: [`src/finproof_bench/`](src/finproof_bench/)

