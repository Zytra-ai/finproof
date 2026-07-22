# FINPROOF — BFSI AI Safety Benchmark

> The first open adversarial benchmark for AI guardrail systems in Banking, Financial Services, and Insurance.

**5,389 prompts · 22 BFSI domains · 0% training contamination · Full regulatory mapping**

Published by [Zytra Tech Solutions](https://zytratechnologies.com) · BKC, Mumbai, India  
Benchmark portal: [finproof.ai](https://finproof.ai)  
Dataset: [huggingface.co/datasets/Zytra/finproof-bench](https://huggingface.co/datasets/Zytra/finproof-bench)  
Contact: finproof@zytratechnologies.com  
License: CC BY 4.0

---

## Leaderboard — May 2026

| Rank | Model | Organisation | BFSI F1 | Precision | Recall | FPR |
|------|-------|-------------|---------|-----------|--------|-----|
| 🥇 1 | **AVAL v1.5** | **Zytra Tech Solutions** | **0.977** | **0.966** | **0.989** | **2.4%** |
| 2 | Granite Guardian 3.3 | IBM Research | 0.813 | 0.979 | 0.695 | 10.2% |
| 3 | ShieldGemma 9B | Google | 0.731 | 0.993 | 0.578 | 2.9% |
| 4 | LlamaGuard 3 | Meta AI | 0.569 | 1.000 | 0.397 | 0.0% |
| 5 | WildGuard 7B | Allen AI | 0.346 | 0.999 | 0.209 | 0.1% |

### Per-domain F1 — B-01 to B-07

| Domain | Semalith v1.4 | Granite Guardian | ShieldGemma | LlamaGuard 3 | WildGuard 7B |
|--------|--------------|-----------------|-------------|--------------|-------------|
| B-01 · Investment advice | **0.985** ★ | 0.936 | 0.792 | 0.885 | 0.053 |
| B-02 · KYC / Card fraud | **0.971** ★ | 0.830 | 0.711 | 0.518 | 0.362 |
| B-03 · Employment fraud | **0.984** ★ | 0.706 | 0.917 | 0.718 | 0.207 |
| B-04 · Regulatory hallucination | **0.926** ★ | 0.756 | **0.957** | 0.315 | 0.252 |
| B-05 · Predatory lending | **1.000** ★ | 0.746 | 0.534 | 0.496 | 0.448 |
| B-06 · Insurance conduct | **0.976** ★ | 0.792 | 0.572 | 0.429 | 0.421 |
| B-07 · Financial instruments | **1.000** ★ | 0.895 | 0.481 | 0.465 | 0.556 |

> Semalith metrics: macro F1 from 3-seed verified validation split (seeds 42/123/456, std=0.0012).  
> Competitor metrics: independent benchmark run, same B-01..B-07 taxonomy, fp16, RTX 4090.  
> Official withheld-set scores at [finproof.ai](https://finproof.ai)

---

## Quick Start

```bash
pip install datasets
```

```python
from datasets import load_dataset

# Tier 1 — 1,078 benign examples (public, no registration)
benign = load_dataset("Zytra/finproof-bench", "benign", split="train")

# Tier 2 — 1,347 attack prompts (email registration)
attacks = load_dataset("Zytra/finproof-bench", "attacks", split="train")
```

**Run the evaluation harness locally:**

```bash
python finproof/eval/run_eval.py \
  --model your-model-name \
  --output scores.json

python finproof/eval/submit.py \
  --submission scores.json \
  --email you@yourorg.com
```

No HuggingFace library required — all data available as plain JSONL for restricted BFSI environments.

---

## Domain Taxonomy

| Domain | Category | Regulatory Anchor |
|--------|----------|------------------|
| B-01 | Investment advice elicitation | SEBI IA Regulations 2013 §3 |
| B-02 | KYC / Card fraud | RBI KYC Master Direction 2023 |
| B-03 | Employment / recruitment fraud | FTC Act §5 · UK Fraud Act 2006 |
| B-04 | Regulatory document hallucination | EU AI Act Art.9(4) |
| B-05 | Predatory lending | EU MCD Art.18 · RBI FSLRC |
| B-06 | Insurance conduct violations | FCA ICOBS · EU Solvency II |
| B-07 | Financial instruments misuse | MiFID II · SEBI Regulations |
| B-08 | Financial fraud / extortion | SEC Rule 10b-5 · RBI cybercrime |
| B-09 | Unlicensed financial advice | SEC §202(a)(11) · MiFID II Art.24 |
| B-10 | Regulatory compliance enquiries | EU AI Act Art.52 · SEBI CIR |
| B-11 | AML / Sanctions evasion | FATF Rec.10-16 · EU 6AMLD |
| D-01 | Direct prompt injection | OWASP LLM Top 10 2025 |
| D-06 | Agentic pipeline injection | AgentHarm ICLR 2025 |
| D-07 | Indirect / RAG injection | BIPIA · INJECAGENT |

Full taxonomy: [finproof/tiers/TAXONOMY.md](./finproof/tiers/TAXONOMY.md)

---

## Data Architecture

| Tier | Prompts | Content | Access |
|------|---------|---------|--------|
| Tier 1 | 1,078 | Benign — FPR calibration baseline | Public · No registration |
| Tier 2 | 1,347 | Easy-difficulty attacks | Email registration |
| Tier 3 | 1,886 | Medium-difficulty · QCBM-generated | Research agreement required |
| Tier 4 | 1,078 | Hard-difficulty · Official test set | **Withheld permanently** |

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
python finproof/eval/run_eval.py --model your-model --output submission.json
python finproof/eval/submit.py --submission submission.json --email you@yourorg.com
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
- Research page: [zytratechnologies.com/research/finproof](https://zytratechnologies.com/research/finproof)
- Semalith model: [huggingface.co/zytra-ai/semalith-bfsi-v4](https://huggingface.co/zytra-ai/semalith-bfsi-v4)
- SSRN paper: [papers.ssrn.com/sol3/papers.cfm?abstract_id=6728799](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6728799)
- Contact: finproof@zytratechnologies.com

---

© 2026 Zytra Tech Solutions · DIPP199187 · BKC, Mumbai, Maharashtra 400051, India  
FINPROOF™ Trademark Pending · Patent Pending
