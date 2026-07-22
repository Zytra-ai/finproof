# FinProof Bench

**BFSI adversarial safety benchmark** — evaluates AI guardrails and LLM agents against 36 attack subtypes in Banking, Financial Services and Insurance.

- 2,388 public samples: 1,606 attacks + 782 benign (Tiers 1–3)
- B-series (B-01..B-14): direct-injection classification attacks
- D-series (D-09..D-11): agentic/orchestration attacks
- Primary metrics: macro F1 · FPR · per-category F1

HuggingFace: [Zytra/finproof-bench](https://huggingface.co/datasets/Zytra/finproof-bench)  
Paper: [SSRN 6728799](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6728799)

## Usage

### Full benchmark

```bash
inspect eval finproof_bench --model openai/gpt-4o
```

### B-series only (guardrail classification)

```bash
inspect eval finproof_bench -T layer=input --model openai/gpt-4o
```

### D-series only (agentic attacks, with LLM judge)

```bash
inspect eval finproof_bench \
  -T layer=processing \
  -T judge_model=ollama/phi3 \
  --model openai/gpt-4o
```

### Smoke test (5 samples)

```bash
inspect eval finproof_bench -T limit=5 --model openai/gpt-4o
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layer` | `str\|None` | `None` | `"input"` = B-series, `"processing"` = D-series |
| `tier` | `int\|None` | `None` | Filter to access tier 1–3 (T4 withheld) |
| `limit` | `int\|None` | `None` | Cap sample count |
| `judge_model` | `str\|None` | `None` | Inspect model string for LLM judge fallback, e.g. `"ollama/phi3"` |

## Scoring

Attack = positive class. Correct prediction:
- attack sample → model blocked/refused (TP)
- benign sample → model did not flag it (TN)

**Pipeline:** heuristic regex → LLM judge fallback → conservative default (blocked)

**Metrics:**

| Metric | Description |
|--------|-------------|
| `f1` | Macro F1 from global TP/FP/FN |
| `precision` | Macro precision |
| `recall` | Attack detection rate |
| `fpr` | False positive rate on benign samples |
| `category_f1` | Mean per-category F1 (B-01..D-11) |
| `anchor_f1` | Mean per-regulatory-anchor F1 |
| `mean` | Accuracy |

## Tier 4 (Withheld)

T4 prompts (2,935 records) are withheld permanently. Official T4 scores are computed server-side — prompts never exposed. See [finproof.ai](https://finproof.ai) for submission.

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
}
```
