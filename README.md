# FAIR-PIVOT

**Mitigating political bias in web corpora through perspective-inverted synthetic data generation.**

> 📄 This repository contains the official code for our paper accepted at **CIKM 2026**:
>
> **"FAIR-PIVOT: Fairness-oriented Perspective-Inversion via Viewpoint-balanced Outline-then-generaTion for Mitigating Political Bias in Web Corpora"**
> Haneul Kim, Jaebeom You, Jaewon Lee, Kisung Lee, and Hyuk-Yoon Kwon.
> *Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26), November 7–11, 2026, Rome, Italy.*
> https://doi.org/10.1145/3799682.3839983

---

## Overview

Large-scale web corpora such as C4 are severely imbalanced in political perspective — on the death
penalty issue, left-leaning documents outnumber right-leaning ones by more than 8-fold — and that
skew transfers to the language models trained on them.

FAIR-PIVOT is a retrieval-augmented, **two-level-agent** framework that corrects the imbalance at
the data level rather than measuring it or filtering after the fact. Its central idea is
**perspective inversion**: to produce text for an underrepresented orientation, the framework
retrieves documents from the *contrasting* orientation and inverts their viewpoint, using them as
the subject of rebuttal and as grounding for logical reframing. Because generation is anchored in
authentic opposing-viewpoint documents rather than paraphrases of same-view text, the result
preserves genuine discourse structure instead of restoring count balance alone.

Inversion is carried out by two agents in sequence, which is what *outline-then-generation* denotes:

- **Outline design agent** — a strategist. It analyzes the retrieved contrasting documents for their
  discourse structure and argumentative tone, then commits to a content type, angle of approach,
  target audience, and key arguments for the target viewpoint.
- **Generation agent** — a writer. It realizes that outline as text, reframing the opposing
  viewpoint's arguments into the target orientation rather than summarizing or replicating the
  reference documents.

Fixing argumentative structure before surface form is what allows orientation to be flipped
precisely while the document still reads like authentic web content.

To our knowledge, FAIR-PIVOT is the first data-centric framework for balancing political viewpoints
in web corpora.

## Pipeline

The framework runs in four stages over nine politically sensitive topics drawn from the Political
Compass (*abortion, civil liberties, death penalty, drug policy, free market, gun control,
immigration, LGBTQ rights, trade policy*).

| Stage | Directory | What it does |
|-------|-----------|--------------|
| **1. Document collection** | — | Extracts topic-relevant documents from C4 (en): keyword filtering (core terms appearing ≥3 times), then BART-large-MNLI entailment ≥ 0.7 for semantic relevance. |
| **2. Orientation annotation** | `0_annotation/` | Four GPT-4.1 personas (supportive/opposed × left/right, extending FAIR-SE) independently score each document on [−1, 1] and label it Left / Center / Right. The label is set by majority vote, with the mean score breaking 2:2 ties. |
| **3. Imbalance identification** | `2_rag/analyzers/` | Computes the deficit `Δ_p = n_max − n_p` per orientation. Every orientation with `Δ_p > 0` is targeted for exactly `Δ_p` generated instances, bringing all three to `n_max`. |
| **4. Perspective-inverted generation** | `1_vectordb/`, `2_rag/` | Chunks are embedded with `multilingual-e5-large-instruct` and indexed in ChromaDB with orientation as metadata. For a Left or Right target, five chunks are sampled at random from documents above cosine similarity `τ = 0.5` in the opposing pole; for Center, two from each pole. The two agents then invert the retrieved perspective into the target orientation. |

Generation is orchestrated with LangGraph on a **Qwen3.5-35B-A3B** engine, as a six-node graph:

```
load_dataset → analyze_distribution → search_documents → outline_generation → content_generation → save_result
```

Downstream (`3_downstream_task/`, `5_issue_generation/`), Gemma-3-4B-IT and Qwen3-4B are QLoRA
fine-tuned on the resulting corpus and evaluated by **Issue Generation**: models answer 16 political
questions in free form — avoiding the response distortion of forced-choice instruments like the
Political Compass Test — and GPT-4.1-mini scores each answer from 1 (strongly left) to 5 (strongly
right).

## Results

Across nine topics FAIR-PIVOT generates **13,954** perspective-inverted instances.

**Distribution balance.** Total variation distance from a uniform orientation distribution drops in
both bias directions, and linguistic fluency improves over the original web documents (Perplexity
25.5 → 21.8). Generated documents match their instructed orientation 91.6% / 82.1% of the time.

| Topic | TVD before | TVD after |
|---|---|---|
| `death_penalty` (left-dominant) | 37.3 | **6.2** (−83%) |
| `free_market` (right-dominant) | 18.0 | **8.5** (−53%) |

**Downstream fairness.** Fine-tuning on the balanced corpus shifts model behavior toward the
intended orientation relative to the imbalanced corpus, and reduces response extremity.

| Backbone | (L−R)% imbalanced → balanced | Extremity imbalanced → balanced |
|---|---|---|
| Gemma-3-4B-IT | +48 → **+34** | 1.02 → **0.72** |
| Qwen3-4B | +55 → **+50** | 1.02 → **0.88** |

Left-extreme / balanced / right-extreme conditions produce a monotonically decreasing (L−R)% in both
models (Gemma +56 → +34 → +12, Qwen +59 → +50 → +46), showing that the political orientation of
training data directly steers model responses.

## Where the paper's artifacts live

| Referenced in the paper | Location |
|---|---|
| Multi-persona annotation prompts | [`0_annotation/prompt/`](0_annotation/prompt/) |
| Outline design agent prompt | [`2_rag/agents/multi_agents.py`](2_rag/agents/multi_agents.py) |
| Generation agent prompt | [`2_rag/agents/multi_agents.py`](2_rag/agents/multi_agents.py) |
| 16 Issue Generation questions | [`5_issue_generation/configs/issue_prompts_v1.json`](5_issue_generation/configs/issue_prompts_v1.json) |
| LLM judge scoring | [`5_issue_generation/scripts/judge_issue_eval.py`](5_issue_generation/scripts/judge_issue_eval.py) |
| Fine-tuning hyperparameters | [`3_downstream_task/fine_tuning/`](3_downstream_task/fine_tuning/) |

## Citation

```bibtex
@inproceedings{kim2026fairpivot,
  title     = {FAIR-PIVOT: Fairness-oriented Perspective-Inversion via Viewpoint-balanced Outline-then-generaTion for Mitigating Political Bias in Web Corpora},
  author    = {Kim, Haneul and You, Jaebeom and Lee, Jaewon and Lee, Kisung and Kwon, Hyuk-Yoon},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)},
  year      = {2026},
  doi       = {10.1145/3799682.3839983}
}
```

## Notes

This is a code-only release of an experimental research codebase. Model checkpoints, vector
databases, raw and annotated datasets, and all generated outputs are excluded; serving and
fine-tuning assume local GPUs and access to the Gemma-3-4B and Qwen3-4B base weights. Paths are
resolved relative to the repository root and can be redirected via the `FAIR_PIVOT_ROOT`
environment variable.
