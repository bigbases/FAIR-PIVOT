# FAIR-PIVOT

**Fairness-Aware synthetic data generation and downstream evaluation for mitigating political bias in language models.**

> 📄 This repository contains the official code for our paper accepted at **CIKM 2026**:
>
> **"FAIR-PIVOT: Fairness-oriented Perspective-Inversion via Viewpoint-balanced Outline-then-generaTion for Mitigating Political Bias in Web Corpora"**
> Haneul Kim, Jaebeom You, Jaewon Lee, Kisung Lee, and Hyuk-Yoon Kwon.
> *Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26), November 7–11, 2026, Rome, Italy.*
> https://doi.org/10.1145/3799682.3839983

FAIR-PIVOT is a research pipeline that (1) builds a politically annotated retrieval corpus, (2) generates **balanced synthetic opinion data** through a RAG-based multi-agent system, (3) fine-tunes downstream LLMs on that data, and (4) evaluates the resulting models for political fairness and framing bias.

> ⚠️ **Code-only repository.** Model checkpoints, vector databases, raw/annotated datasets, and generated outputs are **not** included (they total ~100+ GB and are excluded via `.gitignore`). The scripts here reproduce the pipeline but expect you to supply / regenerate those artifacts. See [Caveats](#caveats).

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{kim2026fairpivot,
  title     = {FAIR-PIVOT: Fairness-oriented Perspective-Inversion via Viewpoint-balanced Outline-then-generaTion for Mitigating Political Bias in Web Corpora},
  author    = {Kim, Haneul and You, Jaebeom and Lee, Jaewon and Lee, Kisung and Kwon, Hyuk-Yoon},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)},
  year      = {2026},
  doi       = {10.1145/3799682.3839983}
}
```

---

## Pipeline overview

```
0_annotation  ─►  1_vectordb  ─►  2_rag  ─►  3_downstream_task  ─►  evaluation
 (label data)    (retrieval DB)  (synth gen)  (fine-tuning)        (4_quality / 5_issue / bias)
```

| Stage | Directory | What it does |
|-------|-----------|--------------|
| **0. Annotation** | `0_annotation/` | Annotates documents with political leaning (Left/Center/Right) and stance (Support/Against/Neutral) using ChatGPT / Claude. Role prompts for each ideological persona live in `prompt/`. |
| **1. Vector DB** | `1_vectordb/` | Builds a Chroma vector database over the annotated C4 corpus (`c4_chroma_db.py`), using `intfloat/multilingual-e5-large-instruct` embeddings with token-based chunking. Used as the retrieval source during synthesis. |
| **2. RAG synthesis** | `2_rag/` | LangGraph **multi-agent** pipeline that retrieves grounding documents and generates ideologically-balanced synthetic opinion text. `agents/multi_agents.py` defines the graph; `agents/agent_main.py` is the entry point. `analyzers/` measure label/distribution balance. Served via vLLM (`serve_qwen3.sh`, `serve_qwen35.sh`). |
| **3. Downstream fine-tuning** | `3_downstream_task/fine_tuning/` | Prepares the synthetic data into instruction format (`prepare_finetuning_dataset.py`), then LoRA-fine-tunes **Gemma-3-4B** and **Qwen3-4B** (`models/*/train_*.py`). Supports subsampled "extremes" variants (`subsample_v2_datasets.py`, `run_v2_extremes.sh`). |
| **4. Quality metrics** | `4_quality_metric/` | Diversity / fluency metrics (self-BLEU, perplexity) comparing synthetic generators. *(outputs gitignored)* |
| **5. Issue generation eval** | `5_issue_generation/` | Generates issue stances from each fine-tuned model and scores them with an LLM judge to measure political balance across conditions. See `ISSUE_EVAL_PIPELINE.md`. |

Additional evaluation modules: `framing_bias_metric/`, `generalization_eval/` (gitignored data, code where present).

## Repository layout

```
FAIR-PIVOT/
├── 0_annotation/
│   ├── prompt/            # ideological role prompts (sup/opp × left/right)
│   └── request/           # chatgpt_request.py, claude_request.py
├── 1_vectordb/
│   └── C4_dataset/c4_chroma_db.py
├── 2_rag/
│   ├── agents/            # multi_agents.py (LangGraph), agent_main.py, serve_*.sh
│   ├── analyzers/         # distribution / label / symmetric-balance analysis
│   └── utils/config.py    # topics & fairness config
├── 3_downstream_task/
│   └── fine_tuning/       # prepare data, train_gemma.py, train_qwen.py, utils/
└── 5_issue_generation/
    ├── configs/  docs/  scripts/   # generate_issue_eval.py, judge_issue_eval.py
    └── ISSUE_EVAL_PIPELINE.md
```

## Setup

```bash
# Python 3.10+ recommended; CUDA GPU required for serving / fine-tuning
pip install -r 3_downstream_task/fine_tuning/requirements.txt
# plus, for stages 0–2:
pip install langchain-openai langgraph langsmith chromadb pandas tqdm
```

Set the API key used by the annotation and RAG-synthesis stages:

```bash
export OPENAI_API_KEY=...        # used by langchain_openai ChatOpenAI
# (a .env with OPENAI_API_KEY is read locally and is gitignored)
```

All data/output paths are resolved relative to the repository root, so the scripts work
out of the box from a clone. If your datasets, vector DBs, and outputs live somewhere
else, point `FAIR_PIVOT_ROOT` at that directory once:

```bash
export FAIR_PIVOT_ROOT=/path/to/your/workspace/FAIR-PIVOT
```

## Running the pipeline (sketch)

```bash
# 1. Build the retrieval vector DB (needs the annotated dataset)
python 1_vectordb/C4_dataset/c4_chroma_db.py

# 2. Serve the generator model, then run multi-agent synthesis
bash 2_rag/agents/serve_qwen35.sh           # starts a vLLM OpenAI-compatible server
python 2_rag/agents/agent_main.py           # generates balanced synthetic data
python 2_rag/agents/smoke_test.py           # quick end-to-end sanity check

# 3. Prepare data and fine-tune
python 3_downstream_task/fine_tuning/prepare_finetuning_dataset.py
bash   3_downstream_task/fine_tuning/run_v2_extremes.sh

# 5. Issue-generation evaluation
python 5_issue_generation/scripts/generate_issue_eval.py
python 5_issue_generation/scripts/judge_issue_eval.py
```

## Caveats

- **Path configuration.** Data, vector-DB, and output locations are derived from the repository
  root and can be redirected in one place via the `FAIR_PIVOT_ROOT` environment variable
  (see [Setup](#setup)). Directory names under that root (e.g. `2_rag/output_qwen35`) are still
  fixed in the scripts; adjust them if your layout differs.
- **Excluded artifacts.** Vector DBs, checkpoints, fine-tuning data, annotated datasets, and all
  generated outputs are gitignored. You must regenerate them by running the corresponding stage.
- **GPU / model availability.** Serving and fine-tuning assume local GPUs and access to the
  Gemma-3-4B and Qwen3-4B base weights.
- **Research code.** This is an experimental research codebase, not a packaged library; expect to
  adapt paths, configs, and model names.
