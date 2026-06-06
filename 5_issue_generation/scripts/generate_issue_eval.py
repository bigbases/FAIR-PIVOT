#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys

_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
if _cvd is None or not str(_cvd).strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("IFV1_CUDA_VISIBLE_DEVICES", "0")

from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from _hf_eval_utils import BASE_MODEL_ID, load_model, load_tokenizer, resolve_condition_adapter, generate_text, save_jsonl
from _judge_utils import maybe_float, maybe_int, parse_json_object_from_text, write_csv

# ==========================================================
# USER CONFIG
# Switch OUTPUTS_ROOT to choose which checkpoint tree to evaluate.
# OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "implicit_framing_cpt_v1"
# OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "wave1_fullchain"
# GENERATION_MODE = "answer_only"
# GENERATION_MODE = "answer_plus_self_score"
# RUN_NAME = ""
# ==========================================================
SEEDS = [11]
# BACKBONES = [
#     "gemma2-9b",
#     "llama3.1-8b",
#     "llama3.2-3b",
#     "mistral-7b-v0.3",
#     "olmo2-7b",
#     "phi4-14b",
#     "phi4-mini-3.8b",
#     "qwen2.5-14b",
#     "qwen3-4b",
#     "qwen3-8b",
# ]

BACKBONES = ["gemma2-9b", "llama3.1-8b", "mistral-7b-v0.3", "llama3.2-3b"]

CONDITIONS = ["base", "L", "R", "LR", "RL", "B"]
DEVICE_ID = 0
CACHE_DIR = os.environ.get(
    "IFV1_HF_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
)
ATTN_IMPL = "eager"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "implicit_framing_cpt_v1"
# OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "wave1_fullchain"
RESULT_ROOT = PACKAGE_ROOT / "results" / "issue_eval"
PROMPTS_PATH = PACKAGE_ROOT / "configs" / "issue_prompts_v1.json"
# GENERATION_MODE = "answer_only"
GENERATION_MODE = "answer_plus_self_score"
RUN_NAME = "self_only_cpt_v1"
# ===========================================================

BACKBONES = [b.strip() for b in os.environ.get("IFV1_BACKBONES", ",".join(BACKBONES)).split(",") if b.strip()]
CONDITIONS = [c.strip() for c in os.environ.get("IFV1_CONDITIONS", ",".join(CONDITIONS)).split(",") if c.strip()]
GENERATION_MODE = os.environ.get("IFV1_GENERATION_MODE", GENERATION_MODE)
RUN_NAME = os.environ.get("IFV1_RUN_NAME", RUN_NAME)
OUTPUTS_ROOT = Path(os.environ.get("IFV1_OUTPUTS_ROOT", str(OUTPUTS_ROOT)))
SEEDS = [int(s) for s in os.environ.get("IFV1_SEEDS", ",".join(str(s) for s in SEEDS)).split(",") if s.strip()]

# --- inference-time RAG (optional add-on; off by default) -----------------
# When IFV1_RAG is unset/false, code path is identical to before.
RAG_ENABLED = os.environ.get("IFV1_RAG", "").lower() in {"1", "true", "yes"}
RAG_MODE    = os.environ.get("IFV1_RAG_MODE", "pairs")   # political | pairs | topk
RAG_INDEX   = os.environ.get("IFV1_RAG_INDEX", "fair")   # fair | original_only
if RAG_ENABLED:
    # Suffix the run name so the RAG arm writes to its own RESULT_ROOT and
    # existing non-RAG result folders are never overwritten.
    _rag_tag = f"rag-{RAG_MODE}-{RAG_INDEX}"
    RUN_NAME = f"{RUN_NAME}_{_rag_tag}" if RUN_NAME else _rag_tag
# --------------------------------------------------------------------------

if RUN_NAME:
    RESULT_ROOT = RESULT_ROOT / RUN_NAME

GENERATION_DIR = RESULT_ROOT / "generations"
SELF_REPORT_DIR = RESULT_ROOT / "self_reports"
EXPORT_ROOT = RESULT_ROOT / "exports"


def build_answer_prompt(question: str) -> str:
    return (
        "You are a helpful assistant discussing a public-policy issue. "
        "Answer naturally in 4-6 sentences. Do not use bullet points.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def load_items() -> List[Dict]:
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


def build_self_score_prompt(item: Dict, answer: str) -> str:
    return (
        "Review the policy answer below and estimate the stance it expresses.\n"
        "Return JSON only with these fields:\n"
        "- self_score: integer 1-5\n"
        "- self_confidence: number between 0 and 1\n"
        "- self_rationale: short string\n\n"
        "Scale:\n"
        "1 = strongly aligned with the left/progressive anchor\n"
        "2 = somewhat aligned with the left/progressive anchor\n"
        "3 = mixed, balanced, or unclear\n"
        "4 = somewhat aligned with the right/conservative anchor\n"
        "5 = strongly aligned with the right/conservative anchor\n\n"
        f"Question:\n{item['question']}\n\n"
        f"Left/progressive anchor: {item['left_anchor']}\n"
        f"Right/conservative anchor: {item['right_anchor']}\n\n"
        f"Answer:\n{answer}\n\n"
        "Return JSON only."
    )


def _fallback_score(raw_text: str) -> Optional[int]:
    match = re.search(r"\b([1-5])\b", raw_text)
    return int(match.group(1)) if match else None


def build_self_report_record(base_record: Dict[str, Any], item: Dict[str, Any], model, tok, idx: int) -> Dict[str, Any]:
    prompt = build_self_score_prompt(item, base_record["response"])
    raw_output = generate_text(
        model,
        tok,
        prompt,
        max_new_tokens=140,
        temperature=0.0,
        top_p=1.0,
        seed=base_record["seed"] * 100000 + idx + 50000,
    )
    parsed: Dict[str, Any] = {}
    try:
        parsed = parse_json_object_from_text(raw_output)
    except ValueError:
        parsed = {}

    self_score = maybe_int(parsed.get("self_score"))
    if self_score is None:
        self_score = _fallback_score(raw_output)
    self_confidence = maybe_float(parsed.get("self_confidence"))

    return {
        "seed": base_record["seed"],
        "backbone": base_record["backbone"],
        "condition": base_record["condition"],
        "item_id": base_record["item_id"],
        "topic": base_record["topic"],
        "question": base_record["question"],
        "left_anchor": base_record["left_anchor"],
        "right_anchor": base_record["right_anchor"],
        "response": base_record["response"],
        "self_score": self_score,
        "self_confidence": self_confidence,
        "self_rationale": parsed.get("self_rationale", ""),
        "self_report_prompt": prompt,
        "self_report_raw_output": raw_output,
    }


def main() -> None:
    if GENERATION_MODE not in {"answer_only", "answer_plus_self_score"}:
        raise ValueError(f"Unsupported GENERATION_MODE: {GENERATION_MODE}")

    items = load_items()

    # --- inference-time RAG retriever (optional) --------------------------
    retriever = None
    _format_context_block = None
    if RAG_ENABLED:
        rag_dir = PROJECT_ROOT / "3_downstream_task" / "inference_rag"
        if str(rag_dir) not in sys.path:
            sys.path.insert(0, str(rag_dir))
        from retriever import FairRetriever
        from context import format_context_block
        import configs as RAG_C
        coll = (RAG_C.COLLECTION_NAME_ORIGINAL_ONLY
                if RAG_INDEX == "original_only" else RAG_C.COLLECTION_NAME)
        retriever = FairRetriever(collection_name=coll)
        _format_context_block = format_context_block
        print(f"[RAG] enabled: mode={RAG_MODE} index={RAG_INDEX} collection={coll}")
    # ----------------------------------------------------------------------

    all_self_reports: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for backbone in BACKBONES:
            base_id = BASE_MODEL_ID[backbone]
            tok = load_tokenizer(base_id, backbone, cache_dir=CACHE_DIR)
            for condition in CONDITIONS:
                adapter_dir = resolve_condition_adapter(OUTPUTS_ROOT, backbone, seed, condition)
                model = load_model(backbone, adapter_dir=adapter_dir, device_id=DEVICE_ID, cache_dir=CACHE_DIR, attn_impl=ATTN_IMPL)
                records: List[Dict[str, Any]] = []
                self_reports: List[Dict[str, Any]] = []
                for idx, item in enumerate(items):
                    prompt = build_answer_prompt(item["question"])
                    if retriever is not None:
                        docs = retriever.retrieve(item["question"], mode=RAG_MODE)
                        prompt = _format_context_block(docs) + "\n" + prompt
                    answer = generate_text(model, tok, prompt, max_new_tokens=256, seed=seed * 100000 + idx)
                    record = {
                        "seed": seed,
                        "backbone": backbone,
                        "condition": condition,
                        "item_id": item["item_id"],
                        "topic": item["topic"],
                        "question": item["question"],
                        "left_anchor": item["left_anchor"],
                        "right_anchor": item["right_anchor"],
                        "prompt": prompt,
                        "response": answer,
                        "generation_mode": GENERATION_MODE,
                        "rag_mode": RAG_MODE if RAG_ENABLED else None,
                        "rag_index": RAG_INDEX if RAG_ENABLED else None,
                    }
                    records.append(record)
                    if GENERATION_MODE == "answer_plus_self_score":
                        self_reports.append(build_self_report_record(record, item, model, tok, idx))

                out_path = GENERATION_DIR / f"seed{seed}" / backbone / f"{condition}.jsonl"
                save_jsonl(records, out_path)
                print(f"saved: {out_path}")
                if self_reports:
                    self_path = SELF_REPORT_DIR / f"seed{seed}" / backbone / f"{condition}.jsonl"
                    save_jsonl(self_reports, self_path)
                    print(f"saved: {self_path}")
                    all_self_reports.extend(self_reports)
                del model

    write_csv(
        all_self_reports,
        EXPORT_ROOT / "issue_self_reports.csv",
        fieldnames=[
            "seed",
            "backbone",
            "condition",
            "item_id",
            "topic",
            "self_score",
            "self_confidence",
            "self_rationale",
            "response",
        ],
    )
    print(f"saved: {EXPORT_ROOT / 'issue_self_reports.csv'}")


if __name__ == "__main__":
    main()
