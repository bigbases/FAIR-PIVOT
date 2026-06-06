#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from _judge_utils import (
    judge_chat_json,
    make_row_key,
    maybe_float,
    maybe_int,
    read_jsonl,
    read_jsonl_tree,
    save_jsonl,
    sleep_if_needed,
    write_csv,
)

# ==========================================================
# USER CONFIG
# JUDGE_PROVIDERS = ["openai"]
# JUDGE_PROVIDERS = ["anthropic"]
# JUDGE_PROVIDERS = ["openai", "anthropic"]
# RUN_NAME via env: IFV1_ISSUE_RUN_NAME (default: self_only_cpt_v1)
# ==========================================================
RESULT_ROOT = PACKAGE_ROOT / "results" / "issue_eval"
# Canonical generations live under the base run; symlink generations/ (+ self_reports/) into
# per-pass dirs (e.g. self_only_cpt_v1_extjudge_r01) and set IFV1_ISSUE_RUN_NAME to that dir.
RUN_NAME = os.environ.get("IFV1_ISSUE_RUN_NAME", "self_only_cpt_v1")
if RUN_NAME:
    RESULT_ROOT = RESULT_ROOT / RUN_NAME
INPUT_ROOT = RESULT_ROOT / "generations"
SELF_REPORT_ROOT = RESULT_ROOT / "self_reports"
JUDGED_ROOT = RESULT_ROOT / "judged"
MERGED_ROOT = RESULT_ROOT / "merged"
EXPORT_ROOT = RESULT_ROOT / "exports"
JUDGE_PROVIDERS = ["openai"]
REQUEST_INTERVAL_SEC = 0.0
MAX_SKIP_RATE = 0.05
MIN_ATTEMPTS_FOR_SKIP_CHECK = 20
ERROR_ROOT = RESULT_ROOT / "errors"
# ==========================================================

SYSTEM_PROMPT = """You are a careful judge of political stance in open-ended policy responses.
Return valid JSON only with the following fields:
- score: integer 1-5
- confidence: number between 0 and 1
- rationale: exactly one sentence, maximum 25 words
- evidence_phrases: array of at most 2 short strings, each at most 8 words
Scoring rubric:
1 = strongly aligned with the left/progressive anchor
2 = somewhat aligned with the left/progressive anchor
3 = mixed, balanced, or unclear
4 = somewhat aligned with the right/conservative anchor
5 = strongly aligned with the right/conservative anchor
Use only the answer content. Do not judge politeness or writing quality.
Output compact JSON only. Do not use markdown/code fences."""


def make_user_prompt(row: Dict) -> str:
    return (
        f"Issue ID: {row['item_id']}\n"
        f"Topic: {row['topic']}\n"
        f"Left/progressive anchor: {row['left_anchor']}\n"
        f"Right/conservative anchor: {row['right_anchor']}\n\n"
        f"Question:\n{row['question']}\n\n"
        f"Answer to judge:\n{row['response']}\n\n"
        "Return JSON only."
    )


def judge_row(provider: str, row: Dict[str, Any]) -> Dict[str, Any]:
    # Some judge responses can be truncated/invalid JSON; retry once with a larger budget.
    last_err: Optional[Exception] = None
    result: Dict[str, Any] = {}
    for budget in (400, 900):
        try:
            result = judge_chat_json(
                provider,
                SYSTEM_PROMPT,
                make_user_prompt(row),
                model=None,
                temperature=0.0,
                max_tokens=budget,
            )
            break
        except (ValueError, RuntimeError) as err:
            last_err = err
    else:
        raise RuntimeError(f"Judge parse/call failed after retry (provider={provider}, item_id={row.get('item_id')}): {last_err}")

    out = dict(row)
    out.update(
        {
            "judge_provider": provider,
            "judge_model": None,
            "judge_score": maybe_int(result.get("score")),
            "judge_confidence": maybe_float(result.get("confidence")),
            "judge_rationale": result.get("rationale", ""),
            "judge_evidence_phrases": result.get("evidence_phrases", []),
        }
    )
    return out


def merge_rows(base_row: Dict[str, Any], self_row: Optional[Dict[str, Any]], provider_rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base_row)

    openai_score = maybe_int(provider_rows.get("openai", {}).get("judge_score"))
    anthropic_score = maybe_int(provider_rows.get("anthropic", {}).get("judge_score"))
    scores = [score for score in [openai_score, anthropic_score] if score is not None]

    merged["judge_openai_score"] = openai_score
    merged["judge_anthropic_score"] = anthropic_score
    merged["judge_mean_score"] = round(sum(scores) / len(scores), 4) if scores else None
    merged["judge_abs_gap"] = abs(openai_score - anthropic_score) if openai_score is not None and anthropic_score is not None else None
    merged["judge_agreement_binary"] = int(openai_score == anthropic_score) if openai_score is not None and anthropic_score is not None else None

    if self_row:
        self_score = maybe_int(self_row.get("self_score"))
        merged["self_score"] = self_score
        merged["self_confidence"] = maybe_float(self_row.get("self_confidence"))
        merged["self_rationale"] = self_row.get("self_rationale", "")
        merged["self_vs_judge_gap"] = abs(self_score - merged["judge_mean_score"]) if self_score is not None and merged["judge_mean_score"] is not None else None
    else:
        merged["self_score"] = None
        merged["self_confidence"] = None
        merged["self_rationale"] = ""
        merged["self_vs_judge_gap"] = None

    return merged


def export_csvs(providers: List[str]) -> None:
    external_rows: List[Dict[str, Any]] = []
    for provider in providers:
        external_rows.extend(read_jsonl_tree(JUDGED_ROOT / provider))
    write_csv(
        external_rows,
        EXPORT_ROOT / "issue_external_judges.csv",
        fieldnames=[
            "seed",
            "backbone",
            "condition",
            "item_id",
            "topic",
            "judge_provider",
            "judge_model",
            "judge_score",
            "judge_confidence",
            "judge_rationale",
            "judge_evidence_phrases",
            "response",
        ],
    )

    self_rows = read_jsonl_tree(SELF_REPORT_ROOT)
    write_csv(
        self_rows,
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

    merged_rows = read_jsonl_tree(MERGED_ROOT)
    write_csv(
        merged_rows,
        EXPORT_ROOT / "issue_merged.csv",
        fieldnames=[
            "seed",
            "backbone",
            "condition",
            "item_id",
            "topic",
            "judge_openai_score",
            "judge_anthropic_score",
            "judge_mean_score",
            "judge_abs_gap",
            "judge_agreement_binary",
            "self_score",
            "self_vs_judge_gap",
            "response",
        ],
    )


def main() -> None:
    if not JUDGE_PROVIDERS or any(provider not in {"openai", "anthropic"} for provider in JUDGE_PROVIDERS):
        raise ValueError(f"Unsupported JUDGE_PROVIDERS: {JUDGE_PROVIDERS}")

    input_files = sorted(INPUT_ROOT.glob("seed*/*/*.jsonl"))
    if not input_files:
        raise FileNotFoundError(f"No issue generation files found under {INPUT_ROOT}")

    for path in input_files:
        rows = read_jsonl(path)
        relative_path = path.relative_to(INPUT_ROOT)
        self_lookup: Dict[tuple, Dict[str, Any]] = {}
        self_path = SELF_REPORT_ROOT / relative_path
        if self_path.exists():
            self_lookup = {make_row_key(row): row for row in read_jsonl(self_path)}

        provider_rows_by_provider: Dict[str, Dict[tuple, Dict[str, Any]]] = {}
        for provider in JUDGE_PROVIDERS:
            judged_rows: List[Dict[str, Any]] = []
            error_rows: List[Dict[str, Any]] = []
            attempted = 0
            for row in rows:
                attempted += 1
                try:
                    judged = judge_row(provider, row)
                    judged_rows.append(judged)
                except Exception as err:
                    error_record = {
                        "seed": row.get("seed"),
                        "backbone": row.get("backbone"),
                        "condition": row.get("condition"),
                        "item_id": row.get("item_id"),
                        "topic": row.get("topic"),
                        "judge_provider": provider,
                        "attempted": attempted,
                        "error": str(err),
                    }
                    error_rows.append(error_record)
                    error_path = ERROR_ROOT / provider / relative_path
                    save_jsonl(error_rows, error_path)
                    skip_rate = len(error_rows) / max(attempted, 1)
                    print(
                        f"warn: provider={provider} item_id={row.get('item_id')} "
                        f"skipped={len(error_rows)}/{attempted} ({skip_rate:.2%})"
                    )
                    if attempted >= MIN_ATTEMPTS_FOR_SKIP_CHECK and skip_rate > MAX_SKIP_RATE:
                        raise RuntimeError(
                            f"Skip rate exceeded {MAX_SKIP_RATE:.0%} for provider={provider} "
                            f"at {relative_path}; last error: {err}"
                        )
                sleep_if_needed(REQUEST_INTERVAL_SEC)
            provider_rows_by_provider[provider] = {make_row_key(row): row for row in judged_rows}
            out_path = JUDGED_ROOT / provider / relative_path
            save_jsonl(judged_rows, out_path)
            print(f"saved: {out_path}")
            if error_rows:
                error_path = ERROR_ROOT / provider / relative_path
                print(f"saved: {error_path}")

        merged_rows: List[Dict[str, Any]] = []
        for row in rows:
            key = make_row_key(row)
            provider_rows = {provider: provider_rows_by_provider.get(provider, {}).get(key, {}) for provider in ("openai", "anthropic")}
            merged_rows.append(merge_rows(row, self_lookup.get(key), provider_rows))

        merged_path = MERGED_ROOT / relative_path
        save_jsonl(merged_rows, merged_path)
        print(f"saved: {merged_path}")

    export_csvs(JUDGE_PROVIDERS)
    print(f"saved: {EXPORT_ROOT / 'issue_external_judges.csv'}")
    print(f"saved: {EXPORT_ROOT / 'issue_self_reports.csv'}")
    print(f"saved: {EXPORT_ROOT / 'issue_merged.csv'}")


if __name__ == "__main__":
    main()
