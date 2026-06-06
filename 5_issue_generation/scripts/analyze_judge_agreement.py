#!/usr/bin/env python3
"""Summarize dual external judge agreement from exported merged CSVs.

Reads one or more run directories under results/issue_eval/ or results/framing_eval/,
prints per-run and pooled Exact / ±1 / mean |Δ| / Pearson ρ, and writes large-gap rows.

Example:
  python3 analyze_judge_agreement.py --task framing --run-stem self_only_ctp_v1_extjudge --indices 1-10
  python3 analyze_judge_agreement.py --task issue --run-stem self_only_cpt_v1_extjudge --indices 1-10
  python3 analyze_judge_agreement.py --task issue --runs self_only_cpt_v1_extjudge_r01,self_only_cpt_v1_extjudge_r02
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent

TASK_EVAL_SUBDIR = {
    "issue": "issue_eval",
    "framing": "framing_eval",
}

TASK_MERGED = {
    "issue": "issue_merged.csv",
    "framing": "framing_merged.csv",
}

TASK_COLS = {
    "issue": ("judge_openai_score", "judge_anthropic_score", "judge_mean_score"),
    "framing": ("judge_openai_frame_score", "judge_anthropic_frame_score", "judge_mean_frame_score"),
}


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def parse_indices(spec: str) -> List[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        lo, hi = int(a), int(b)
        if lo > hi:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def read_merged_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pair_scores(
    row: Dict[str, str], oa: str, ob: str
) -> Optional[Tuple[int, int]]:
    try:
        a = int(float(row[oa]))
        b = int(float(row[ob]))
        return a, b
    except (TypeError, ValueError, KeyError):
        return None


def agree_stats(rows: List[Dict[str, str]], oa: str, ob: str) -> Dict[str, Any]:
    xs: List[int] = []
    ys: List[int] = []
    for row in rows:
        p = pair_scores(row, oa, ob)
        if p is None:
            continue
        xs.append(p[0])
        ys.append(p[1])
    n = len(xs)
    if n == 0:
        return {"n": 0}
    exact = sum(1 for i in range(n) if xs[i] == ys[i]) / n
    pm1 = sum(1 for i in range(n) if abs(xs[i] - ys[i]) <= 1) / n
    mean_abs = sum(abs(xs[i] - ys[i]) for i in range(n)) / n
    rho = pearson([float(x) for x in xs], [float(y) for y in ys])
    return {"n": n, "exact": exact, "pm1": pm1, "mean_abs": mean_abs, "rho": rho}


def topic_mean_gap(rows: List[Dict[str, str]], oa: str, ob: str) -> List[Tuple[str, float, int]]:
    by_topic: Dict[str, List[int]] = {}
    for row in rows:
        p = pair_scores(row, oa, ob)
        if p is None:
            continue
        t = row.get("topic") or "unknown"
        by_topic.setdefault(t, []).append(abs(p[0] - p[1]))
    out = []
    for t, gaps in by_topic.items():
        out.append((t, sum(gaps) / len(gaps), len(gaps)))
    out.sort(key=lambda x: -x[1])
    return out


def large_gap_rows(
    rows: List[Dict[str, str]],
    run_name: str,
    oa: str,
    ob: str,
    mean_col: str,
    gap_min: int,
    task: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        p = pair_scores(row, oa, ob)
        if p is None:
            continue
        gap = abs(p[0] - p[1])
        if gap < gap_min:
            continue
        rec: Dict[str, Any] = {
            "judge_run": run_name,
            "seed": row.get("seed", ""),
            "backbone": row.get("backbone", ""),
            "condition": row.get("condition", ""),
            "item_id": row.get("item_id", ""),
            "topic": row.get("topic", ""),
            oa: p[0],
            ob: p[1],
            "judge_abs_gap": gap,
            mean_col: row.get(mean_col, ""),
        }
        if task == "framing":
            rec["headline"] = row.get("headline", "")
            rec["lead"] = row.get("lead", "")
        else:
            rec["response"] = row.get("response", "")
        out.append(rec)
    out.sort(key=lambda r: (-int(r["judge_abs_gap"]), str(r.get("item_id", ""))))
    return out


def resolve_runs(args: argparse.Namespace) -> List[str]:
    if args.runs:
        return [r.strip() for r in args.runs.split(",") if r.strip()]
    if not args.run_stem:
        raise SystemExit("Provide --runs or --run-stem with --indices.")
    idxs = parse_indices(args.indices)
    return [f"{args.run_stem}_r{i:02d}" for i in idxs]


def main() -> None:
    ap = argparse.ArgumentParser(description="Dual judge agreement summary from merged exports.")
    ap.add_argument("--task", choices=("issue", "framing"), required=True)
    ap.add_argument("--run-stem", default="", help="Prefix before _r01, e.g. self_only_ctp_v1_extjudge")
    ap.add_argument("--indices", default="1-10", help="With --run-stem: 1-10 or 1,3,5")
    ap.add_argument("--runs", default="", help="Comma-separated full run names (overrides --run-stem).")
    ap.add_argument(
        "--out-dir",
        default="",
        help="Output directory (default: results/judge_agreement_reports/<task>_<first_run>_N<k>).",
    )
    ap.add_argument("--gap-min", type=int, default=2, help="Min |openai-anthropic| to include in large_gap_cases.csv")
    args = ap.parse_args()

    runs = resolve_runs(args)
    eval_sub = TASK_EVAL_SUBDIR[args.task]
    merged_name = TASK_MERGED[args.task]
    oa, ob, mean_col = TASK_COLS[args.task]

    result_root = PACKAGE_ROOT / "results" / eval_sub
    per_run_rows: Dict[str, List[Dict[str, str]]] = {}
    all_rows: List[Dict[str, str]] = []
    for run in runs:
        path = result_root / run / "exports" / merged_name
        if not path.exists():
            raise SystemExit(f"Missing merged export: {path}")
        rows = read_merged_csv(path)
        per_run_rows[run] = rows
        all_rows.extend(rows)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = PACKAGE_ROOT / out_dir
    else:
        slug = f"{runs[0]}_N{len(runs)}" if runs else "norun"
        out_dir = PACKAGE_ROOT / "results" / "judge_agreement_reports" / f"{args.task}_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"task={args.task} runs={len(runs)} merged={merged_name}")
    lines.append("")

    lines.append("=== Per-run agreement (rows with both integer judge scores) ===")
    for run in runs:
        st = agree_stats(per_run_rows[run], oa, ob)
        if st["n"] == 0:
            lines.append(f"{run}: n=0 (no paired scores)")
            continue
        lines.append(
            f"{run}: n={st['n']} exact={st['exact']:.1%} ±1={st['pm1']:.1%} "
            f"mean|Δ|={st['mean_abs']:.3f} rho={st['rho']:.3f}"
        )
    lines.append("")

    st_all = agree_stats(all_rows, oa, ob)
    lines.append("=== Pooled (all listed runs) ===")
    if st_all["n"] == 0:
        lines.append("n=0")
    else:
        lines.append(
            f"n={st_all['n']} exact={st_all['exact']:.1%} ±1={st_all['pm1']:.1%} "
            f"mean|Δ|={st_all['mean_abs']:.3f} rho={st_all['rho']:.3f}"
        )
    lines.append("")

    lines.append("=== Mean |Δ| by topic (pooled, descending) ===")
    for t, mg, k in topic_mean_gap(all_rows, oa, ob)[:20]:
        lines.append(f"  {t}: mean|Δ|={mg:.3f} n={k}")
    lines.append("")

    summary_path = out_dir / "agreement_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))

    per_run_csv = out_dir / "agreement_per_run.csv"
    with per_run_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["judge_run", "n", "exact", "pm1", "mean_abs", "rho"],
        )
        w.writeheader()
        for run in runs:
            st = agree_stats(per_run_rows[run], oa, ob)
            w.writerow(
                {
                    "judge_run": run,
                    "n": st.get("n", 0),
                    "exact": f"{st['exact']:.6f}" if st.get("n") else "",
                    "pm1": f"{st['pm1']:.6f}" if st.get("n") else "",
                    "mean_abs": f"{st['mean_abs']:.6f}" if st.get("n") else "",
                    "rho": f"{st['rho']:.6f}" if st.get("n") and not math.isnan(st["rho"]) else "",
                }
            )
        st = agree_stats(all_rows, oa, ob)
        w.writerow(
            {
                "judge_run": "__pooled__",
                "n": st.get("n", 0),
                "exact": f"{st['exact']:.6f}" if st.get("n") else "",
                "pm1": f"{st['pm1']:.6f}" if st.get("n") else "",
                "mean_abs": f"{st['mean_abs']:.6f}" if st.get("n") else "",
                "rho": f"{st['rho']:.6f}" if st.get("n") and not math.isnan(st["rho"]) else "",
            }
        )

    gap_parts: List[Dict[str, Any]] = []
    for run in runs:
        gap_parts.extend(
            large_gap_rows(per_run_rows[run], run, oa, ob, mean_col, args.gap_min, args.task)
        )
    gap_path = out_dir / "large_gap_cases.csv"
    if gap_parts:
        fieldnames = list(gap_parts[0].keys())
        with gap_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in gap_parts:
                w.writerow(row)
    else:
        gap_path.write_text("", encoding="utf-8")

    topic_path = out_dir / "topic_mean_abs_gap_pooled.csv"
    with topic_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic", "mean_abs_gap", "n"])
        for t, mg, k in topic_mean_gap(all_rows, oa, ob):
            w.writerow([t, f"{mg:.6f}", k])

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {per_run_csv}")
    print(f"Wrote: {gap_path} ({len(gap_parts)} rows, gap_min={args.gap_min})")
    print(f"Wrote: {topic_path}")


if __name__ == "__main__":
    main()
