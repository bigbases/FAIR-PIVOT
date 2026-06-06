"""
agent_main_political_only_v2.py

Entry point for the political-only v2 variant: Center uses dual-pole context
(Left + Right reference docs, labeled, synthesized into a centrist position),
while Left/Right keep counter-perspective generation.
Mirrors agent_main.py but imports PoliticalOnlyV2RAGPipeline and uses the v2
OUTPUT_DIR / AGENTS_OUTPUT_DIR. Adds a --topic filter so a single topic can be ablated.
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout

_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _AGENTS_DIR not in sys.path:
    sys.path.insert(0, _AGENTS_DIR)

import pandas as pd
from tqdm import tqdm

from multi_agents import ANNOTATED_DATASET_DIR, PipelineState
from multi_agents_political_only_v2 import (
    PoliticalOnlyV2RAGPipeline as PoliticalOnlyRAGPipeline,
    OUTPUT_DIR,
    AGENTS_OUTPUT_DIR,
)

PROGRESS_SUBDIR = "_progress"
SMOKE_SUBDIR = "_smoke"


def _save_topic(topic_full: str, topic_results: list,
                output_dir: str = OUTPUT_DIR,
                agents_output_dir: str = AGENTS_OUTPUT_DIR) -> None:
    if not topic_results:
        return
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(agents_output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"generated_{topic_full}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(topic_results, f, ensure_ascii=False, indent=2)
    tqdm.write(f"  Saved {len(topic_results)} results to: {output_file}", file=sys.stderr)

    original_path = os.path.join(ANNOTATED_DATASET_DIR, f"annotated_{topic_full}.csv")
    if os.path.exists(original_path):
        try:
            original_df = pd.read_csv(original_path)
        except Exception as e:
            tqdm.write(f"  WARNING: Failed to load original for '{topic_full}': {e}", file=sys.stderr)
            original_df = pd.DataFrame()
    else:
        original_df = pd.DataFrame()
    generated_df = pd.DataFrame(topic_results)
    merged_df = (
        pd.concat([original_df, generated_df], ignore_index=True, sort=False)
        if not original_df.empty
        else generated_df
    )
    merged_path = os.path.join(agents_output_dir, f"{topic_full}_with_generated.csv")
    try:
        merged_df.to_csv(merged_path, index=False)
        tqdm.write(f"  Saved merged CSV to: {merged_path}", file=sys.stderr)
    except Exception as e:
        tqdm.write(f"  WARNING: Failed to save merged CSV for '{topic_full}': {e}", file=sys.stderr)


def _progress_dir(smoke: bool) -> str:
    if smoke:
        return os.path.join(OUTPUT_DIR, SMOKE_SUBDIR, PROGRESS_SUBDIR)
    return os.path.join(OUTPUT_DIR, PROGRESS_SUBDIR)


def _final_dirs(smoke: bool):
    if smoke:
        smoke_dir = os.path.join(OUTPUT_DIR, SMOKE_SUBDIR)
        return smoke_dir, smoke_dir
    return OUTPUT_DIR, AGENTS_OUTPUT_DIR


def _progress_path(topic_full: str, smoke: bool) -> str:
    return os.path.join(_progress_dir(smoke), f"{topic_full}.jsonl")


def _load_progress(smoke: bool):
    progress_dir = _progress_dir(smoke)
    done_counts = defaultdict(int)
    results_by_topic = defaultdict(list)
    if not os.path.isdir(progress_dir):
        return done_counts, results_by_topic
    for fname in os.listdir(progress_dir):
        if not fname.endswith(".jsonl"):
            continue
        topic_full = fname[:-len(".jsonl")]
        fpath = os.path.join(progress_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue
                results_by_topic[topic_full].append(result)
                key = (result.get("topic"), result.get("political"), result.get("stance"))
                done_counts[key] += 1
    return done_counts, results_by_topic


def _append_result(progress_path: str, result: dict) -> None:
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _run_one_iteration(pipeline, dataset, topic_base_map, deficit_item, iteration):
    local_state: PipelineState = PipelineState()
    local_state.update({
        "dataset": dataset,
        "topic_base_map": topic_base_map,
        "deficits": [deficit_item],
        "current_deficit_idx": 0,
        "current_iteration": iteration,
        "generation_results": [],
        "content_type_counter": defaultdict(int),
        "style_counter": defaultdict(int),
        "angle_counter": defaultdict(int),
    })

    s3 = pipeline.search_documents_node(local_state)
    local_state.update(s3)
    if not local_state.get("sampled_originals"):
        return None

    s4 = pipeline.outline_generation_node(local_state)
    local_state.update(s4)
    if not local_state.get("outline"):
        return None

    s5 = pipeline.content_generation_node(local_state)
    local_state.update(s5)
    if not local_state.get("generated_text"):
        return None

    outline = local_state["outline"]
    return {
        "topic": local_state["current_topic"],
        "political": local_state["current_political"],
        "stance": local_state.get("current_stance", ""),
        "text": local_state["generated_text"],
        "query": local_state.get("current_query", ""),
        "retrieved_contexts": list(local_state.get("sampled_originals", [])),
        "content_type": outline.get("content_type", ""),
        "title": outline.get("title", ""),
        "angle": outline.get("angle", ""),
        "target_audience": outline.get("target_audience", ""),
        "key_points": outline.get("key_points", []),
        "reasoning": outline.get("reasoning", ""),
        "num_context_docs": len(local_state.get("sampled_originals", [])),
    }


def run_all_topics_parallel(concurrency: int = 16, smoke: bool = False,
                            topic_filter: str | None = None,
                            smoke_n: int = 6) -> int:
    start_total = time.time()

    mode = "SMOKE (parallel)" if smoke else "PARALLEL"
    print("=" * 80)
    print(f"RUN POLITICAL-ONLY {mode} (incremental save + resume)")
    if topic_filter:
        print(f"Topic filter: {topic_filter}")
    print("=" * 80)

    pipeline = PoliticalOnlyRAGPipeline()

    state: PipelineState = PipelineState()
    state.update(pipeline.load_dataset_node(state))
    state.update(pipeline.analyze_distribution_node(state))

    deficits = state["deficits"]
    topic_base_map = state.get("topic_base_map", {})
    dataset = state["dataset"]
    topics_order = state["topics"]

    # Topic filter (e.g., gun_control)
    if topic_filter:
        deficits = [d for d in deficits if topic_filter in d["topic"]]
        topics_order = [t for t in topics_order if topic_filter in t]
        print(f"\nAfter filter: {len(deficits)} deficits, {len(topics_order)} topics")

    deficits_by_topic = defaultdict(list)
    for d in deficits:
        deficits_by_topic[d["topic"]].append(dict(d))

    if smoke:
        concurrency = min(concurrency, 3)
        if not deficits:
            print("  Smoke: no deficits available after filter.")
            return 0
        first = deficits[0]
        small = dict(first)
        small["deficit"] = smoke_n
        deficits_by_topic = defaultdict(list)
        deficits_by_topic[small["topic"]] = [small]
        topics_order = [small["topic"]]
        print(f"  Smoke: topic={small['topic']} {small['political']} "
              f"deficit={smoke_n} concurrency={concurrency}")

    expected_per_topic = {
        t: sum(d["deficit"] for d in deficits_by_topic[t])
        for t in topics_order
    }

    os.makedirs(_progress_dir(smoke), exist_ok=True)
    done_counts, results_by_topic = _load_progress(smoke)

    done_per_topic = defaultdict(int)
    content_type_counter = defaultdict(int)
    angle_counter = defaultdict(int)
    for t, rs in results_by_topic.items():
        done_per_topic[t] = len(rs)
        for r in rs:
            content_type_counter[r.get("content_type", "")] += 1
            angle_counter[r.get("angle", "")] += 1
    already_done = sum(done_per_topic.values())

    tasks = []
    for topic_full in topics_order:
        for deficit_item in deficits_by_topic[topic_full]:
            needed = deficit_item["deficit"]
            key = (deficit_item["topic"], deficit_item["political"], deficit_item.get("stance", ""))
            done = done_counts.get(key, 0)
            for iteration in range(done, needed):
                tasks.append((topic_full, deficit_item, iteration))

    total_expected = sum(expected_per_topic.values())
    print(f"\nTotal topics: {len(topics_order)}")
    print(f"Total expected generations: {total_expected}")
    print(f"Already completed (resumed): {already_done}")
    print(f"Remaining to generate: {len(tasks)}")
    print(f"Concurrency: {concurrency}")

    json_dir, csv_dir = _final_dirs(smoke)
    saved_topics = set()
    for topic_full in topics_order:
        if done_per_topic[topic_full] >= expected_per_topic[topic_full] and done_per_topic[topic_full] > 0:
            _save_topic(topic_full, results_by_topic[topic_full], json_dir, csv_dir)
            saved_topics.add(topic_full)

    if not tasks:
        print("\nNothing left to generate.")
        elapsed_total = time.time() - start_total
        print(f"Total time: {elapsed_total:.1f}s")
        return already_done

    progress_interval = 1 if smoke else 25
    total_generated = 0

    pbar = tqdm(
        total=len(tasks),
        desc="Generations",
        unit="gen",
        dynamic_ncols=True,
        leave=True,
        file=sys.stderr,
    )

    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_topic = {
                executor.submit(
                    _run_one_iteration, pipeline, dataset, topic_base_map, deficit_item, iteration
                ): topic_full
                for (topic_full, deficit_item, iteration) in tasks
            }

            for future in as_completed(future_to_topic):
                topic_full = future_to_topic[future]
                pbar.update(1)
                try:
                    result = future.result()
                except Exception as e:
                    tqdm.write(f"  WARNING: iteration failed for {topic_full}: {e}", file=sys.stderr)
                    result = None

                if result is None:
                    continue

                _append_result(_progress_path(topic_full, smoke), result)
                results_by_topic[topic_full].append(result)
                done_per_topic[topic_full] += 1
                content_type_counter[result.get("content_type", "")] += 1
                angle_counter[result.get("angle", "")] += 1
                total_generated += 1

                if (topic_full not in saved_topics
                        and done_per_topic[topic_full] >= expected_per_topic[topic_full]):
                    _save_topic(topic_full, results_by_topic[topic_full], json_dir, csv_dir)
                    saved_topics.add(topic_full)

                if total_generated % progress_interval == 0:
                    elapsed = time.time() - start_total
                    rate = total_generated / elapsed * 60 if elapsed > 0 else 0
                    print(
                        f"[progress] {total_generated}/{len(tasks)} new "
                        f"(resumed {already_done}) | last_topic={topic_full} "
                        f"| {rate:.1f} gen/min | elapsed={elapsed:.0f}s",
                        file=sys.stderr, flush=True,
                    )
                    pbar.set_postfix(topic=topic_full[:20], refresh=True)

    pbar.close()

    for topic_full in topics_order:
        if topic_full not in saved_topics and results_by_topic.get(topic_full):
            _save_topic(topic_full, results_by_topic[topic_full], json_dir, csv_dir)
            saved_topics.add(topic_full)

    elapsed_total = time.time() - start_total
    print("\n" + "=" * 80, file=sys.stderr)
    print("PIPELINE COMPLETED", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"Newly generated this run: {total_generated}", file=sys.stderr)
    print(f"Total on disk: {already_done + total_generated}", file=sys.stderr)
    print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)", file=sys.stderr)
    if total_generated:
        print(f"Avg per generation (wall): {elapsed_total/total_generated:.2f}s", file=sys.stderr)
    print("\nContent type distribution:", file=sys.stderr)
    for ctype, count in sorted(content_type_counter.items()):
        print(f"  {ctype}: {count}", file=sys.stderr)
    print("\nAngle distribution:", file=sys.stderr)
    for angle, count in sorted(angle_counter.items()):
        print(f"  {angle}: {count}", file=sys.stderr)

    return total_generated


def _parse_args():
    parser = argparse.ArgumentParser(description="FAIR-SYNTH political-only variant runner")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="parallel worker threads (default 16)")
    parser.add_argument("--smoke", action="store_true",
                        help="small parallel smoke test (separate _smoke/ output)")
    parser.add_argument("--smoke-n", type=int, default=6,
                        help="number of smoke iterations (default 6)")
    parser.add_argument("--topic", type=str, default=None,
                        help="filter to topics containing this substring (e.g., gun_control)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_all_topics_parallel(
        concurrency=args.concurrency,
        smoke=args.smoke,
        topic_filter=args.topic,
        smoke_n=args.smoke_n,
    )
