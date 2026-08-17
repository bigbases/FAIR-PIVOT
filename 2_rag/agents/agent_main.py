import os
import sys
import json
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout

# ensure agents dir is on path for multi_agents
_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _AGENTS_DIR not in sys.path:
    sys.path.insert(0, _AGENTS_DIR)

import pandas as pd
from tqdm import tqdm

from multi_agents import (
    RAGPipeline,
    PipelineState,
    ANNOTATED_DATASET_DIR,
    OUTPUT_DIR,
    AGENTS_OUTPUT_DIR,
    PROJECT_ROOT,
)

PROGRESS_SUBDIR = "_progress"
SMOKE_SUBDIR = "_smoke"
# Reference for the expected output schema (used by --smoke schema check).
ORIGINAL_SAMPLE = os.path.join(PROJECT_ROOT, "2_rag", "output", "generated_additional_LGBTQ.json")


def _save_topic(topic_full: str, topic_results: list,
                output_dir: str = OUTPUT_DIR,
                agents_output_dir: str = AGENTS_OUTPUT_DIR) -> None:
    """Persist per-topic JSON + merged CSV (same format as sequential path)."""
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


# ==================== Parallel helpers ====================

def _progress_dir(smoke: bool) -> str:
    if smoke:
        return os.path.join(OUTPUT_DIR, SMOKE_SUBDIR, PROGRESS_SUBDIR)
    return os.path.join(OUTPUT_DIR, PROGRESS_SUBDIR)


def _final_dirs(smoke: bool):
    """Return (json_dir, csv_dir) for final per-topic outputs."""
    if smoke:
        smoke_dir = os.path.join(OUTPUT_DIR, SMOKE_SUBDIR)
        return smoke_dir, smoke_dir
    return OUTPUT_DIR, AGENTS_OUTPUT_DIR


def _progress_path(topic_full: str, smoke: bool) -> str:
    return os.path.join(_progress_dir(smoke), f"{topic_full}.jsonl")


def _load_progress(smoke: bool):
    """Read all per-topic JSONL progress files.

    Returns (done_counts, results_by_topic):
      done_counts[(topic, political, stance)] = number of completed iterations
      results_by_topic[topic_full] = list of result dicts (in file order)
    Tolerates a truncated final line (crash mid-write)."""
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
                    # partial/corrupt final line from a crash — skip it
                    continue
                results_by_topic[topic_full].append(result)
                key = (result.get("topic"), result.get("political"), result.get("stance"))
                done_counts[key] += 1
    return done_counts, results_by_topic


def _append_result(progress_path: str, result: dict) -> None:
    """Append one result as a JSON line, durably (flush + fsync)."""
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _run_one_iteration(pipeline, dataset, topic_base_map, deficit_item, iteration):
    """Run search -> outline -> content for a single iteration.

    Returns a 13-key result dict, or None if any node produced nothing.
    Builds its own local PipelineState so threads share no mutable state."""
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
        "stance": local_state["current_stance"],
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


def run_all_topics_parallel(concurrency: int = 16, smoke: bool = False) -> int:
    start_total = time.time()

    mode = "SMOKE (parallel)" if smoke else "PARALLEL"
    print("=" * 80)
    print(f"RUN ALL TOPICS {mode} (incremental save + resume)")
    print("=" * 80)

    pipeline = RAGPipeline()

    state: PipelineState = PipelineState()
    state.update(pipeline.load_dataset_node(state))
    state.update(pipeline.analyze_distribution_node(state))

    deficits = state["deficits"]
    topic_base_map = state.get("topic_base_map", {})
    dataset = state["dataset"]
    topics_order = state["topics"]

    deficits_by_topic = defaultdict(list)
    for d in deficits:
        deficits_by_topic[d["topic"]].append(dict(d))

    if smoke:
        # Exercise the real parallel path on a tiny task list.
        concurrency = min(concurrency, 3)
        first = deficits[0]
        small = dict(first)
        small["deficit"] = 6
        deficits_by_topic = defaultdict(list)
        deficits_by_topic[small["topic"]] = [small]
        topics_order = [small["topic"]]
        print(f"  Smoke: topic={small['topic']} {small['political']}-{small['stance']} deficit=6 concurrency={concurrency}")

    expected_per_topic = {
        t: sum(d["deficit"] for d in deficits_by_topic[t])
        for t in topics_order
    }

    # Resume: load what's already on disk.
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

    # Build remaining task list, resuming each deficit from where it left off.
    tasks = []
    for topic_full in topics_order:
        for deficit_item in deficits_by_topic[topic_full]:
            needed = deficit_item["deficit"]
            key = (deficit_item["topic"], deficit_item["political"], deficit_item["stance"])
            done = done_counts.get(key, 0)
            for iteration in range(done, needed):
                tasks.append((topic_full, deficit_item, iteration))

    total_expected = sum(expected_per_topic.values())
    print(f"\nTotal topics: {len(topics_order)}")
    print(f"Total expected generations: {total_expected}")
    print(f"Already completed (resumed): {already_done}")
    print(f"Remaining to generate: {len(tasks)}")
    print(f"Concurrency: {concurrency}")

    # Any topic already complete on disk but not yet converted -> convert now.
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

    # Suppress the very verbose per-node print() output once, around the whole
    # executor block. redirect_stdout is process-global, so it is set a single
    # time here (never toggled per-thread). tqdm/progress go to stderr.
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

                # Durable append (main thread only -> no lock needed).
                _append_result(_progress_path(topic_full, smoke), result)
                results_by_topic[topic_full].append(result)
                done_per_topic[topic_full] += 1
                content_type_counter[result.get("content_type", "")] += 1
                angle_counter[result.get("angle", "")] += 1
                total_generated += 1

                # Eager convert when a topic reaches its target.
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

    # Final sweep: convert any topic with results that wasn't saved eagerly.
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

    if smoke:
        _smoke_schema_check(results_by_topic)

    return total_generated


def _smoke_schema_check(results_by_topic) -> None:
    print("\n" + "=" * 80, file=sys.stderr)
    print("SMOKE SCHEMA CHECK", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    sample = None
    for rs in results_by_topic.values():
        if rs:
            sample = rs[0]
            break
    if sample is None:
        print("  FAIL: no results generated.", file=sys.stderr)
        return
    gen_keys = set(sample.keys())
    print(f"  Generated keys ({len(gen_keys)}): {sorted(gen_keys)}", file=sys.stderr)
    if os.path.exists(ORIGINAL_SAMPLE):
        try:
            with open(ORIGINAL_SAMPLE, "r", encoding="utf-8") as f:
                ref = json.load(f)
            if ref:
                ref_keys = set(ref[0].keys())
                print(f"  Reference keys ({len(ref_keys)}): {sorted(ref_keys)}", file=sys.stderr)
                if gen_keys == ref_keys:
                    print("  SCHEMA MATCH ✓", file=sys.stderr)
                else:
                    print(f"  SCHEMA MISMATCH: missing={ref_keys - gen_keys} extra={gen_keys - ref_keys}", file=sys.stderr)
        except Exception as e:
            print(f"  Could not load reference: {e}", file=sys.stderr)
    else:
        print(f"  (reference {ORIGINAL_SAMPLE} not found; skipping comparison)", file=sys.stderr)


def run_all_topics_sequential():
    """Original sequential implementation."""
    start_total = time.time()

    print("=" * 80)
    print("RUN ALL TOPICS SEQUENTIAL (with progress)")
    print("=" * 80)

    pipeline = RAGPipeline()

    state: PipelineState = PipelineState()
    state.update(pipeline.load_dataset_node(state))
    state.update(pipeline.analyze_distribution_node(state))

    deficits = state["deficits"]
    topic_base_map = state.get("topic_base_map", {})

    topics_order = state["topics"]
    deficits_by_topic = defaultdict(list)
    for d in deficits:
        deficits_by_topic[d["topic"]].append(dict(d))

    total_generations = sum(d["deficit"] for d in deficits)
    print(f"\nTotal topics: {len(topics_order)}")
    print(f"Total deficits: {len(deficits)}")
    print(f"Total generations to run: {total_generations}")

    generation_results = []
    content_type_counter = defaultdict(int)
    angle_counter = defaultdict(int)
    total_generated = 0

    pbar_total = tqdm(
        total=total_generations,
        desc="Total generations",
        unit="gen",
        dynamic_ncols=True,
        leave=True,
    )

    for topic_full in tqdm(topics_order, desc="Topics", unit="topic", leave=True):
        topic_deficits = deficits_by_topic.get(topic_full, [])
        if not topic_deficits:
            continue

        topic_start = time.time()
        topic_gen_count = 0

        for deficit_item in topic_deficits:
            needed = deficit_item["deficit"]

            local_state: PipelineState = PipelineState()
            local_state.update({
                "dataset": state["dataset"],
                "topics": state["topics"],
                "topic_base_map": topic_base_map,
                "deficits": [deficit_item],
                "current_deficit_idx": 0,
                "generation_results": [],
                "content_type_counter": defaultdict(int),
                "style_counter": defaultdict(int),
                "angle_counter": defaultdict(int),
            })

            for iteration in range(needed):
                local_state["current_iteration"] = iteration

                s3 = pipeline.search_documents_node(local_state)
                local_state.update(s3)
                if not local_state.get("sampled_originals"):
                    pbar_total.update(1)
                    continue

                s4 = pipeline.outline_generation_node(local_state)
                local_state.update(s4)
                if not local_state.get("outline"):
                    pbar_total.update(1)
                    continue

                s5 = pipeline.content_generation_node(local_state)
                local_state.update(s5)
                if not local_state.get("generated_text"):
                    pbar_total.update(1)
                    continue

                outline = local_state["outline"]
                result = {
                    "topic": local_state["current_topic"],
                    "political": local_state["current_political"],
                    "stance": local_state["current_stance"],
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
                generation_results.append(result)
                content_type_counter[outline.get("content_type", "")] += 1
                angle_counter[outline.get("angle", "")] += 1
                total_generated += 1
                topic_gen_count += 1

                pbar_total.update(1)
                pbar_total.set_postfix(topic=topic_full[:20], total=total_generated, refresh=True)

        topic_elapsed = time.time() - topic_start
        tqdm.write(f"  Topic {topic_full}: {topic_gen_count} generated in {topic_elapsed:.1f}s")

        topic_results = [r for r in generation_results if r["topic"] == topic_full]
        _save_topic(topic_full, topic_results)

    pbar_total.close()

    elapsed_total = time.time() - start_total
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Total generated: {total_generated}")
    print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    if total_generated:
        print(f"Avg per generation: {elapsed_total/total_generated:.1f}s")
    print("\nContent type distribution:")
    for ctype, count in sorted(content_type_counter.items()):
        print(f"  {ctype}: {count}")
    print("\nAngle distribution:")
    for angle, count in sorted(angle_counter.items()):
        print(f"  {angle}: {count}")

    print(f"\nTotal results in memory: {len(generation_results)}")
    return total_generated


def _parse_args():
    parser = argparse.ArgumentParser(description="FAIR-SYNTH synthetic generation runner")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="parallel worker threads (default 16)")
    parser.add_argument("--sequential", action="store_true",
                        help="force the original sequential path")
    parser.add_argument("--smoke", action="store_true",
                        help="small parallel smoke test (separate _smoke/ output)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.sequential:
        run_all_topics_sequential()
    else:
        run_all_topics_parallel(concurrency=args.concurrency, smoke=args.smoke)
