"""Smoke test for Qwen3 replacement.

Runs the real RAGPipeline functions on the first topic / first deficit for
N_ITER iterations. Writes results to OUTPUT_DIR / AGENTS_OUTPUT_DIR (which
are now output_qwen3/ and agents_output_qwen3/ — original gpt-oss-20b
outputs untouched). Then compares the output schema against an existing
gpt-oss-20b JSON to confirm bit-identical keys.
"""
import json
import os
import sys

_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _AGENTS_DIR not in sys.path:
    sys.path.insert(0, _AGENTS_DIR)

from multi_agents import (  # noqa: E402
    AGENTS_OUTPUT_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    PipelineState,
    RAGPipeline,
)

N_ITER = 2

ORIGINAL_SAMPLE = os.path.join(PROJECT_ROOT, "2_rag", "output", "generated_additional_LGBTQ.json")


def main():
    pipeline = RAGPipeline()
    state: PipelineState = PipelineState()
    state.update(pipeline.load_dataset_node(state))
    state.update(pipeline.analyze_distribution_node(state))

    deficits = state["deficits"]
    if not deficits:
        print("No deficits found; nothing to smoke-test.")
        return 1

    first = deficits[0]
    print(f"\nSmoke test target: {first['topic']} / {first['political']}-{first['stance']}")
    print(f"Will run {N_ITER} iterations (original deficit={first['deficit']}).\n")

    state["deficits"] = [dict(first, deficit=N_ITER)]
    state["current_deficit_idx"] = 0

    results = []
    for i in range(N_ITER):
        state["current_iteration"] = i
        state.update(pipeline.search_documents_node(state))
        if not state.get("sampled_originals"):
            print(f"[iter {i}] no sampled_originals; skip")
            continue
        state.update(pipeline.outline_generation_node(state))
        if not state.get("outline"):
            print(f"[iter {i}] no outline; skip")
            continue
        state.update(pipeline.content_generation_node(state))
        if not state.get("generated_text"):
            print(f"[iter {i}] no generated_text; skip")
            continue

        outline = state["outline"]
        result = {
            "topic": state["current_topic"],
            "political": state["current_political"],
            "stance": state["current_stance"],
            "text": state["generated_text"],
            "query": state.get("current_query", ""),
            "retrieved_contexts": list(state.get("sampled_originals", [])),
            "content_type": outline.get("content_type", ""),
            "title": outline.get("title", ""),
            "angle": outline.get("angle", ""),
            "target_audience": outline.get("target_audience", ""),
            "key_points": outline.get("key_points", []),
            "reasoning": outline.get("reasoning", ""),
            "num_context_docs": len(state.get("sampled_originals", [])),
        }
        results.append(result)
        print(f"[iter {i}] OK — text len={len(result['text'])} chars, title={result['title'][:60]!r}")

    if not results:
        print("\nNo successful generations.")
        return 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(AGENTS_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"smoke_{first['topic']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(results)} results to: {out_path}")

    print("\n=== Schema comparison vs original gpt-oss-20b output ===")
    if os.path.exists(ORIGINAL_SAMPLE):
        with open(ORIGINAL_SAMPLE) as f:
            orig = json.load(f)
        orig_keys = set(orig[0].keys())
        new_keys = set(results[0].keys())
        print(f"Original sample: {ORIGINAL_SAMPLE}")
        print(f"Original keys ({len(orig_keys)}): {sorted(orig_keys)}")
        print(f"New keys      ({len(new_keys)}): {sorted(new_keys)}")
        if orig_keys == new_keys:
            print("\nSCHEMA MATCH: bit-identical keys.")
        else:
            print(f"\nSCHEMA MISMATCH")
            print(f"  Only in original: {orig_keys - new_keys}")
            print(f"  Only in new:      {new_keys - orig_keys}")
    else:
        print(f"Original sample not found at {ORIGINAL_SAMPLE}; skipping comparison.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
