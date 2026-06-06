"""
subsample_v2_datasets.py

From the v2 political-only combined dataset (agent_balanced.jsonl, ~30,328 rows
with political-only prefix), build three 5,000-row datasets that preserve the
source distribution while scaling down:

  - balanced:      keep (topic x political) cell proportions
  - left_extreme:  Left-only, keep topic proportions
  - right_extreme: Right-only, keep topic proportions

Each output is exactly TARGET rows (largest-remainder method), {"text": ...} only.
Labels/topics are parsed from the prefix text (no separate metadata needed).
"""

import json
import os
import re
import random
from collections import Counter, defaultdict

SRC = "finetuning_data_qwen35_political_only_v2/agent_balanced.jsonl"
TARGET = 5000
SEED = 42

POL_MAP = {"liberal": "Left", "conservative": "Right", "centrist": "Center", "undecided": "Undecided"}

OUTPUTS = {
    "balanced":      "finetuning_data_v2_balanced/balanced.jsonl",
    "left_extreme":  "finetuning_data_v2_left_extreme/left_extreme.jsonl",
    "right_extreme": "finetuning_data_v2_right_extreme/right_extreme.jsonl",
}


def norm_topic(t: str) -> str:
    return t.replace("_", " ").replace("-", " ").strip().lower()


def parse(text: str):
    pm = re.search(r"politically (liberal|conservative|centrist|undecided)", text[:200])
    tm = re.search(r"This text discusses (.+?) from a politically", text[:200])
    if not pm or not tm:
        return None, None
    return norm_topic(tm.group(1)), POL_MAP[pm.group(1)]


def largest_remainder(counts: dict, target: int) -> dict:
    """Allocate `target` across keys proportional to counts; sum == target exactly."""
    tot = sum(counts.values())
    raw = {k: v / tot * target for k, v in counts.items()}
    floor = {k: int(v) for k, v in raw.items()}
    rem = target - sum(floor.values())
    order = sorted(raw, key=lambda k: raw[k] - floor[k], reverse=True)
    for k in order[:rem]:
        floor[k] += 1
    return floor


def main():
    rng = random.Random(SEED)

    rows = []  # (topic, political, text)
    with open(SRC, "r", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            topic, pol = parse(text)
            if topic and pol:
                rows.append((topic, pol, text))

    total = len(rows)
    print(f"Loaded {total} parsed rows from {SRC}")

    by_topic_pol = defaultdict(list)   # (topic, pol) -> [text]
    for topic, pol, text in rows:
        by_topic_pol[(topic, pol)].append(text)

    # ---- balanced: (topic x political) proportional ----
    cell_counts = {k: len(v) for k, v in by_topic_pol.items()}
    alloc = largest_remainder(cell_counts, TARGET)
    balanced = []
    for cell, n in alloc.items():
        pool = by_topic_pol[cell]
        balanced.extend(rng.sample(pool, min(n, len(pool))))
    rng.shuffle(balanced)

    # ---- extremes: single political, topic proportional ----
    def build_extreme(target_pol: str):
        topic_pool = {t: texts for (t, p), texts in by_topic_pol.items() if p == target_pol}
        topic_counts = {t: len(v) for t, v in topic_pool.items()}
        alloc_t = largest_remainder(topic_counts, TARGET)
        out = []
        for topic, n in alloc_t.items():
            pool = topic_pool[topic]
            out.extend(rng.sample(pool, min(n, len(pool))))
        rng.shuffle(out)
        return out

    datasets = {
        "balanced": balanced,
        "left_extreme": build_extreme("Left"),
        "right_extreme": build_extreme("Right"),
    }

    for name, data in datasets.items():
        out_path = OUTPUTS[name]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for text in data:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

        pol_c = Counter()
        topic_c = Counter()
        for text in data:
            t, p = parse(text)
            pol_c[p] += 1
            topic_c[t] += 1
        print(f"\n=== {name}: {len(data)} rows -> {out_path} ===")
        print(f"  political: {dict(pol_c)}")
        print(f"  topics ({len(topic_c)}): {dict(sorted(topic_c.items()))}")


if __name__ == "__main__":
    main()
