"""
multi_agents_political_only_v2.py

Proposal 1 variant: when generating Center,
show Left K/2 + Right K/2 documents with explicit [LEFT]/[RIGHT] labels,
ask LLM to acknowledge both poles and synthesize a Center position.

Left and Right targets keep counter-perspective (same as v1).

Goal: reduce Center -> Left bias (53% in v1) by giving LLM both poles
as reference, forcing it to find a true middle ground rather than
following a single (left-leaning) source pool framing.
"""

import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from multi_agents import (
    ANGLES,
    CONTENT_TYPES,
    FIXED_K,
    PROJECT_ROOT,
    PipelineState,
    RAGPipeline,
    STYLE_GUIDES,
    TARGET_AUDIENCES,
)

# Override paths (relative to PROJECT_ROOT; see FAIR_PIVOT_ROOT in multi_agents.py)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "2_rag", "output_qwen35_political_only_v2")
AGENTS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "2_rag", "agents_output_qwen35_political_only_v2")

# Stance-free query
QUERY_TEMPLATE = "Article about {topic} from {political} perspective"

# Center -> self, Left <-> Right (used by the base political-only pipeline below)
COUNTER_POLITICAL = {
    "Left": "Right",
    "Right": "Left",
    "Center": "Center",
}

# Center retrieval: Left K_HALF + Right K_HALF docs
K_HALF = 2  # 2 Left + 2 Right = total 4 docs for Center


def _get_political_from_row(row: pd.Series, ann_cols: List[str]) -> str | None:
    """Majority political label from a row (stance ignored)."""
    pol = row.get("political_major")
    if isinstance(pol, str) and pol.strip():
        return pol.strip()

    labels: List[str] = []
    for col in ann_cols:
        val = row.get(col)
        if not isinstance(val, str) or not val.strip():
            continue
        try:
            ann = json.loads(val)
        except Exception:
            continue
        pol2 = ann.get("Political", {}).get("label")
        if pol2:
            labels.append(str(pol2).strip())

    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


class PoliticalOnlyRAGPipeline(RAGPipeline):
    """political-only labels, Center self-perspective, L/R counter.

    Base pipeline (formerly in multi_agents_political_only.py). v2 subclasses this
    and overrides Center generation to use dual-pole context; Left/Right targets
    keep the counter-perspective logic defined here.
    """

    # ==================== Node 2: Analyze Distribution ====================
    def analyze_distribution_node(self, state: PipelineState) -> PipelineState:
        print("\n" + "=" * 60)
        print("[Node 2: Analyze Distribution (political-only)]")
        print("=" * 60)

        topics = state["topics"]
        topic_distributions: Dict[str, Dict[str, int]] = {}
        all_deficits: List[Dict] = []

        for topic_full in topics:
            print(f"\n[Topic: {topic_full}]")
            df = state["dataset"][topic_full]

            ann_cols = [c for c in df.columns if "gpt-4.1_" in c or "claude-sonnet" in c]

            political_counts: Counter = Counter()
            for _, row in df.iterrows():
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                pol = _get_political_from_row(row, ann_cols)
                if pol:
                    political_counts[pol] += 1

            topic_distributions[topic_full] = dict(political_counts)
            print(f"  Distribution: {dict(political_counts)}")

            if not political_counts:
                continue

            target = max(political_counts.values())
            for political in ["Left", "Center", "Right"]:
                current = political_counts.get(political, 0)
                deficit = target - current
                if deficit <= 0:
                    continue
                all_deficits.append({
                    "topic": topic_full,
                    "political": political,
                    "stance": "",
                    "current_count": current,
                    "target_count": target,
                    "deficit": deficit,
                })
                print(f"  Deficit: {political} needs +{deficit} "
                      f"(current: {current}, target: {target})")

        total_items = sum(d["deficit"] for d in all_deficits)
        print(f"\n{'=' * 60}")
        print(f"Total deficits: {len(all_deficits)}")
        print(f"Total items to generate: {total_items}")
        print(f"{'=' * 60}")

        return {
            "topic_distributions": topic_distributions,
            "deficits": all_deficits,
        }

    # ==================== Node 3: Search Documents ====================
    def search_documents_node(self, state: PipelineState) -> PipelineState:
        deficits = state["deficits"]
        current_deficit_idx = state.get("current_deficit_idx", 0)
        current_iteration = state.get("current_iteration", 0)

        if current_iteration > 0:
            current_deficit_item = deficits[current_deficit_idx]
            if current_iteration >= current_deficit_item["deficit"]:
                current_deficit_idx += 1
                current_iteration = 0

        if current_deficit_idx >= len(deficits):
            return {"current_deficit_idx": current_deficit_idx}

        deficit_item = deficits[current_deficit_idx]
        topic_full = deficit_item["topic"]
        topic_base_map = state.get("topic_base_map", {})
        base_topic = topic_base_map.get(topic_full, topic_full)
        political = deficit_item["political"]

        print("\n" + "=" * 60)
        print("[Node 3: Search Documents (political-only)]")
        print(f"Deficit {current_deficit_idx + 1}/{len(deficits)}: "
              f"{topic_full} - {political}")
        print(f"Iteration {current_iteration + 1}/{deficit_item['deficit']}")
        print("=" * 60)

        context_political = COUNTER_POLITICAL[political]
        if context_political == political:
            print(f"  Self-perspective: context = target = {political}")
        else:
            print(f"  Counter-perspective: target={political}, "
                  f"context={context_political}")

        query = QUERY_TEMPLATE.format(topic=base_topic, political=political)
        print(f"  Query: {query}")

        where_filter = {
            "$and": [
                {"topic": {"$eq": base_topic}},
                {"political_major": {"$eq": context_political}},
            ]
        }

        all_filtered = self.collection.get(
            where=where_filter,
            include=["metadatas"],
        )
        total_filtered = len(all_filtered["metadatas"]) if all_filtered["metadatas"] else 0
        print(f"  Total documents matching {context_political}: {total_filtered}")

        if total_filtered == 0:
            print("  WARNING: No documents found. Skipping.")
            return {
                "current_topic": topic_full,
                "current_political": political,
                "current_stance": "",
                "current_query": query,
                "current_deficit": deficit_item["deficit"],
                "current_deficit_idx": current_deficit_idx,
                "current_iteration": current_iteration,
                "searched_documents": [],
                "sampled_originals": [],
                "context_political": context_political,
                "context_stance": "",
            }

        n_results = min(total_filtered, 10000)
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results.get("distances") else []

        num_to_select = max(int(len(documents) * 0.5), 5)
        num_to_select = min(num_to_select, len(documents))

        doc_list = [
            {
                "chunk": doc,
                "original_text": meta.get("original_text", ""),
                "metadata": meta,
                "distance": dist,
            }
            for doc, meta, dist in zip(
                documents[:num_to_select],
                metadatas[:num_to_select],
                distances[:num_to_select] if distances else [0] * num_to_select,
            )
        ]

        print(f"  Selected top 50%: {num_to_select} documents")

        text_groups = defaultdict(list)
        for doc in doc_list:
            original_text = doc["original_text"]
            if original_text:
                text_groups[original_text].append(doc)

        unique_originals = list(text_groups.keys())
        print(f"  Unique original documents: {len(unique_originals)}")

        k = FIXED_K
        if len(unique_originals) < k:
            print(f"  WARNING: Only {len(unique_originals)} unique documents "
                  f"(requested: {k})")
            sampled_originals = unique_originals
        else:
            sampled_originals = random.sample(unique_originals, k)

        print(f"  Sampled {len(sampled_originals)} original documents (k={k})")

        return {
            "current_topic": topic_full,
            "current_political": political,
            "current_stance": "",
            "current_query": query,
            "context_political": context_political,
            "context_stance": "",
            "current_deficit": deficit_item["deficit"],
            "current_deficit_idx": current_deficit_idx,
            "current_iteration": current_iteration,
            "searched_documents": doc_list,
            "sampled_originals": sampled_originals,
        }

    # ==================== Node 4: Outline Generation ====================
    def outline_generation_node(self, state: PipelineState) -> PipelineState:
        print("\n" + "=" * 60)
        print("[Node 4: Outline Generation (political-only)]")
        print("=" * 60)

        if "current_topic" not in state:
            print("  No current topic. Skipping.")
            return {"outline": {}}

        topic = state["current_topic"]
        political = state["current_political"]
        context_political = state.get("context_political", political)
        sampled_originals = state["sampled_originals"]

        if not sampled_originals:
            print("  ERROR: No documents.")
            return {"outline": {}}

        context_docs = sampled_originals
        doc_length = random.randint(1000, 2000)
        context_text = "\n\n".join([
            f"[Reference Document {i+1}]\n{doc[:doc_length]}..."
            for i, doc in enumerate(context_docs)
        ])

        available_content_types = ", ".join(CONTENT_TYPES)
        available_angles = ", ".join(ANGLES)
        available_audiences = ", ".join(TARGET_AUDIENCES)

        is_self = (context_political == political)

        if is_self:
            system_prompt = f"""You are an experienced content strategist and analyst.

            Your task is to:
            1. Carefully read and analyze the provided reference documents
            2. Understand what arguments, perspectives, and information they present about {topic}, from a {political} perspective
            3. Determine the most appropriate content format, angle, and audience based on document characteristics
            4. Plan a creative content piece that clearly expresses {political} perspective on {topic}, drawing on and extending the reference documents

            Important guidelines:
            - The reference documents share your target {political} perspective on {topic} — use their framing, evidence, and tone as a foundation
            - Build on the arguments and narratives in the documents to create a compelling piece
            - Let the documents guide your choice of format, angle, and audience (data-heavy, personal stories, news-oriented, etc.)
            - If documents are data-heavy: consider analysis/report formats with economic or policy angles
            - If documents are personal stories: consider essay/feature formats with individual rights or cultural angles
            - If documents are news-oriented: consider news/column formats with timely angles
            - Choose the content type, angle, and audience that naturally fit the document content and a {political} voice
            - Be creative and diverse in your approach
            """
            user_prompt_intro = f"""Here are reference documents about {topic} (from a {political} perspective):

            {context_text}

            Task:
            Analyze these documents and plan content that expresses {political} perspective on {topic}, drawing on the evidence and narratives in the documents.
            """
        else:
            system_prompt = f"""You are an experienced content strategist and analyst.

            Your task is to:
            1. Carefully read and analyze the provided reference documents
            2. Understand what arguments, perspectives, and information they present about {topic}, mostly from a {context_political} perspective
            3. Determine the most appropriate content format, angle, and audience based on document characteristics
            4. Plan a creative content piece that clearly expresses {political} perspective on {topic}, responding to or reframing the reference documents

            Important guidelines:
            - Treat the reference documents as source material, not as the final stance you must defend
            - If documents support {context_political} perspective, you should critically analyze and respond from {political} perspective
            - Let the documents guide your choice of format, angle, and audience (data-heavy, personal stories, news-oriented, etc.)
            - If documents are data-heavy: consider analysis/report formats with economic or policy angles
            - If documents are personal stories: consider essay/feature formats with individual rights or cultural angles
            - If documents are news-oriented: consider news/column formats with timely angles
            - Choose the content type, angle, and audience that naturally fit the document content and your {political} response
            - Be creative and diverse in your approach
            """
            user_prompt_intro = f"""Here are reference documents about {topic} (mostly from a {context_political} perspective):

            {context_text}

            Task:
            Analyze these documents and plan content that expresses {political} perspective on {topic}, either by:
            - directly defending {political} using evidence and narratives from the documents, or
            - critically responding to and reframing the {context_political} arguments in the documents.
            """

        user_prompt = user_prompt_intro + f"""

            Step 1 - Analyze the documents:
            - What is the primary focus of these documents? (data/policy, personal stories, news events, cultural themes, etc.)
            - What evidence or arguments do they present?
            - What emotional tone or style do they have?

            Step 2 - Choose appropriate format and approach:
            Based on your analysis, select the most fitting options:

            Available content types: {available_content_types}
            Available approach angles: {available_angles}
            Available target audiences: {available_audiences}

            Step 3 - Plan the content:
            - Create a title that captures both the content and your chosen angle
            - Identify key points from the documents that you will use, support, or challenge
            - Explain your reasoning for the choices you made

            Respond in JSON format:
            {{
            "content_type": "your selected content type from the available options",
            "title": "specific and creative title",
            "angle": "your selected angle from the available options",
            "target_audience": "your selected audience from the available options",
            "key_points": ["key point 1 based on documents", "key point 2 based on documents", "key point 3 based on documents"],
            "reasoning": "explain why you chose this specific content type, angle, and audience based on what you saw in the documents (2-3 sentences)"
            }}
            """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            outline = json.loads(content)

            print(f"  Generated outline:")
            print(f"    - Type: {outline.get('content_type')}")
            print(f"    - Title: {outline.get('title', '')[:60]}...")
            if "reasoning" in outline:
                print(f"    - Reasoning: {outline.get('reasoning', '')[:100]}...")

            return {"outline": outline}

        except Exception as e:
            print(f"  ERROR: Failed to generate outline: {e}")
            return {"outline": {}}

    # ==================== Node 5: Content Generation ====================
    def content_generation_node(self, state: PipelineState) -> PipelineState:
        print("\n" + "=" * 60)
        print("[Node 5: Content Generation (political-only)]")
        print("=" * 60)

        if "current_topic" not in state:
            print("  No current topic. Skipping.")
            return {"generated_text": ""}

        topic = state["current_topic"]
        political = state["current_political"]
        context_political = state.get("context_political", political)
        sampled_originals = state["sampled_originals"]
        outline = state.get("outline", {})
        iteration = state["current_iteration"]

        if not outline or not sampled_originals:
            print("  ERROR: Missing outline or documents.")
            return {"generated_text": ""}

        is_self = (context_political == political)

        context_docs = sampled_originals
        style_counter = state.get("style_counter", defaultdict(int))
        least_used_styles = sorted(STYLE_GUIDES, key=lambda x: style_counter[x])
        forced_style = least_used_styles[iteration % len(STYLE_GUIDES)]

        varied_temp = random.uniform(0.7, 0.9)
        varied_llm = self.llm.bind(temperature=varied_temp)

        doc_length = random.randint(1500, 2500)
        context_text = "\n\n".join([
            f"[Reference Document {i+1}]\n{doc[:doc_length]}..."
            for i, doc in enumerate(context_docs)
        ])

        content_type = outline.get("content_type", "article")
        title = outline.get("title", "")
        angle = outline.get("angle", "")
        target_audience = outline.get("target_audience", "general public")
        key_points = outline.get("key_points", [])
        reasoning = outline.get("reasoning", "")

        avg_length = sum(len(doc) for doc in context_docs) / len(context_docs)
        if avg_length > 2000:
            min_length, max_length = random.randint(3000, 4000), random.randint(7000, 9000)
        elif avg_length < 1000:
            min_length, max_length = random.randint(1500, 2500), random.randint(5000, 7000)
        else:
            min_length, max_length = random.randint(2000, 3000), random.randint(6000, 8000)

        print(f"  Style: {forced_style}")
        print(f"  Temperature: {varied_temp:.2f}")
        print(f"  Target length: {min_length}-{max_length}")

        writer_roles = [
            f"{content_type} professional writer",
            f"{content_type} writer and journalist",
            f"experienced {content_type} writer",
            f"creative {content_type} writer",
            f"{content_type} expert and writer",
        ]
        dynamic_writer_role = writer_roles[iteration % len(writer_roles)]

        if is_self:
            system_prompt = (
                f"You are a {dynamic_writer_role}.\n\n"
                f"Your task is to write content that clearly expresses {political} perspective on {topic}.\n\n"
                "Important guidelines:\n"
                f"- The reference documents share the target {political} perspective; use them as a foundation\n"
                f"- Build on data, stories, and arguments from the reference documents to support {political}\n"
                "- Write original and natural content, do not copy directly\n"
                f"- {political} perspective must be clear throughout\n"
                f"- Tone: {forced_style}\n"
                f"- Target audience: {target_audience}\n"
                f"- Length: {min_length}-{max_length} characters\n"
                f"- Format must match {content_type} conventions\n"
            )
        else:
            system_prompt = (
                f"You are a {dynamic_writer_role}.\n\n"
                f"Your task is to write content that clearly expresses {political} perspective on {topic}.\n\n"
                "Important guidelines:\n"
                "- Reference documents are for context and inspiration, do not copy directly\n"
                f"- The reference documents may mostly reflect an opposite or different perspective ({context_political}); "
                f"you must respond from {political} viewpoint\n"
                f"- You can use data, stories, and arguments from the reference documents, but reinterpret or "
                f"challenge them as needed to support {political}\n"
                "- Write original and natural content\n"
                f"- {political} perspective must be clear throughout\n"
                f"- Tone: {forced_style}\n"
                f"- Target audience: {target_audience}\n"
                f"- Length: {min_length}-{max_length} characters\n"
                f"- Format must match {content_type} conventions\n"
            )

        key_points_text = "\n".join([f"  * {kp}" for kp in key_points])
        challenge_phrase = " or extending" if is_self else ", reframing, or challenging"

        user_prompt = f"""Content planning information:
        - Format: {content_type}
        - Title: {title}
        - Approach angle: {angle}
        - Target audience: {target_audience}
        - Tone: {forced_style}
        - Key points to cover:
        {key_points_text}

        Planning reasoning: {reasoning}

        Reference documents for context:
        {context_text}

        Now write a complete {content_type} that:
        1. Expresses {political} perspective on {topic}
        2. Uses the {angle} approach
        3. Incorporates the key points above (supporting{challenge_phrase} the reference documents as needed)
        4. Follows {content_type} format conventions
        5. Maintains {forced_style}
        6. Targets {target_audience}

        Write the content directly. Do not include meta-commentary or explanations about what you're doing.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = varied_llm.invoke(messages)
            generated_text = response.content.strip()

            generated_text = re.sub(r"^```[\w]*\n", "", generated_text)
            generated_text = re.sub(r"\n```$", "", generated_text)

            print(f"  Generated content length: {len(generated_text)} characters")

            style_counter[forced_style] += 1

            return {
                "generated_text": generated_text,
                "style_counter": style_counter,
            }

        except Exception as e:
            print(f"  ERROR: Failed to generate content: {e}")
            return {"generated_text": ""}


class PoliticalOnlyV2RAGPipeline(PoliticalOnlyRAGPipeline):
    """v2: Center target sees Left+Right docs (labeled). L/R targets unchanged."""

    # ==================== Node 3: Search Documents ====================
    def search_documents_node(self, state: PipelineState) -> PipelineState:
        deficits = state["deficits"]
        current_deficit_idx = state.get("current_deficit_idx", 0)
        current_iteration = state.get("current_iteration", 0)

        if current_iteration > 0:
            current_deficit_item = deficits[current_deficit_idx]
            if current_iteration >= current_deficit_item["deficit"]:
                current_deficit_idx += 1
                current_iteration = 0

        if current_deficit_idx >= len(deficits):
            return {"current_deficit_idx": current_deficit_idx}

        deficit_item = deficits[current_deficit_idx]
        topic_full = deficit_item["topic"]
        topic_base_map = state.get("topic_base_map", {})
        base_topic = topic_base_map.get(topic_full, topic_full)
        political = deficit_item["political"]

        print("\n" + "=" * 60)
        print("[Node 3: Search Documents (political-only v2)]")
        print(f"Deficit {current_deficit_idx + 1}/{len(deficits)}: "
              f"{topic_full} - {political}")
        print(f"Iteration {current_iteration + 1}/{deficit_item['deficit']}")
        print("=" * 60)

        # Center target: dual-pole retrieval
        if political == "Center":
            return self._search_center_dual_pole(
                state, deficit_item, current_deficit_idx, current_iteration,
                topic_full, base_topic,
            )

        # Left / Right: same as v1 (counter-perspective)
        return super().search_documents_node(state)

    def _search_center_dual_pole(self, state, deficit_item, current_deficit_idx,
                                  current_iteration, topic_full, base_topic):
        """For Center target: retrieve K_HALF Left + K_HALF Right docs, labeled."""
        political = "Center"
        query = QUERY_TEMPLATE.format(topic=base_topic, political=political)
        print(f"  Center dual-pole: target={political}, "
              f"context=Left x {K_HALF} + Right x {K_HALF}")
        print(f"  Query: {query}")

        def fetch_pole(pole_label: str, k: int):
            where_filter = {
                "$and": [
                    {"topic": {"$eq": base_topic}},
                    {"political_major": {"$eq": pole_label}},
                ]
            }
            all_filtered = self.collection.get(
                where=where_filter,
                include=["metadatas"],
            )
            total_filtered = len(all_filtered["metadatas"]) if all_filtered["metadatas"] else 0
            if total_filtered == 0:
                print(f"  WARNING: No {pole_label} docs found.")
                return []

            n_results = min(total_filtered, 10000)
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            num_to_select = max(int(len(documents) * 0.5), 5)
            num_to_select = min(num_to_select, len(documents))

            # Group by original_text within this pole
            text_groups = defaultdict(list)
            for doc, meta in zip(documents[:num_to_select], metadatas[:num_to_select]):
                ot = meta.get("original_text", "")
                if ot:
                    text_groups[ot].append(doc)

            unique_originals = list(text_groups.keys())
            if len(unique_originals) < k:
                print(f"  WARNING: Only {len(unique_originals)} unique {pole_label} docs "
                      f"(requested: {k})")
                sampled = unique_originals
            else:
                sampled = random.sample(unique_originals, k)
            print(f"  {pole_label} pool: {total_filtered} total, {len(unique_originals)} unique top50%, "
                  f"sampled {len(sampled)}")
            return sampled

        left_docs = fetch_pole("Left", K_HALF)
        right_docs = fetch_pole("Right", K_HALF)

        # Label them: list of (label, text) tuples; preserved order for prompt construction
        labeled_originals = [("LEFT", d) for d in left_docs] + [("RIGHT", d) for d in right_docs]
        # Shuffle so LLM doesn't see all Left first
        random.shuffle(labeled_originals)

        if not labeled_originals:
            print("  ERROR: No docs from either pole.")
            return {
                "current_topic": topic_full,
                "current_political": political,
                "current_stance": "",
                "current_query": query,
                "current_deficit": deficit_item["deficit"],
                "current_deficit_idx": current_deficit_idx,
                "current_iteration": current_iteration,
                "searched_documents": [],
                "sampled_originals": [],
                "context_political": "DualPole",
                "context_stance": "",
                "context_labels": [],
            }

        # sampled_originals = list of texts (drop labels here; labels go to state separately)
        sampled_originals = [t for _, t in labeled_originals]
        context_labels = [lbl for lbl, _ in labeled_originals]

        print(f"  Combined context: {len(sampled_originals)} docs "
              f"(labels: {context_labels})")

        return {
            "current_topic": topic_full,
            "current_political": political,
            "current_stance": "",
            "current_query": query,
            "context_political": "DualPole",
            "context_stance": "",
            "context_labels": context_labels,
            "current_deficit": deficit_item["deficit"],
            "current_deficit_idx": current_deficit_idx,
            "current_iteration": current_iteration,
            "searched_documents": [],
            "sampled_originals": sampled_originals,
        }

    # ==================== Node 4: Outline Generation ====================
    def outline_generation_node(self, state: PipelineState) -> PipelineState:
        # Non-Center: delegate to parent (v1 behavior)
        political = state.get("current_political", "")
        if political != "Center":
            return super().outline_generation_node(state)

        print("\n" + "=" * 60)
        print("[Node 4: Outline Generation (political-only v2, Center dual-pole)]")
        print("=" * 60)

        topic = state["current_topic"]
        sampled_originals = state["sampled_originals"]
        context_labels = state.get("context_labels", [])

        if not sampled_originals:
            print("  ERROR: No documents.")
            return {"outline": {}}

        doc_length = random.randint(1000, 2000)
        # Build labeled context text
        parts = []
        for i, (label, doc) in enumerate(zip(context_labels, sampled_originals)):
            parts.append(f"[Reference Document {i+1}, labeled {label}]\n{doc[:doc_length]}...")
        context_text = "\n\n".join(parts)

        available_content_types = ", ".join(CONTENT_TYPES)
        available_angles = ", ".join(ANGLES)
        available_audiences = ", ".join(TARGET_AUDIENCES)

        system_prompt = f"""You are an experienced content strategist and analyst.

        Your task is to:
        1. Carefully read the provided reference documents. Each one is labeled as LEFT or RIGHT, indicating its political perspective on {topic}.
        2. Acknowledge the arguments and concerns of BOTH poles (LEFT and RIGHT) explicitly.
        3. Synthesize a CENTER (centrist / moderate / non-partisan) position that genuinely balances both sides, rather than picking one side or producing empty neutrality.
        4. Determine the most appropriate content format, angle, and audience for this Center synthesis.

        Important guidelines:
        - The LEFT and RIGHT documents represent the two ends of the spectrum. Your Center position must engage with both, neither dismissing one side nor uncritically accepting the other.
        - A genuine Center position is NOT "I refuse to take sides" — it actively reconciles valid concerns from both poles into a moderate, evidence-based stance.
        - Avoid clichéd centrist framings like "beyond the binary" or generic "pragmatic middle ground" labels unless they emerge naturally from your specific synthesis.
        - Let the documents guide your choice of format, angle, and audience.
        - Be creative and specific; avoid generic centrist boilerplate.
        """

        user_prompt = f"""Here are reference documents on {topic}, each labeled with its political perspective:

        {context_text}

        Task:
        Plan content that expresses a CENTER position on {topic}, by:
        1. Identifying the legitimate concerns of the LEFT documents.
        2. Identifying the legitimate concerns of the RIGHT documents.
        3. Building a coherent moderate position that addresses both sets of concerns.

        Step 1 - Analyze the documents:
        - What does the LEFT side argue? What evidence/values do they invoke?
        - What does the RIGHT side argue? What evidence/values do they invoke?
        - Where do they agree (if at all)? Where do they fundamentally disagree?

        Step 2 - Choose format and approach:
        Available content types: {available_content_types}
        Available approach angles: {available_angles}
        Available target audiences: {available_audiences}

        Step 3 - Plan the content:
        - Title (specific, not generic "Beyond X" pattern)
        - Key points that engage with BOTH sides
        - Reasoning for your synthesis

        Respond in JSON format:
        {{
        "content_type": "your selected content type from the available options",
        "title": "specific and creative title (avoid 'Beyond the Binary' style clichés)",
        "angle": "your selected angle from the available options",
        "target_audience": "your selected audience from the available options",
        "key_points": ["key point engaging LEFT concern", "key point engaging RIGHT concern", "key point synthesizing both"],
        "reasoning": "explain how you balance LEFT and RIGHT into a Center position (2-3 sentences)"
        }}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            outline = json.loads(content)

            print(f"  Generated outline:")
            print(f"    - Type: {outline.get('content_type')}")
            print(f"    - Title: {outline.get('title', '')[:60]}...")
            if "reasoning" in outline:
                print(f"    - Reasoning: {outline.get('reasoning', '')[:100]}...")

            return {"outline": outline}

        except Exception as e:
            print(f"  ERROR: Failed to generate outline: {e}")
            return {"outline": {}}

    # ==================== Node 5: Content Generation ====================
    def content_generation_node(self, state: PipelineState) -> PipelineState:
        political = state.get("current_political", "")
        if political != "Center":
            return super().content_generation_node(state)

        print("\n" + "=" * 60)
        print("[Node 5: Content Generation (political-only v2, Center dual-pole)]")
        print("=" * 60)

        topic = state["current_topic"]
        sampled_originals = state["sampled_originals"]
        context_labels = state.get("context_labels", [])
        outline = state.get("outline", {})
        iteration = state["current_iteration"]

        if not outline or not sampled_originals:
            print("  ERROR: Missing outline or documents.")
            return {"generated_text": ""}

        style_counter = state.get("style_counter", defaultdict(int))
        least_used_styles = sorted(STYLE_GUIDES, key=lambda x: style_counter[x])
        forced_style = least_used_styles[iteration % len(STYLE_GUIDES)]

        varied_temp = random.uniform(0.7, 0.9)
        varied_llm = self.llm.bind(temperature=varied_temp)

        doc_length = random.randint(1500, 2500)
        parts = []
        for i, (label, doc) in enumerate(zip(context_labels, sampled_originals)):
            parts.append(f"[Reference Document {i+1}, labeled {label}]\n{doc[:doc_length]}...")
        context_text = "\n\n".join(parts)

        content_type = outline.get("content_type", "article")
        title = outline.get("title", "")
        angle = outline.get("angle", "")
        target_audience = outline.get("target_audience", "general public")
        key_points = outline.get("key_points", [])
        reasoning = outline.get("reasoning", "")

        avg_length = sum(len(doc) for doc in sampled_originals) / len(sampled_originals)
        if avg_length > 2000:
            min_length, max_length = random.randint(3000, 4000), random.randint(7000, 9000)
        elif avg_length < 1000:
            min_length, max_length = random.randint(1500, 2500), random.randint(5000, 7000)
        else:
            min_length, max_length = random.randint(2000, 3000), random.randint(6000, 8000)

        print(f"  Style: {forced_style}")
        print(f"  Temperature: {varied_temp:.2f}")
        print(f"  Target length: {min_length}-{max_length}")

        writer_roles = [
            f"{content_type} professional writer",
            f"{content_type} writer and journalist",
            f"experienced {content_type} writer",
            f"creative {content_type} writer",
            f"{content_type} expert and writer",
        ]
        dynamic_writer_role = writer_roles[iteration % len(writer_roles)]

        system_prompt = (
            f"You are a {dynamic_writer_role}.\n\n"
            f"Your task is to write content that expresses a CENTER (moderate, non-partisan) position on {topic}.\n\n"
            "Important guidelines:\n"
            "- The reference documents are labeled LEFT or RIGHT. Engage with BOTH explicitly.\n"
            "- A genuine Center position acknowledges valid concerns from each side and proposes a balanced approach.\n"
            "- Do NOT write empty neutrality (\"both sides have a point\" without substance) — show how you synthesize them.\n"
            "- Do NOT simply pick one side; if you find yourself agreeing entirely with LEFT or RIGHT, you are not writing Center.\n"
            "- Avoid clichéd centrist titles and framings (\"Beyond the Binary\", \"Pragmatic Middle Ground\", \"Center for X\") unless they fit your specific synthesis naturally.\n"
            "- Write original, natural prose. Do not copy reference documents verbatim.\n"
            f"- Tone: {forced_style}\n"
            f"- Target audience: {target_audience}\n"
            f"- Length: {min_length}-{max_length} characters\n"
            f"- Format must match {content_type} conventions\n"
        )

        key_points_text = "\n".join([f"  * {kp}" for kp in key_points])

        user_prompt = f"""Content planning information:
        - Format: {content_type}
        - Title: {title}
        - Approach angle: {angle}
        - Target audience: {target_audience}
        - Tone: {forced_style}
        - Key points to cover:
        {key_points_text}

        Planning reasoning: {reasoning}

        Reference documents (each labeled LEFT or RIGHT):
        {context_text}

        Now write a complete {content_type} that:
        1. Expresses a CENTER position on {topic}.
        2. Engages with BOTH the LEFT and the RIGHT perspectives shown in the references.
        3. Uses the {angle} approach.
        4. Incorporates the key points above (synthesizing concerns from both poles).
        5. Follows {content_type} format conventions.
        6. Maintains {forced_style}.
        7. Targets {target_audience}.

        Write the content directly. Do not include meta-commentary or explanations about what you're doing.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = varied_llm.invoke(messages)
            generated_text = response.content.strip()

            generated_text = re.sub(r"^```[\w]*\n", "", generated_text)
            generated_text = re.sub(r"\n```$", "", generated_text)

            print(f"  Generated content length: {len(generated_text)} characters")

            style_counter[forced_style] += 1

            return {
                "generated_text": generated_text,
                "style_counter": style_counter,
            }

        except Exception as e:
            print(f"  ERROR: Failed to generate content: {e}")
            return {"generated_text": ""}
