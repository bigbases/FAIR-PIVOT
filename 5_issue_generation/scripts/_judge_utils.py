#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl_tree(root: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("seed*/*/*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def write_csv(rows: Sequence[Dict], path: Path, fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fields = list(fieldnames or [])
    for row in rows:
        for key in row.keys():
            if key not in resolved_fields:
                resolved_fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=resolved_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in resolved_fields})


def make_row_key(row: Dict, key_fields: Sequence[str] = ("seed", "backbone", "condition", "item_id")) -> tuple:
    return tuple(row.get(field) for field in key_fields)


def parse_json_object_from_text(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty text cannot be parsed as JSON")

    # Handle markdown fenced JSON responses, e.g. ```json ... ```
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"Could not parse JSON object from text: {stripped[:240]}")


def maybe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def maybe_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _openai_model_name(model: Optional[str]) -> str:
    return model or os.environ.get("OPENAI_JUDGE_MODEL") or os.environ.get("JUDGE_MODEL", "gpt-4.1-mini")


def openai_chat_json(system_prompt: str, user_prompt: str, model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 400) -> Dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required for judge scripts")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model_name = _openai_model_name(model)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=120)
    if resp.status_code >= 300:
        raise RuntimeError(f"Judge API failed: {resp.status_code} {resp.text}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return parse_json_object_from_text(content)


def _anthropic_model_name(model: Optional[str]) -> str:
    return model or os.environ.get("ANTHROPIC_JUDGE_MODEL") or os.environ.get("JUDGE_MODEL", "claude-3-5-haiku-latest")


def anthropic_chat_json(system_prompt: str, user_prompt: str, model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 400) -> Dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is required for Anthropic judge scripts")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/")
    model_name = _anthropic_model_name(model)

    headers = {
        "x-api-key": api_key,
        "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
        "content-type": "application/json",
    }
    payload = {
        "model": model_name,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(f"{base_url}/messages", headers=headers, json=payload, timeout=120)
    if resp.status_code >= 300:
        raise RuntimeError(f"Anthropic judge API failed: {resp.status_code} {resp.text}")
    data = resp.json()
    content_blocks = data.get("content", [])
    text = "".join(block.get("text", "") for block in content_blocks if isinstance(block, dict))
    return parse_json_object_from_text(text)


def judge_chat_json(provider: str, system_prompt: str, user_prompt: str, model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 400) -> Dict:
    if provider == "openai":
        return openai_chat_json(system_prompt, user_prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    if provider == "anthropic":
        return anthropic_chat_json(system_prompt, user_prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    raise ValueError(f"Unsupported judge provider: {provider}")


def sleep_if_needed(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)
