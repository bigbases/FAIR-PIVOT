#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import TypedDict
from typing import Dict, Optional, Tuple

# Must run before `import torch`: restrict visible GPUs (Blackwell sm_120 + old PyTorch → "no kernel image").
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
if _cvd is None or not str(_cvd).strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("IFV1_CUDA_VISIBLE_DEVICES", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
    BNB_CONFIG_AVAILABLE = True
except Exception:
    BitsAndBytesConfig = None
    BNB_CONFIG_AVAILABLE = False

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


def disable_peft_bnb() -> None:
    if not PEFT_AVAILABLE:
        return
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.model as peft_lora_model

        if hasattr(peft_import_utils.is_bnb_available, "cache_clear"):
            peft_import_utils.is_bnb_available.cache_clear()
        if hasattr(peft_import_utils.is_bnb_4bit_available, "cache_clear"):
            peft_import_utils.is_bnb_4bit_available.cache_clear()
        peft_import_utils.is_bnb_available = lambda: False
        peft_import_utils.is_bnb_4bit_available = lambda: False
        peft_lora_model.is_bnb_available = lambda: False
        peft_lora_model.is_bnb_4bit_available = lambda: False
    except Exception as exc:
        print(f"[load_model] warning: failed to disable PEFT bitsandbytes hooks: {exc}")

BASE_MODEL_ID = {
    "llama3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma2-9b": "google/gemma-2-9b-it",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "phi4-mini-3.8b": "microsoft/Phi-4-mini-instruct",
    "phi4-14b": "microsoft/phi-4",
    "mistral-7b-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "olmo2-7b": "allenai/OLMo-2-1124-7B-Instruct",
}


def cleanup_hard() -> None:
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def make_bnb_config():
    if not BNB_CONFIG_AVAILABLE:
        raise RuntimeError("BitsAndBytesConfig is not available")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def prefer_bnb() -> bool:
    return os.environ.get("IFV1_DISABLE_BNB", "0") not in {"1", "true", "TRUE", "yes", "YES"}


def fallback_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def ensure_transformers_compat(backbone: str) -> None:
    # Some remote-code checkpoints in wave1 expect newer transformers symbols.
    if backbone != "phi4-mini-3.8b":
        return
    try:
        import transformers.utils as tf_utils

        if not hasattr(tf_utils, "LossKwargs"):
            class LossKwargs(TypedDict, total=False):
                pass

            tf_utils.LossKwargs = LossKwargs
            print("[load_model] patched transformers.utils.LossKwargs for phi4-mini-3.8b compatibility")
    except Exception as exc:
        print(f"[load_model] warning: failed to patch transformers compatibility helpers: {exc}")


def huggingface_hub_token():
    """Read token for gated HF models. Prefer HF_TOKEN / HUGGING_FACE_HUB_TOKEN; else True uses `huggingface-cli login` cache."""
    t = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    return t if t else True


def use_fast_tokenizer_for(backbone: str) -> bool:
    return backbone != "mistral-7b-v0.3"


def load_tokenizer(base_id: str, backbone: str, cache_dir: Optional[str] = None):
    tok = AutoTokenizer.from_pretrained(
        base_id,
        trust_remote_code=True,
        padding_side="left",
        use_fast=use_fast_tokenizer_for(backbone),
        cache_dir=cache_dir,
        token=huggingface_hub_token(),
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def adapter_files_ok(adapter_dir: Path) -> Tuple[bool, str]:
    cfg = adapter_dir / "adapter_config.json"
    if not cfg.exists():
        return False, f"missing {cfg.name}"
    candidates = [
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
        adapter_dir / "pytorch_model.bin",
    ]
    if not any(p.exists() for p in candidates):
        return False, "missing adapter weights"
    return True, "ok"


def load_model(backbone: str, adapter_dir: Optional[Path], device_id: int = 0, cache_dir: Optional[str] = None, attn_impl: str = "eager"):
    cleanup_hard()
    base_id = BASE_MODEL_ID[backbone]
    ensure_transformers_compat(backbone)

    common_kwargs = {
        "device_map": {"": device_id},
        "trust_remote_code": True,
        "attn_implementation": attn_impl,
        "cache_dir": cache_dir,
        "token": huggingface_hub_token(),
    }

    base = None
    bnb_error = None
    if prefer_bnb():
        try:
            base = AutoModelForCausalLM.from_pretrained(
                base_id,
                quantization_config=make_bnb_config(),
                **common_kwargs,
            )
            print(f"[load_model] loaded {backbone} with 4-bit bitsandbytes")
        except Exception as exc:
            bnb_error = exc
            print(f"[load_model] bitsandbytes load failed for {backbone}; retrying with dtype fallback: {exc}")

    if base is None:
        torch_dtype = fallback_dtype()
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **common_kwargs,
        )
        print(f"[load_model] loaded {backbone} without bitsandbytes using torch_dtype={torch_dtype}")

    base.eval()

    if adapter_dir is None:
        return base
    if not PEFT_AVAILABLE:
        raise RuntimeError("peft not available but adapter_dir was provided")
    ok, msg = adapter_files_ok(adapter_dir)
    if not ok:
        raise FileNotFoundError(f"Adapter not usable in {adapter_dir}: {msg}")
    if not prefer_bnb():
        disable_peft_bnb()
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    model.eval()
    return model


def resolve_condition_adapter(outputs_root: Path, backbone: str, seed: int, condition: str) -> Optional[Path]:
    if condition == "base":
        return None
    return outputs_root / backbone / f"seed_{seed}" / condition


def generate_text(model, tok, prompt: str, max_prompt_len: int = 768, max_new_tokens: int = 180, temperature: float = 0.7, top_p: float = 0.9, seed: Optional[int] = None) -> str:
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    do_sample = float(temperature) > 0.0
    generate_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": 1.1,
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    out = model.generate(
        **generate_kwargs,
    )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def save_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
