import argparse
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers.cache_utils import DynamicCache

from run import DEFAULT_MODEL


os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

NUM_SAMPLES_PER_SUBSET = 10
OUTPUT_ROOT = Path("kv_cache_substitution_runs")
SUBSETS = [
    "chart",
    "counting",
    "document",
    "general",
    "grounding",
    "math",
    "scene",
    "spatial",
    "table",
]


def load_model_module(model_name: str):
    if model_name == "HuggingFaceTB/SmolVLM2-2.2B-Instruct":
        import models.smolvlm as model_module
    elif model_name == "Qwen/Qwen3-VL-2B-Instruct":
        import models.qwen3vl as model_module
    else:
        raise ValueError(f"unknown model_name={model_name!r}")
    return model_module


def blank_image_like(image):
    return Image.new("RGB", image.size, color=(255, 255, 255))


def build_inputs(image, prompt: str, processor, model):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    return inputs


def prefill(image, prompt: str, processor, model, multimodal_masks):
    inputs = build_inputs(image, prompt, processor, model)
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    image_mask, _ = multimodal_masks(processor, input_ids[0])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_mask": image_mask,
        "past_key_values": outputs.past_key_values,
        "next_logits": outputs.logits[:, -1, :].detach().float(),
    }


def same_layout(real, blank) -> bool:
    if real["input_ids"].shape != blank["input_ids"].shape:
        return False
    if not torch.equal(real["input_ids"], blank["input_ids"]):
        return False
    if not torch.equal(real["image_mask"], blank["image_mask"]):
        return False
    real_attention = real["attention_mask"]
    blank_attention = blank["attention_mask"]
    if (real_attention is None) != (blank_attention is None):
        return False
    if real_attention is not None and not torch.equal(real_attention, blank_attention):
        return False
    return bool(real["image_mask"].sum().item())


def substitute_visual_kv(real_past, blank_past, image_mask, keep_len: int):
    real_layers = list(real_past)
    blank_layers = list(blank_past)
    if len(real_layers) != len(blank_layers):
        raise ValueError("real and blank caches have different layer counts")

    visual_positions = image_mask[:keep_len].to(real_layers[0][0].device)
    hybrid_layer_data = []
    for real_layer, blank_layer in zip(real_layers, blank_layers):
        real_k, real_v = real_layer[:2]
        blank_k, blank_v = blank_layer[:2]
        if real_k.shape != blank_k.shape or real_v.shape != blank_v.shape:
            raise ValueError("real and blank cache tensors have different shapes")
        if real_k.shape[-2] < keep_len or real_v.shape[-2] < keep_len:
            raise ValueError("cache is shorter than the prompt prefix")

        hybrid_k = real_k[..., :keep_len, :].clone()
        hybrid_v = real_v[..., :keep_len, :].clone()
        hybrid_k[..., visual_positions, :] = blank_k[..., :keep_len, :][
            ..., visual_positions, :
        ]
        hybrid_v[..., visual_positions, :] = blank_v[..., :keep_len, :][
            ..., visual_positions, :
        ]
        hybrid_layer_data.append((hybrid_k, hybrid_v))
    return DynamicCache(hybrid_layer_data)


def decode_position_ids_for_last_token(model, attention_mask):
    qwen3_model = getattr(model, "model", None)
    rope_deltas = getattr(qwen3_model, "rope_deltas", None)
    if rope_deltas is None or attention_mask is None:
        return None

    batch_size = attention_mask.shape[0]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    position_ids = position_ids[:, -1:].view(1, batch_size, 1).repeat(3, 1, 1)
    rope_deltas = rope_deltas.repeat_interleave(
        batch_size // rope_deltas.shape[0],
        dim=0,
    )
    return position_ids.to(model.device) + rope_deltas.to(model.device)


def decode_one_step_from_prefix(model, input_ids, attention_mask, past_key_values):
    prompt_len = input_ids.shape[1]
    decode_kwargs = {
        "input_ids": input_ids[:, -1:],
        "past_key_values": past_key_values,
        "use_cache": False,
    }
    if attention_mask is not None:
        decode_kwargs["attention_mask"] = attention_mask
    position_ids = decode_position_ids_for_last_token(model, attention_mask)
    if position_ids is not None:
        decode_kwargs["position_ids"] = position_ids
    try:
        if position_ids is None:
            decode_kwargs["cache_position"] = torch.arange(
                prompt_len - 1,
                prompt_len,
                device=input_ids.device,
            )
        with torch.inference_mode():
            outputs = model(**decode_kwargs)
    except TypeError:
        decode_kwargs.pop("cache_position", None)
        with torch.inference_mode():
            outputs = model(**decode_kwargs)
    return outputs.logits[:, -1, :].detach().float()


def kl(p_logits, q_logits) -> float:
    logp = F.log_softmax(p_logits, dim=-1)
    logq = F.log_softmax(q_logits, dim=-1)
    return float((logp.exp() * (logp - logq)).sum(dim=-1).item())


def token_metrics(real_logits, blank_logits, hybrid_logits) -> dict:
    hybrid_logp = F.log_softmax(hybrid_logits, dim=-1)
    real_greedy = int(real_logits.argmax(dim=-1).item())
    blank_greedy = int(blank_logits.argmax(dim=-1).item())
    hybrid_greedy = int(hybrid_logits.argmax(dim=-1).item())

    real_token_logit = hybrid_logits[:, real_greedy]
    blank_token_logit = hybrid_logits[:, blank_greedy]
    return {
        "real_greedy_token_id": real_greedy,
        "blank_greedy_token_id": blank_greedy,
        "hybrid_greedy_token_id": hybrid_greedy,
        "hybrid_agrees_with_real": int(hybrid_greedy == real_greedy),
        "hybrid_agrees_with_blank": int(hybrid_greedy == blank_greedy),
        "real_greedy_rank_under_hybrid": int(
            (hybrid_logits > real_token_logit[:, None]).sum(dim=-1).item() + 1
        ),
        "blank_greedy_rank_under_hybrid": int(
            (hybrid_logits > blank_token_logit[:, None]).sum(dim=-1).item() + 1
        ),
        "real_greedy_prob_under_hybrid": float(
            hybrid_logp[:, real_greedy].exp().item()
        ),
        "blank_greedy_prob_under_hybrid": float(
            hybrid_logp[:, blank_greedy].exp().item()
        ),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}

    metric_names = [
        "kl_real_blank",
        "kl_real_hybrid",
        "kl_blank_hybrid",
        "fraction_of_blank_shift",
        "hybrid_agrees_with_real",
        "hybrid_agrees_with_blank",
        "real_greedy_rank_under_hybrid",
        "blank_greedy_rank_under_hybrid",
        "real_greedy_prob_under_hybrid",
        "blank_greedy_prob_under_hybrid",
    ]
    out = {"n": len(rows)}
    for name in metric_names:
        values = [row[name] for row in rows if row[name] is not None]
        if values:
            out[f"mean_{name}"] = sum(values) / len(values)
    return out


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run(model_name: str, run_id: str) -> Path:
    from datasets import load_dataset

    model_module = load_model_module(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = model_module.load_model(device)
    model.eval()

    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "kv_cache_substitution_config.json",
        {
            "run_id": run_id,
            "model": model_name,
            "subsets": SUBSETS,
            "num_samples_per_subset": NUM_SAMPLES_PER_SUBSET,
            "device": str(device),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
    )

    rows = []
    skipped = []
    for subset in SUBSETS:
        ds = load_dataset("DatologyAI/DatBench", subset, split="test")
        subset_ds = ds.select(range(min(NUM_SAMPLES_PER_SUBSET, len(ds))))
        for sample in tqdm(subset_ds, desc=subset):
            prompt_format = sample["prompt_format"]
            prompt = prompt_format["prefix"] + sample["question"] + prompt_format["suffix"]

            try:
                real = prefill(
                    sample["image"],
                    prompt,
                    processor,
                    model,
                    model_module.multimodal_masks,
                )
                blank = prefill(
                    blank_image_like(sample["image"]),
                    prompt,
                    processor,
                    model,
                    model_module.multimodal_masks,
                )
                if not same_layout(real, blank):
                    skipped.append(
                        {
                            "subset": subset,
                            "sample_id": sample["id"],
                            "reason": "real/blank prompt layouts differ",
                        }
                    )
                    continue

                prompt_len = real["input_ids"].shape[1]
                hybrid_past = substitute_visual_kv(
                    real["past_key_values"],
                    blank["past_key_values"],
                    real["image_mask"],
                    keep_len=prompt_len - 1,
                )
                hybrid_logits = decode_one_step_from_prefix(
                    model,
                    real["input_ids"],
                    real["attention_mask"],
                    hybrid_past,
                )

                real_logits = real["next_logits"]
                blank_logits = blank["next_logits"]
                kl_real_blank = kl(real_logits, blank_logits)
                kl_real_hybrid = kl(real_logits, hybrid_logits)
                blank_shift_denom = kl_real_blank if kl_real_blank > 1e-12 else None
                row = {
                    "subset": subset,
                    "sample_id": sample["id"],
                    "prompt_image_token_count": int(real["image_mask"].sum().item()),
                    "prompt_length": int(prompt_len),
                    "kl_real_blank": kl_real_blank,
                    "kl_real_hybrid": kl_real_hybrid,
                    "kl_blank_hybrid": kl(blank_logits, hybrid_logits),
                    "fraction_of_blank_shift": None
                    if blank_shift_denom is None
                    else kl_real_hybrid / blank_shift_denom,
                    **token_metrics(real_logits, blank_logits, hybrid_logits),
                }
                rows.append(row)
            except Exception as exc:
                skipped.append(
                    {
                        "subset": subset,
                        "sample_id": sample["id"],
                        "reason": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    write_jsonl(run_dir / "kv_cache_substitution.jsonl", rows)
    write_jsonl(run_dir / "kv_cache_substitution_skipped.jsonl", skipped)
    write_json(
        run_dir / "kv_cache_substitution_summary.json",
        {
            "overall": summarize(rows),
            "by_subset": {
                subset: summarize([row for row in rows if row["subset"] == subset])
                for subset in SUBSETS
            },
            "skipped": len(skipped),
        },
    )
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--run-id",
        default=f"kv_cache_substitution_{uuid.uuid4().hex[:8]}",
    )
    args = parser.parse_args()
    run_dir = run(args.model, args.run_id)
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
