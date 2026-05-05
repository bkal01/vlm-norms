import argparse
import csv
import json
import os
import resource
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.interventions import ScaledIntervention
from run import DEFAULT_ALPHAS


os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ALPHAS = DEFAULT_ALPHAS.copy()
BASELINE_ALPHA = 1.0
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
NUM_SAMPLES_PER_SUBSET = 10
OUTPUT_ROOT = Path("kv_stability_under_alpha_runs")
SUBSETS = [
    "chart",
    "counting",
    "document",
    "general",
    "grounding",
    "scene",
    "spatial",
    "table",
]


def memory_summary() -> str:
    parts = [f"rss_gb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024:.2f}"]
    if torch.cuda.is_available():
        parts.extend(
            [
                f"cuda_alloc_gb={torch.cuda.memory_allocated() / 1024**3:.2f}",
                f"cuda_reserved_gb={torch.cuda.memory_reserved() / 1024**3:.2f}",
            ]
        )
    return " ".join(parts)


def log_step(message: str) -> None:
    print(f"[kv-stability] {time.strftime('%Y-%m-%d %H:%M:%S')} {message} {memory_summary()}", file=sys.stderr, flush=True)


def load_model_module(model_name: str):
    if model_name == "Qwen/Qwen3-VL-2B-Instruct":
        import models.qwen3vl as model_module
    elif model_name == "HuggingFaceTB/SmolVLM2-2.2B-Instruct":
        import models.smolvlm as model_module
    else:
        raise ValueError(f"unknown model_name={model_name!r}")
    return model_module


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
    layer_inputs = []
    hooks = []
    layers = language_layers(model)
    for layer in layers:
        hooks.append(
            layer.register_forward_pre_hook(
                lambda module, args: layer_inputs.append(args[0].detach())
            )
        )

    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    input_ids = inputs["input_ids"][0]
    image_mask, _ = multimodal_masks(processor, input_ids)
    if len(layer_inputs) != len(layers):
        raise ValueError("did not capture one input hidden state per layer")
    hidden_states = torch.stack(layer_inputs, dim=0).squeeze(1)
    if not bool(image_mask.sum().item()):
        raise ValueError("prompt has no image tokens")
    return {
        "input_ids": input_ids.detach().cpu(),
        "image_mask": image_mask.detach().cpu(),
        "hidden_states": hidden_states,
        "prompt_length": int(input_ids.shape[0]),
        "prompt_image_token_count": int(image_mask.sum().item()),
    }


def language_layers(model):
    model_body = getattr(model, "model", None)
    if model_body is None:
        raise ValueError("model has no .model body")

    language_model = getattr(model_body, "language_model", None)
    if language_model is not None and hasattr(language_model, "layers"):
        return language_model.layers

    text_model = getattr(model_body, "text_model", None)
    if text_model is not None and hasattr(text_model, "layers"):
        return text_model.layers

    raise ValueError("could not locate language decoder layers")


def visual_kv_for_layer(model, hidden_states: torch.Tensor, image_mask: torch.Tensor, layer: int):
    decoder_layer = language_layers(model)[layer]
    visual_h = hidden_states[image_mask.to(hidden_states.device)]
    normed = decoder_layer.input_layernorm(visual_h)
    attn = decoder_layer.self_attn
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None:
        num_heads = getattr(attn, "num_heads", None) or getattr(attn, "num_attention_heads", None)
        if num_heads is None:
            raise ValueError("could not infer attention head dimension")
        head_dim = attn.hidden_size // num_heads

    k = attn.k_proj(normed).view(normed.shape[0], -1, head_dim)
    if hasattr(attn, "k_norm"):
        k = attn.k_norm(k)
    v = attn.v_proj(normed).view(normed.shape[0], -1, head_dim)
    if hasattr(attn, "v_norm"):
        v = attn.v_norm(v)
    return k.float(), v.float(), visual_h.float()


def mean_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=-1).mean().item())


def mean_rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(a - b, dim=-1)
    denominator = torch.linalg.vector_norm(b, dim=-1).clamp_min(1e-8)
    return float((numerator / denominator).mean().item())


def hidden_mean_rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(a - b, dim=-1)
    denominator = torch.linalg.vector_norm(b, dim=-1).clamp_min(1e-8)
    return float((numerator / denominator).mean().item())


def finite_mean(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def finite_median(values: list[float]) -> float | None:
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None and y is not None
    ]
    if len(pairs) < 2:
        return None
    x = torch.tensor([pair[0] for pair in pairs])
    y = torch.tensor([pair[1] for pair in pairs])
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom) == 0.0:
        return None
    return float((x * y).sum().item() / denom.item())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_decode_run(model_name: str) -> Path | None:
    candidates = []
    for config_path in Path("runs").glob("*/config.json"):
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            continue
        run_dir = config_path.parent
        required = [
            run_dir / "answers.jsonl",
            run_dir / "attention_divergence_from_baseline.jsonl",
            run_dir / "logit_sensitivity.jsonl",
        ]
        if config.get("model") == model_name and all(path.exists() for path in required):
            candidates.append(run_dir)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def selected_sample_ids(decode_run_dir: Path | None) -> dict[str, list[str]]:
    if decode_run_dir is None:
        return {}

    selected = defaultdict(list)
    seen = set()
    with (decode_run_dir / "answers.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("condition") != "real_alpha_1":
                continue
            key = (row["subset"], row["sample_id"])
            if key in seen:
                continue
            if len(selected[row["subset"]]) >= NUM_SAMPLES_PER_SUBSET:
                continue
            seen.add(key)
            selected[row["subset"]].append(row["sample_id"])
    return dict(selected)


def load_decode_metrics(decode_run_dir: Path | None) -> tuple[dict, dict]:
    if decode_run_dir is None:
        return {}, {}

    attention_values = defaultdict(lambda: defaultdict(list))
    with (decode_run_dir / "attention_divergence_from_baseline.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            key = (row["subset"], row["sample_id"], float(row["alpha"]), int(row["layer"]))
            attention_values[key]["decode_attention_kl_mean"].append(float(row["kl_mean"]))
            attention_values[key]["decode_attention_cosine_mean"].append(float(row["cosine_mean"]))

    attention = {}
    for key, values in attention_values.items():
        attention[key] = {name: finite_mean(items) for name, items in values.items()}

    logit_values = defaultdict(lambda: defaultdict(list))
    with (decode_run_dir / "logit_sensitivity.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            key = (row["subset"], row["sample_id"], float(row["alpha"]))
            logit_values[key]["logit_kl_mean"].append(float(row["kl"]))
            logit_values[key]["baseline_token_prob_mean"].append(float(row["baseline_token_prob"]))
            logit_values[key]["baseline_token_rank_median"].append(float(row["baseline_token_rank"]))
            logit_values[key]["greedy_agreement_mean"].append(float(row["greedy_agreement"]))

    logit = {}
    for key, values in logit_values.items():
        logit[key] = {
            "logit_kl_mean": finite_mean(values["logit_kl_mean"]),
            "baseline_token_prob_mean": finite_mean(values["baseline_token_prob_mean"]),
            "baseline_token_rank_median": finite_median(values["baseline_token_rank_median"]),
            "greedy_agreement_mean": finite_mean(values["greedy_agreement_mean"]),
        }

    return attention, logit


def load_samples(selected: dict[str, list[str]]):
    from datasets import load_dataset

    if not selected:
        for subset in SUBSETS:
            ds = load_dataset("DatologyAI/DatBench", subset, split="test")
            n = min(NUM_SAMPLES_PER_SUBSET, len(ds))
            yield subset, ds.select(range(n))
        return

    for subset, sample_ids in selected.items():
        log_step(f"loading dataset subset={subset} selected_ids={len(sample_ids)}")
        ds = load_dataset("DatologyAI/DatBench", subset, split="test")
        id_to_index = {sample_id: index for index, sample_id in enumerate(ds["id"])}
        indices = [
            id_to_index[sample_id]
            for sample_id in sample_ids
            if sample_id in id_to_index
        ]
        rows = ds.select(indices)
        log_step(f"loaded dataset subset={subset} selected_rows={len(rows)}")
        yield subset, rows


def compute_rows_for_sample(model, model_name: str, subset: str, sample: dict, prefilled_by_alpha: dict):
    baseline = prefilled_by_alpha[BASELINE_ALPHA]
    image_mask = baseline["image_mask"]
    layers = language_layers(model)

    if any(not torch.equal(item["input_ids"], baseline["input_ids"]) for item in prefilled_by_alpha.values()):
        raise ValueError("input token layout changed across alphas")
    if any(not torch.equal(item["image_mask"], image_mask) for item in prefilled_by_alpha.values()):
        raise ValueError("image token mask changed across alphas")

    projected = {}
    for alpha, item in prefilled_by_alpha.items():
        projected[alpha] = []
        for layer in range(len(layers)):
            h_in = item["hidden_states"][layer]
            projected[alpha].append(visual_kv_for_layer(model, h_in, image_mask, layer))

    rows = []
    for alpha in ALPHAS:
        layer0_k, layer0_v, layer0_h = projected[alpha][0]
        prev_k = prev_v = prev_h = None
        for layer, (k, v, h) in enumerate(projected[alpha]):
            baseline_k, baseline_v, baseline_h = projected[BASELINE_ALPHA][layer]
            row = {
                "model": model_name,
                "subset": subset,
                "sample_id": sample["id"],
                "alpha": alpha,
                "layer": layer,
                "prompt_length": baseline["prompt_length"],
                "prompt_image_token_count": baseline["prompt_image_token_count"],
                "k_cos_to_layer0": mean_cos(k, layer0_k),
                "v_cos_to_layer0": mean_cos(v, layer0_v),
                "k_adjacent_cos": None if prev_k is None else mean_cos(k, prev_k),
                "v_adjacent_cos": None if prev_v is None else mean_cos(v, prev_v),
                "k_rel_update": None if prev_k is None else mean_rel_diff(k, prev_k),
                "v_rel_update": None if prev_v is None else mean_rel_diff(v, prev_v),
                "k_cos_to_baseline": mean_cos(k, baseline_k),
                "v_cos_to_baseline": mean_cos(v, baseline_v),
                "k_rel_diff_from_baseline": mean_rel_diff(k, baseline_k),
                "v_rel_diff_from_baseline": mean_rel_diff(v, baseline_v),
                "vision_hidden_cos_to_layer0": mean_cos(h, layer0_h),
                "vision_hidden_adjacent_cos": None if prev_h is None else mean_cos(h, prev_h),
                "vision_hidden_rel_update": None if prev_h is None else hidden_mean_rel_diff(h, prev_h),
                "vision_hidden_cos_to_baseline": mean_cos(h, baseline_h),
                "vision_hidden_rel_diff_from_baseline": hidden_mean_rel_diff(h, baseline_h),
            }
            rows.append(row)
            prev_k, prev_v, prev_h = k, v, h
    return rows


def summarize_by(keys: list[str], rows: list[dict], metric_names: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)

    out = []
    for group_key, group_rows in sorted(grouped.items()):
        row = dict(zip(keys, group_key))
        row["n_points"] = len(group_rows)
        for name in metric_names:
            row[f"mean_{name}"] = finite_mean([item.get(name) for item in group_rows])
        out.append(row)
    return out


def correlation_summary(rows: list[dict]) -> list[dict]:
    kv_metrics = [
        "k_cos_to_layer0",
        "v_cos_to_layer0",
        "k_rel_update",
        "v_rel_update",
        "k_cos_to_baseline",
        "v_cos_to_baseline",
        "k_rel_diff_from_baseline",
        "v_rel_diff_from_baseline",
        "vision_hidden_cos_to_layer0",
        "vision_hidden_rel_update",
        "vision_hidden_rel_diff_from_baseline",
    ]
    decode_metrics = [
        "decode_attention_kl_mean",
        "logit_kl_mean",
        "baseline_token_rank_median",
        "greedy_agreement_mean",
    ]
    out = []
    nonbaseline_rows = [row for row in rows if row["alpha"] != BASELINE_ALPHA]
    for kv_metric in kv_metrics:
        for decode_metric in decode_metrics:
            paired = [
                row
                for row in nonbaseline_rows
                if row.get(kv_metric) is not None and row.get(decode_metric) is not None
            ]
            out.append(
                {
                    "kv_metric": kv_metric,
                    "decode_metric": decode_metric,
                    "pearson": pearson(
                        [row[kv_metric] for row in paired],
                        [row[decode_metric] for row in paired],
                    ),
                    "n_points": len(paired),
                }
            )
    return out


def run(model_name: str) -> Path:
    log_step(f"starting model={model_name}")
    model_module = load_model_module(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_step(f"loading model device={device}")
    processor, model = model_module.load_model(device)
    model.eval()
    log_step("model loaded")
    intervention = ScaledIntervention(BASELINE_ALPHA)
    model_module.register_intervention(model, intervention)

    log_step("discovering decode run")
    decode_run_dir = discover_decode_run(model_name)
    log_step(f"loading decode metrics decode_run_dir={decode_run_dir}")
    attention_metrics, logit_metrics = load_decode_metrics(decode_run_dir)
    log_step(
        f"loaded decode metrics attention_keys={len(attention_metrics)} logit_keys={len(logit_metrics)}"
    )
    selected = selected_sample_ids(decode_run_dir)
    log_step(f"selected samples subsets={len(selected)} total={sum(len(ids) for ids in selected.values())}")

    run_id = f"kv_stability_under_alpha_{uuid.uuid4().hex[:8]}"
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "kv_stability_config.json",
        {
            "run_id": run_id,
            "model": model_name,
            "alphas": ALPHAS,
            "baseline_alpha": BASELINE_ALPHA,
            "subsets": SUBSETS,
            "num_samples_per_subset": NUM_SAMPLES_PER_SUBSET,
            "decode_metrics_run_dir": str(decode_run_dir) if decode_run_dir else None,
            "device": str(device),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
    )

    rows = []
    skipped = []
    for subset, samples in load_samples(selected):
        for sample in tqdm(samples, desc=subset):
            prompt_format = sample["prompt_format"]
            prompt = prompt_format["prefix"] + sample["question"] + prompt_format["suffix"]
            try:
                log_step(f"sample start subset={subset} sample_id={sample['id']}")
                prefilled_by_alpha = {}
                for alpha in ALPHAS:
                    log_step(f"prefill start subset={subset} sample_id={sample['id']} alpha={alpha}")
                    intervention.alpha = alpha
                    prefilled_by_alpha[alpha] = prefill(
                        sample["image"],
                        prompt,
                        processor,
                        model,
                        model_module.multimodal_masks,
                    )
                    log_step(f"prefill done subset={subset} sample_id={sample['id']} alpha={alpha}")

                sample_rows = compute_rows_for_sample(
                    model,
                    model_name,
                    subset,
                    sample,
                    prefilled_by_alpha,
                )
                for row in sample_rows:
                    attention_key = (
                        row["subset"],
                        row["sample_id"],
                        float(row["alpha"]),
                        int(row["layer"]),
                    )
                    logit_key = (row["subset"], row["sample_id"], float(row["alpha"]))
                    row.update(
                        {
                            "decode_attention_kl_mean": None,
                            "decode_attention_cosine_mean": None,
                            "logit_kl_mean": None,
                            "baseline_token_prob_mean": None,
                            "baseline_token_rank_median": None,
                            "greedy_agreement_mean": None,
                        }
                    )
                    row.update(attention_metrics.get(attention_key, {}))
                    row.update(logit_metrics.get(logit_key, {}))
                rows.extend(sample_rows)
                log_step(f"sample done subset={subset} sample_id={sample['id']} rows={len(sample_rows)}")
            except Exception as exc:
                log_step(f"sample skipped subset={subset} sample_id={sample['id']} reason={type(exc).__name__}")
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

    metric_names = [
        "k_cos_to_layer0",
        "v_cos_to_layer0",
        "k_adjacent_cos",
        "v_adjacent_cos",
        "k_rel_update",
        "v_rel_update",
        "k_cos_to_baseline",
        "v_cos_to_baseline",
        "k_rel_diff_from_baseline",
        "v_rel_diff_from_baseline",
        "vision_hidden_cos_to_layer0",
        "vision_hidden_adjacent_cos",
        "vision_hidden_rel_update",
        "vision_hidden_cos_to_baseline",
        "vision_hidden_rel_diff_from_baseline",
        "decode_attention_kl_mean",
        "decode_attention_cosine_mean",
        "logit_kl_mean",
        "baseline_token_prob_mean",
        "baseline_token_rank_median",
        "greedy_agreement_mean",
    ]
    alpha_summary = summarize_by(["alpha"], rows, metric_names)
    alpha_layer_summary = summarize_by(["alpha", "layer"], rows, metric_names)
    correlations = correlation_summary(rows)

    write_jsonl(run_dir / "kv_stability.jsonl", rows)
    write_jsonl(run_dir / "kv_stability_skipped.jsonl", skipped)
    write_csv(run_dir / "kv_stability_summary.csv", alpha_summary)
    write_csv(run_dir / "kv_stability_layer_summary.csv", alpha_layer_summary)
    write_csv(run_dir / "kv_stability_correlations.csv", correlations)
    write_json(
        run_dir / "kv_stability_summary.json",
        {
            "n_rows": len(rows),
            "n_skipped": len(skipped),
            "by_alpha": alpha_summary,
            "correlations": correlations,
        },
    )
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    run_dir = run(args.model)
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
