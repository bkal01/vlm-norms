import argparse
from dataclasses import asdict, is_dataclass
import json
import os
import uuid
from pathlib import Path

import yaml

from models.interventions import ScaledIntervention


os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


DEFAULT_SUBSETS = [
    "chart",
    "counting",
    "document",
    "general",
    "grounding",
    "scene",
    "spatial",
    "table",
]

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_ALPHAS = [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 1.0, 3.0]
SUPPORTED_MODELS = [
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
]
SUPPORTED_INTERVENTIONS = ["scaled"]


def parse_subsets(value: str | list[str]) -> list[str]:
    if isinstance(value, str) and value.strip().lower() == "all":
        return DEFAULT_SUBSETS.copy()

    if isinstance(value, str):
        subsets = [subset.strip() for subset in value.split(",") if subset.strip()]
    elif isinstance(value, list):
        subsets = value
    else:
        raise ValueError("subsets must be 'all', a comma-separated string, or a list")

    if not all(isinstance(subset, str) for subset in subsets):
        raise ValueError("subsets must contain only strings")

    unknown = sorted(set(subsets) - set(DEFAULT_SUBSETS))
    if unknown:
        raise ValueError(
            f"unknown subset(s): {', '.join(unknown)}; "
            f"expected any of: {', '.join(DEFAULT_SUBSETS)}"
        )
    if not subsets:
        raise ValueError("at least one subset is required")
    return subsets


def parse_intervention(value: dict | None) -> dict:
    if value is None:
        return {"type": "scaled", "alphas": DEFAULT_ALPHAS.copy()}

    if not isinstance(value, dict):
        raise ValueError("intervention must be a YAML mapping")

    allowed_keys = {"type", "alphas"}
    unknown_keys = sorted(set(value) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"unknown intervention key(s): {', '.join(unknown_keys)}; "
            f"expected: {', '.join(sorted(allowed_keys))}"
        )

    intervention_type = value.get("type")
    if intervention_type not in SUPPORTED_INTERVENTIONS:
        raise ValueError(
            f"unknown intervention type={intervention_type!r}; "
            f"expected one of: {', '.join(SUPPORTED_INTERVENTIONS)}"
        )

    alphas = value.get("alphas", DEFAULT_ALPHAS.copy())
    if not isinstance(alphas, list):
        raise ValueError("intervention.alphas must be a list of numbers")
    if not alphas:
        raise ValueError("intervention.alphas must not be empty")
    if not all(isinstance(alpha, (int, float)) for alpha in alphas):
        raise ValueError("intervention.alphas must contain only numbers")

    alphas = [float(alpha) for alpha in alphas]
    if 1.0 not in alphas:
        raise ValueError("intervention.alphas must include 1.0 as the baseline")

    return {"type": "scaled", "alphas": alphas}


def build_intervention(alpha: float):
    return ScaledIntervention(alpha)


def alpha_label(alpha: float) -> str:
    return f"{alpha:g}"


def alpha_dir_name(alpha: float) -> str:
    return f"alpha_{alpha_label(alpha)}"


def load_config(config_path: Path) -> dict:
    raw_config = yaml.safe_load(config_path.read_text())
    if not isinstance(raw_config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    allowed_keys = {"model", "subsets", "num_samples", "intervention"}
    unknown_keys = sorted(set(raw_config) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"unknown config key(s): {', '.join(unknown_keys)}; "
            f"expected: {', '.join(sorted(allowed_keys))}"
        )

    model_name = raw_config.get("model", DEFAULT_MODEL)
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"unknown model={model_name!r}; expected one of: {', '.join(SUPPORTED_MODELS)}"
        )

    num_samples = raw_config.get("num_samples", 10)
    if not isinstance(num_samples, int):
        raise ValueError("num_samples must be an integer")
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")

    return {
        "model": model_name,
        "subsets": parse_subsets(raw_config.get("subsets", DEFAULT_SUBSETS.copy())),
        "num_samples": num_samples,
        "intervention": parse_intervention(raw_config.get("intervention")),
    }


def load_model_fns(model_name: str):
    if model_name == "Qwen/Qwen3-VL-2B-Instruct":
        from models.qwen3vl import generate, generate_text_only, load_model
        from models.qwen3vl import register_intervention
    elif model_name == "HuggingFaceTB/SmolVLM2-2.2B-Instruct":
        from models.smolvlm import generate, generate_text_only, load_model
        from models.smolvlm import register_intervention
    else:
        raise ValueError(f"unknown model_name={model_name!r}")

    return load_model, generate, generate_text_only, register_intervention


def to_saveable(value):
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: to_saveable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_saveable(item) for item in value)
    if isinstance(value, list):
        return [to_saveable(item) for item in value]
    return value


def tensor_to_json_list(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def write_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def write_jsonl_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_answer_row(
    *,
    run_id: str,
    subset: str,
    sample: dict,
    prompt: str,
    condition: str,
    intervention_config: dict,
    generation: dict,
) -> dict:
    return {
        "run_id": run_id,
        "subset": subset,
        "sample_id": sample["id"],
        "condition": condition,
        "alpha": intervention_config.get("alpha"),
        "intervention": intervention_config,
        "prompt": prompt,
        "generated_text": generation["generated_text"],
        "generated_token_ids": tensor_to_json_list(generation["generated_token_ids"]),
        "answer": sample.get("answer"),
        "all_answers": sample.get("all_answers"),
        "eval_mode": sample.get("eval_mode"),
        "is_circular": sample.get("is_circular"),
        "metadata": sample.get("metadata"),
        "source_info": sample.get("source_info"),
        "eval_metrics": sample.get("eval_metrics"),
    }


def score_answers(
    *,
    run_id: str,
    run_dir: Path,
    subset: str,
    condition: str,
    dataset,
    answer_rows: list[dict],
) -> None:
    from datbench import DatBenchEvaluator, VLMResponse

    if not answer_rows:
        return

    scores_dir = run_dir / "scores"
    scores_dir.mkdir(exist_ok=True)

    evaluator = DatBenchEvaluator(dataset, subset)
    responses = [
        VLMResponse(id=row["sample_id"], raw_output=row["generated_text"])
        for row in answer_rows
    ]
    report = evaluator.compute_metrics(responses)
    report_path = scores_dir / f"{subset}_{condition}.json"
    report.save(str(report_path))

    for result in report.results:
        result_dict = asdict(result) if is_dataclass(result) else dict(result)
        write_jsonl(
            run_dir / "scores.jsonl",
            {
                "run_id": run_id,
                "subset": subset,
                "condition": condition,
                "alpha": answer_rows[0].get("alpha"),
                "intervention": answer_rows[0]["intervention"],
                **result_dict,
            },
        )


def run(
    run_id: str,
    subsets: list[str],
    num_samples: int,
    model_name: str,
    intervention_config: dict,
    config_path: Path,
) -> Path:
    import torch
    from datasets import load_dataset
    from tqdm import tqdm

    from models.utils import (
        compute_attention_divergence_rows,
        compute_attention_metrics,
        compute_logit_sensitivity_rows,
        compute_metrics,
        extract_vision_attention,
        generation_logits,
    )

    load_model, generate, generate_text_only, register_intervention = load_model_fns(
        model_name
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "config_path": str(config_path),
        "subsets": subsets,
        "num_samples_per_subset": num_samples,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "model": model_name,
        "intervention": intervention_config,
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    processor, model = load_model(device)
    model.eval()
    alphas = intervention_config["alphas"]
    intervention = build_intervention(alphas[0])
    register_intervention(model, intervention)
    answers_path = run_dir / "answers.jsonl"
    logit_sensitivity_path = run_dir / "logit_sensitivity.jsonl"
    attention_divergence_path = run_dir / "attention_divergence_from_baseline.jsonl"
    baseline_alpha = 1.0
    ordered_alphas = [baseline_alpha] + [
        alpha for alpha in alphas if alpha != baseline_alpha
    ]
    for subset in subsets:
        ds = load_dataset("DatologyAI/DatBench", subset, split="test")
        n = min(num_samples, len(ds))
        subset_ds = ds.select(range(n))
        real_answer_rows_by_alpha = {alpha: [] for alpha in alphas}
        textonly_answer_rows = []

        for sample in tqdm(subset_ds, total=n, desc=subset):
            sample_id = sample["id"]
            sample_dir = run_dir / subset / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            fmt = sample["prompt_format"]
            prompt = fmt["prefix"] + sample["question"] + fmt["suffix"]

            baseline_logits = None
            baseline_generated_token_ids = None
            baseline_vision_attention = None

            for alpha in ordered_alphas:
                intervention.alpha = alpha
                alpha_intervention_config = {"type": "scaled", "alpha": alpha}
                condition = f"real_alpha_{alpha_label(alpha)}"
                alpha_dir = sample_dir / alpha_dir_name(alpha)
                alpha_dir.mkdir(parents=True, exist_ok=True)

                generation = generate(
                    sample["image"],
                    prompt,
                    processor,
                    model,
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                )
                answer_row = build_answer_row(
                    run_id=run_id,
                    subset=subset,
                    sample=sample,
                    prompt=prompt,
                    condition=condition,
                    intervention_config=alpha_intervention_config,
                    generation=generation,
                )
                write_jsonl(answers_path, answer_row)
                real_answer_rows_by_alpha[alpha].append(answer_row)

                prefill_h = torch.stack(
                    generation["hidden_states"][0], dim=0
                ).squeeze(1)
                vision_h = prefill_h[:, generation["prompt_image_mask"], :]
                text_h = prefill_h[:, generation["prompt_text_mask"], :]
                vm = compute_metrics(vision_h)
                tm = compute_metrics(text_h)

                attention_metrics = compute_attention_metrics(
                    generation["attentions"],
                    generation["prompt_image_mask"],
                    generation["prompt_text_mask"],
                )
                logits = generation_logits(generation)
                vision_attention = extract_vision_attention(generation)
                generated_token_ids = generation["generated_token_ids"].detach().cpu()

                if alpha == baseline_alpha:
                    baseline_logits = logits
                    baseline_generated_token_ids = generated_token_ids
                    baseline_vision_attention = vision_attention

                if logits is not None and baseline_logits is not None:
                    write_jsonl_rows(
                        logit_sensitivity_path,
                        compute_logit_sensitivity_rows(
                            subset=subset,
                            sample_id=sample_id,
                            alpha=alpha,
                            logits=logits,
                            baseline_logits=baseline_logits,
                            baseline_generated_token_ids=baseline_generated_token_ids,
                        ),
                    )

                if (
                    alpha != baseline_alpha
                    and vision_attention is not None
                    and baseline_vision_attention is not None
                ):
                    write_jsonl_rows(
                        attention_divergence_path,
                        compute_attention_divergence_rows(
                            subset=subset,
                            sample_id=sample_id,
                            alpha=alpha,
                            attention=vision_attention,
                            baseline_attention=baseline_vision_attention,
                        ),
                    )

                torch.save(
                    to_saveable({
                        "run_id": run_id,
                        "subset": subset,
                        "sample_id": sample_id,
                        "condition": condition,
                        "alpha": alpha,
                        "intervention": alpha_intervention_config,
                        "vision_norms": vm["norms"].cpu().float(),
                        "vision_abs": vm["abs"].cpu().float(),
                        "vision_rel": vm["rel"].cpu().float(),
                        "vision_cos": vm["cos"].cpu().float(),
                        "vision_update_align": vm["update_align"].cpu().float(),
                        "vision_adjacent_cos": vm["adjacent_cos"].cpu().float(),
                        "text_norms": tm["norms"].cpu().float(),
                        "text_abs": tm["abs"].cpu().float(),
                        "text_rel": tm["rel"].cpu().float(),
                        "text_cos": tm["cos"].cpu().float(),
                        "text_update_align": tm["update_align"].cpu().float(),
                        "text_adjacent_cos": tm["adjacent_cos"].cpu().float(),
                        "vision_attention_mass": attention_metrics[
                            "vision_attention_mass"
                        ],
                        "text_attention_mass": attention_metrics[
                            "text_attention_mass"
                        ],
                        "attention_entropy_over_vision": attention_metrics[
                            "attention_entropy_over_vision"
                        ],
                        "generated_token_ids": generated_token_ids,
                        "prompt_length": generation["prompt_length"],
                        "prompt_image_token_count": int(
                            generation["prompt_image_mask"].sum()
                        ),
                        "prompt_text_token_count": int(
                            generation["prompt_text_mask"].sum()
                        ),
                    }),
                    alpha_dir / "metrics.pt",
                )
                del generation, prefill_h, vision_h, text_h, vm, tm
                del attention_metrics, logits, vision_attention, generated_token_ids

            textonly_generation = generate_text_only(
                prompt,
                processor,
                model,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                output_attentions=False,
                output_scores=False,
            )
            textonly_intervention_config = {"type": "textonly", "alpha": None}
            textonly_answer_row = build_answer_row(
                run_id=run_id,
                subset=subset,
                sample=sample,
                prompt=prompt,
                condition="textonly",
                intervention_config=textonly_intervention_config,
                generation=textonly_generation,
            )
            write_jsonl(answers_path, textonly_answer_row)
            textonly_answer_rows.append(textonly_answer_row)

            textonly_h = torch.stack(
                textonly_generation["hidden_states"][0], dim=0
            ).squeeze(1)
            textonly_h = textonly_h[:, textonly_generation["prompt_text_mask"], :]
            tom = compute_metrics(textonly_h)

            textonly_dir = sample_dir / "textonly"
            textonly_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                to_saveable({
                    "run_id": run_id,
                    "subset": subset,
                    "sample_id": sample_id,
                    "condition": "textonly",
                    "alpha": None,
                    "intervention": textonly_intervention_config,
                    "textonly_norms": tom["norms"].cpu().float(),
                    "textonly_abs": tom["abs"].cpu().float(),
                    "textonly_rel": tom["rel"].cpu().float(),
                    "textonly_cos": tom["cos"].cpu().float(),
                    "textonly_update_align": tom["update_align"].cpu().float(),
                    "textonly_adjacent_cos": tom["adjacent_cos"].cpu().float(),
                    "generated_token_ids": textonly_generation[
                        "generated_token_ids"
                    ].detach().cpu(),
                    "prompt_length": textonly_generation["prompt_length"],
                    "prompt_text_token_count": int(
                        textonly_generation["prompt_text_mask"].sum()
                    ),
                }),
                textonly_dir / "metrics.pt",
            )
            del textonly_generation, textonly_h, tom

        for alpha, real_answer_rows in real_answer_rows_by_alpha.items():
            score_answers(
                run_id=run_id,
                run_dir=run_dir,
                subset=subset,
                condition=f"real_alpha_{alpha_label(alpha)}",
                dataset=subset_ds,
                answer_rows=real_answer_rows,
            )
        score_answers(
            run_id=run_id,
            run_dir=run_dir,
            subset=subset,
            condition="textonly",
            dataset=subset_ds,
            answer_rows=textonly_answer_rows,
        )

    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VLM norm collection locally without Modal."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="YAML config file path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config
    config = load_config(config_path)
    run_id = str(uuid.uuid4())

    run_dir = run(
        run_id=run_id,
        subsets=config["subsets"],
        num_samples=config["num_samples"],
        model_name=config["model"],
        intervention_config=config["intervention"],
        config_path=config_path,
    )

    print(f"run_id={run_id}")
    print(f"artifacts={run_dir}")


if __name__ == "__main__":
    main()
