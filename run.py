import argparse
import json
import os
import uuid
from pathlib import Path

import yaml

from models.interventions import RMSNormIntervention, ScaledIntervention


os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


DEFAULT_SUBSETS = [
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

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
SUPPORTED_MODELS = [
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
]
SUPPORTED_INTERVENTIONS = ["rms", "scaled"]


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


def parse_intervention(value: dict | str | None) -> dict:
    if value is None:
        return {"type": "rms"}

    if isinstance(value, str):
        value = {"type": value}
    if not isinstance(value, dict):
        raise ValueError("intervention must be a string or YAML mapping")

    allowed_keys = {"type", "alpha"}
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

    if intervention_type == "rms":
        if "alpha" in value:
            raise ValueError("intervention.alpha is only valid for type='scaled'")
        return {"type": "rms"}

    alpha = value.get("alpha")
    if not isinstance(alpha, (int, float)):
        raise ValueError("intervention.alpha must be a number for type='scaled'")

    return {"type": "scaled", "alpha": float(alpha)}


def build_intervention(config: dict):
    if config["type"] == "rms":
        return RMSNormIntervention()
    if config["type"] == "scaled":
        return ScaledIntervention(config["alpha"])
    raise ValueError(f"unknown intervention type={config['type']!r}")


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


def generation_to_saveable(generation: dict):
    generation = dict(generation)
    if generation.get("scores", None) is not None:
        generation["logits"] = None
    return to_saveable(generation)


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

    from models.utils import compute_metrics

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
        "model": model_name,
        "intervention": intervention_config,
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    processor, model = load_model(device)
    model.eval()
    register_intervention(model, build_intervention(intervention_config))
    for subset in subsets:
        ds = load_dataset("DatologyAI/DatBench", subset, split="test")
        n = min(num_samples, len(ds))

        for sample in tqdm(ds.select(range(n)), total=n, desc=subset):
            sample_id = sample["id"]
            sample_dir = run_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            fmt = sample["prompt_format"]
            prompt = fmt["prefix"] + sample["question"] + fmt["suffix"]

            generation = generate(sample["image"], prompt, processor, model)
            prefill_h = torch.stack(generation["hidden_states"][0], dim=0).squeeze(1)
            vision_h = prefill_h[:, generation["prompt_image_mask"], :]
            text_h = prefill_h[:, generation["prompt_text_mask"], :]
            vm = compute_metrics(vision_h)
            tm = compute_metrics(text_h)

            textonly_generation = generate_text_only(prompt, processor, model)
            textonly_h = torch.stack(
                textonly_generation["hidden_states"][0], dim=0
            ).squeeze(1)
            textonly_h = textonly_h[:, textonly_generation["prompt_text_mask"], :]
            tom = compute_metrics(textonly_h)

            torch.save(
                to_saveable({
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
                    "textonly_norms": tom["norms"].cpu().float(),
                    "textonly_abs": tom["abs"].cpu().float(),
                    "textonly_rel": tom["rel"].cpu().float(),
                    "textonly_cos": tom["cos"].cpu().float(),
                    "textonly_update_align": tom["update_align"].cpu().float(),
                    "textonly_adjacent_cos": tom["adjacent_cos"].cpu().float(),
                    "generation": generation_to_saveable(generation),
                    "textonly_generation": generation_to_saveable(
                        textonly_generation
                    ),
                }),
                sample_dir / "metrics.pt",
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
