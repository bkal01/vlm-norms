import argparse
import json
import os
import uuid
from pathlib import Path

from models.interventions import ScaledIntervention, RMSNormIntervention


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


def parse_subsets(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return DEFAULT_SUBSETS.copy()

    subsets = [subset.strip() for subset in value.split(",") if subset.strip()]
    unknown = sorted(set(subsets) - set(DEFAULT_SUBSETS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown subset(s): {', '.join(unknown)}; "
            f"expected any of: {', '.join(DEFAULT_SUBSETS)}"
        )
    if not subsets:
        raise argparse.ArgumentTypeError("at least one subset is required")
    return subsets


def load_model_fns(model_name: str):
    if model_name == "Qwen/Qwen3-VL-2B-Instruct":
        from models.qwen3vl import load_model, prefill, prefill_text_only
        from models.qwen3vl import register_intervention
    elif model_name == "HuggingFaceTB/SmolVLM2-2.2B-Instruct":
        from models.smolvlm import load_model, prefill, prefill_text_only
        from models.smolvlm import register_intervention
    else:
        raise ValueError(f"unknown model_name={model_name!r}")

    return load_model, prefill, prefill_text_only, register_intervention


def run(
    run_id: str,
    subsets: list[str],
    num_samples: int,
    model_name: str,
    runs_dir: Path,
    overwrite: bool = False,
) -> Path:
    import torch
    from datasets import load_dataset
    from tqdm import tqdm

    from models.utils import compute_metrics

    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")

    load_model, prefill, prefill_text_only, register_intervention = load_model_fns(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = runs_dir / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{run_dir} already exists and is not empty; use --overwrite or another --run-id"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "subsets": subsets,
        "num_samples_per_subset": num_samples,
        "model": model_name,
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    processor, model = load_model(device)
    model.eval()
    register_intervention(model, RMSNormIntervention())
    for subset in subsets:
        ds = load_dataset("DatologyAI/DatBench", subset, split="test")
        n = min(num_samples, len(ds))

        for sample in tqdm(ds.select(range(n)), total=n, desc=subset):
            sample_id = sample["id"]
            sample_dir = run_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            fmt = sample["prompt_format"]
            prompt = fmt["prefix"] + sample["question"] + fmt["suffix"]

            vision_h, text_h = prefill(sample["image"], prompt, processor, model)
            vm = compute_metrics(vision_h)
            tm = compute_metrics(text_h)

            textonly_h = prefill_text_only(prompt, processor, model)
            tom = compute_metrics(textonly_h)

            torch.save(
                {
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
                },
                sample_dir / "metrics.pt",
            )

    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VLM norm collection locally without Modal."
    )
    parser.add_argument(
        "--subsets",
        type=parse_subsets,
        default=DEFAULT_SUBSETS.copy(),
        help="Comma-separated DatBench subsets, or 'all'.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to run per subset.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=[
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            "Qwen/Qwen3-VL-2B-Instruct",
        ],
        help="Model to run.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Output run id. Defaults to a new UUID.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory where run artifacts are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty run directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_id = args.run_id or str(uuid.uuid4())

    run_dir = run(
        run_id=run_id,
        subsets=args.subsets,
        num_samples=args.num_samples,
        model_name=args.model,
        runs_dir=args.runs_dir,
        overwrite=args.overwrite,
    )

    print(f"run_id={run_id}")
    print(f"artifacts={run_dir}")


if __name__ == "__main__":
    main()
