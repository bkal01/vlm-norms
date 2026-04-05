import json
import uuid
from pathlib import Path

import modal

app = modal.App("vlm-norms")

VOLUME_NAME = "vlm-norms-runs"
runs_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

DEFAULT_SUBSETS = [
    "chart", "counting", "document", "general", "grounding",
    "math", "scene", "spatial", "table",
]


def download_datbench():
    from datasets import load_dataset

    for subset in DEFAULT_SUBSETS:
        load_dataset("DatologyAI/DatBench", subset, split="test")


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .env({"PYTORCH_ALLOC_CONF": "expandable_segments:True"})
    .uv_pip_install(
        "torch",
        "torchvision",
        "transformers",
        "datasets",
        "num2words",
        "tqdm",
    )
    .run_function(download_datbench)
    .add_local_dir(local_path="models", remote_path="/root/models")
)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/runs": runs_vol})
def run(run_id: str, subsets: list[str], num_samples: int):
    import torch
    from datasets import load_dataset
    from tqdm import tqdm

    from models.smolvlm import (
        load_model,
        prefill,
        prefill_text_only,
        compute_metrics,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "subsets": subsets,
        "num_samples_per_subset": num_samples,
        "model": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    processor, model = load_model(device)

    for subset in subsets:
        ds = load_dataset("DatologyAI/DatBench", subset, split="test")
        n = min(num_samples, len(ds))
        for i, sample in tqdm(enumerate(ds), total=n, desc=subset):
            if i >= num_samples:
                break

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
                    "text_norms": tm["norms"].cpu().float(),
                    "text_abs": tm["abs"].cpu().float(),
                    "text_rel": tm["rel"].cpu().float(),
                    "text_cos": tm["cos"].cpu().float(),
                    "textonly_norms": tom["norms"].cpu().float(),
                    "textonly_abs": tom["abs"].cpu().float(),
                    "textonly_rel": tom["rel"].cpu().float(),
                    "textonly_cos": tom["cos"].cpu().float(),
                },
                sample_dir / "metrics.pt",
            )

    runs_vol.commit()
    return run_id


@app.local_entrypoint()
def main(
    subsets: str = ",".join(DEFAULT_SUBSETS),
    num_samples: int = 10,
):
    run_id = str(uuid.uuid4())
    subset_list = [s.strip() for s in subsets.split(",")]
    run.remote(run_id, subset_list, num_samples)
    print(f"run_id={run_id}")
    print(
        f"Pull artifacts: modal volume get {VOLUME_NAME} {run_id} runs"
    )
