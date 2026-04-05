import uuid
from pathlib import Path

import modal

app = modal.App("vlm-norms")

VOLUME_NAME = "vlm-norms-runs"
runs_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

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
    )
    .add_local_dir(local_path="models", remote_path="/root/models")
    .add_local_dir(local_path="assets", remote_path="/root/assets")
)


@app.function(image=image, gpu="A10G", timeout=600, volumes={"/runs": runs_vol})
def run(run_id: str):
    import torch
    from PIL import Image

    from models.smolvlm import load_model, generate, generate_text_only

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = "Can you describe this image?"
    out = Path("/runs") / run_id
    out.mkdir(parents=True, exist_ok=True)

    processor, model = load_model(device)

    vlm_text, vision_h, vlm_text_h = generate(
        image=Image.open("assets/bee.jpg"),
        prompt=prompt,
        processor=processor,
        model=model,
    )
    (out / "vlm_generated.txt").write_text(vlm_text)
    torch.save(vision_h.float().detach().cpu(), out / "vlm_vision_h.pt")
    torch.save(vlm_text_h.float().detach().cpu(), out / "vlm_text_h.pt")
    print(f"VLM: {vlm_text}")

    textonly_text, textonly_h = generate_text_only(
        prompt=prompt,
        processor=processor,
        model=model,
    )
    (out / "textonly_generated.txt").write_text(textonly_text)
    torch.save(textonly_h.float().detach().cpu(), out / "textonly_h.pt")
    print(f"Text-only: {textonly_text}")

    runs_vol.commit()
    return run_id


@app.local_entrypoint()
def main():
    run_id = str(uuid.uuid4())
    run.remote(run_id)
    print(f"run_id={run_id}")
    print(
        f"Pull artifacts: mkdir -p runs && modal volume get {VOLUME_NAME} {run_id} runs"
    )
