import json
import os
import torch

def compute_ratio(run_root):
    run_id = os.listdir(run_root)[0]
    run_dir = os.path.join(run_root, run_id)

    with open(os.path.join(run_dir, "config.json")) as f:
        model_name = json.load(f)["model"]

    ratios = []
    for sample in os.listdir(run_dir):
        path = os.path.join(run_dir, sample, "metrics.pt")
        if not os.path.exists(path):
            continue
        m = torch.load(path, weights_only=False)
        vision_mean = m["vision_norms"][0].item()
        text_mean = m["text_norms"][0].item()
        ratios.append(vision_mean / text_mean)

    return model_name, sum(ratios) / len(ratios)


for run_root in ["runs/qwen", "runs/smolvlm"]:
    model_name, ratio = compute_ratio(run_root)
    print(f"{model_name}   vision/text norm ratio: {ratio:.4f}")
