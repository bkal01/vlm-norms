import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("--runs-dir", default="runs", type=Path)
    args = p.parse_args()

    base = args.runs_dir / args.run_id
    vision_h = torch.load(base / "vision_h.pt", map_location="cpu", weights_only=False).float()
    text_h   = torch.load(base / "text_h.pt",   map_location="cpu", weights_only=False).float()
    txt = (base / "generated.txt").read_text()
    print(txt)
    print(f"vision_h: {vision_h.shape}, text_h: {text_h.shape}")

    L = vision_h.shape[0]
    layers = list(range(L))
    layers_update = list(range(1, L))  # updates defined from layer 1 onwards

    # 1. mean norm per layer
    v_norm = vision_h.norm(dim=-1).mean(dim=-1).numpy()
    t_norm = text_h.norm(dim=-1).mean(dim=-1).numpy()

    # 2. absolute update magnitude: mean over tokens of ||h_l - h_{l-1}||
    v_abs = (vision_h[1:] - vision_h[:-1]).norm(dim=-1).mean(dim=-1).numpy()
    t_abs = (text_h[1:]   - text_h[:-1]).norm(dim=-1).mean(dim=-1).numpy()

    # 3. relative update magnitude: abs / ||h_{l-1}||
    v_rel = (vision_h[1:] - vision_h[:-1]).norm(dim=-1).mean(dim=-1) / vision_h[:-1].norm(dim=-1).mean(dim=-1)
    t_rel = (text_h[1:]   - text_h[:-1]).norm(dim=-1).mean(dim=-1)   / text_h[:-1].norm(dim=-1).mean(dim=-1)
    v_rel = v_rel.numpy()
    t_rel = t_rel.numpy()

    # 4. cosine similarity to layer-0 embedding, mean over tokens
    v_cos = F.cosine_similarity(vision_h, vision_h[0:1].expand_as(vision_h), dim=-1).mean(dim=-1).numpy()
    t_cos = F.cosine_similarity(text_h,   text_h[0:1].expand_as(text_h),   dim=-1).mean(dim=-1).numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Run: {args.run_id}", fontsize=11)

    kw = dict(marker="o", markersize=3)

    ax = axes[0, 0]
    ax.plot(layers, v_norm, label="vision", **kw)
    ax.plot(layers, t_norm, label="text",   **kw)
    ax.set_title("Mean token norm")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"mean $\|h_\ell\|$")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(layers_update, v_abs, label="vision", **kw)
    ax.plot(layers_update, t_abs, label="text",   **kw)
    ax.set_title("Absolute update magnitude")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"mean $\|h_\ell - h_{\ell-1}\|$")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(layers_update, v_rel, label="vision", **kw)
    ax.plot(layers_update, t_rel, label="text",   **kw)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Relative update magnitude")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\|h_\ell - h_{\ell-1}\| \;/\; \|h_{\ell-1}\|$")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(layers, v_cos, label="vision", **kw)
    ax.plot(layers, t_cos, label="text",   **kw)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Cosine similarity to layer-0 embedding")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"mean $\cos(h_\ell,\, h_0)$")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
