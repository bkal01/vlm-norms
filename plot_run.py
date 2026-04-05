import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def plot_hidden_states(hidden_dict: dict[str, torch.Tensor], title: str):
    first = next(iter(hidden_dict.values()))
    L = first.shape[0]
    layers = list(range(L))
    layers_update = list(range(1, L))

    norms, abs_updates, rel_updates, cosines = {}, {}, {}, {}
    for name, h in hidden_dict.items():
        norms[name] = h.norm(dim=-1).mean(dim=-1).numpy()
        abs_updates[name] = (h[1:] - h[:-1]).norm(dim=-1).mean(dim=-1).numpy()
        rel_updates[name] = (
            (h[1:] - h[:-1]).norm(dim=-1).mean(dim=-1) / h[:-1].norm(dim=-1).mean(dim=-1)
        ).numpy()
        cosines[name] = F.cosine_similarity(h, h[0:1].expand_as(h), dim=-1).mean(dim=-1).numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=11)
    kw = dict(marker="o", markersize=3)

    ax = axes[0, 0]
    for name, vals in norms.items():
        ax.plot(layers, vals, label=name, **kw)
    ax.set_title("Mean token norm")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"mean $\|h_\ell\|$")
    ax.legend()

    ax = axes[0, 1]
    for name, vals in abs_updates.items():
        ax.plot(layers_update, vals, label=name, **kw)
    ax.set_title("Absolute update magnitude")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"mean $\|h_\ell - h_{\ell-1}\|$")
    ax.legend()

    ax = axes[1, 0]
    for name, vals in rel_updates.items():
        ax.plot(layers_update, vals, label=name, **kw)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Relative update magnitude")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\|h_\ell - h_{\ell-1}\| \;/\; \|h_{\ell-1}\|$")
    ax.legend()

    ax = axes[1, 1]
    for name, vals in cosines.items():
        ax.plot(layers, vals, label=name, **kw)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Cosine similarity to layer-0 embedding")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"mean $\cos(h_\ell,\, h_0)$")
    ax.legend()

    plt.tight_layout()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("--runs-dir", default="runs", type=Path)
    args = p.parse_args()

    base = args.runs_dir / args.run_id
    _load = lambda f: torch.load(base / f, map_location="cpu", weights_only=False).float()

    vlm_vision_h = _load("vlm_vision_h.pt")
    vlm_text_h = _load("vlm_text_h.pt")
    textonly_h = _load("textonly_h.pt")

    vlm_txt = (base / "vlm_generated.txt").read_text()
    textonly_txt = (base / "textonly_generated.txt").read_text()
    print(f"VLM: {vlm_txt}")
    print(f"Text-only: {textonly_txt}")

    plot_hidden_states(
        {"vision": vlm_vision_h, "text": vlm_text_h},
        title=f"VLM — {args.run_id}",
    )
    plot_hidden_states(
        {"text": textonly_h},
        title=f"Text-only — {args.run_id}",
    )
    plt.show()


if __name__ == "__main__":
    main()
