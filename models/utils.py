import torch
import torch.nn.functional as F


def compute_metrics(h: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    h is of shape (L, N, D)
    we want:
    - norm for each layer averaged over N tokens
    - absolute update magnitude for each layer averaged over tokens
    - relative update magnitude for each layer averaged over tokens
    - cosine similarity to layer-0 embedding for each layer averaged over tokens
    - cosine similarity between update and previous layer, averaged over tokens
    - cosine similarity cos(h_l, h_{l-1}), averaged over tokens
    """
    norms = h.norm(dim=-1).mean(dim=-1)

    updates = h[1:] - h[:-1]
    diffs = updates.norm(dim=-1)
    base = h[:-1].norm(dim=-1).clamp(min=1e-8)
    rel = (diffs / base).mean(dim=-1)

    cos = F.cosine_similarity(h, h[0].unsqueeze(0), dim=-1).mean(dim=-1)
    update_align = F.cosine_similarity(updates, h[:-1], dim=-1).mean(dim=-1)
    adjacent_cos = F.cosine_similarity(h[1:], h[:-1], dim=-1).mean(dim=-1)

    return {
        "norms": norms,
        "abs": diffs.mean(dim=-1),
        "rel": rel,
        "cos": cos,
        "update_align": update_align,
        "adjacent_cos": adjacent_cos,
    }

def compute_attention_metrics(
        attentions,
        V,
):
    # vision attention mass
    vision_attention_mass = torch.zeros((T, L, ))
    for t in range T:
        for l in range L:

