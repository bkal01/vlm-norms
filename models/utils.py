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
        prompt_image_mask,
        prompt_text_mask,
):
    prompt_length = prompt_image_mask.shape[0]
    vision_attention_mass = []
    text_attention_mass = []
    attention_entropy_over_vision = []

    for step_attentions in attentions:
        per_step_vision_mass = []
        per_step_text_mass = []
        per_step_vision_entropy = []
        for attn in step_attentions:
            prompt_attention = attn[..., :prompt_length]
            vision_attn = prompt_attention[..., prompt_image_mask]

            vision_mass = vision_attn.sum(dim=-1)
            vision_mass = vision_mass[..., -1]
            per_step_vision_mass.append(vision_mass)

            text_mass = prompt_attention[..., prompt_text_mask].sum(dim=-1)
            text_mass = text_mass[..., -1]
            per_step_text_mass.append(text_mass)

            vision_mass_for_entropy = vision_attn.sum(dim=-1, keepdim=True)
            vision_probs = vision_attn / vision_mass_for_entropy.clamp(min=1e-8)
            vision_entropy = -(vision_probs * vision_probs.clamp(min=1e-8).log()).sum(dim=-1)
            vision_entropy = vision_entropy[..., -1]
            per_step_vision_entropy.append(vision_entropy)

        per_step_vision_mass = torch.stack(per_step_vision_mass)
        per_step_text_mass = torch.stack(per_step_text_mass)
        per_step_vision_entropy = torch.stack(per_step_vision_entropy)
        vision_attention_mass.append(
            per_step_vision_mass.mean(dim=(1, 2))
        )
        text_attention_mass.append(
            per_step_text_mass.mean(dim=(1, 2))
        )
        attention_entropy_over_vision.append(
            per_step_vision_entropy.mean(dim=(1, 2))
        )

    return {
        "vision_attention_mass": torch.stack(vision_attention_mass),
        "text_attention_mass": torch.stack(text_attention_mass),
        "attention_entropy_over_vision": torch.stack(attention_entropy_over_vision),
    }
