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


def generation_logits(generation: dict) -> torch.Tensor | None:
    logits = generation.get("scores")
    if logits is None:
        logits = generation.get("logits")
    if logits is None:
        return None
    return torch.cat([step.detach().cpu().float() for step in logits], dim=0)


def compute_logit_sensitivity_rows(
    *,
    subset: str,
    sample_id: str,
    alpha: float,
    logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    baseline_generated_token_ids: torch.Tensor,
) -> list[dict]:
    if logits.shape[-1] != baseline_logits.shape[-1]:
        return []

    n_steps = min(
        logits.shape[0],
        baseline_logits.shape[0],
        baseline_generated_token_ids.shape[0],
    )
    logits = logits[:n_steps]
    baseline_prefix = baseline_logits[:n_steps]
    baseline_token_prefix = baseline_generated_token_ids[:n_steps].long()

    logp = torch.log_softmax(logits, dim=-1)
    logq = torch.log_softmax(baseline_prefix, dim=-1)
    p = logp.exp()
    kl = (p * (logp - logq)).sum(dim=-1)

    baseline_token_logp = logp.gather(1, baseline_token_prefix[:, None]).squeeze(1)
    baseline_token_prob = baseline_token_logp.exp()
    baseline_token_logits = logits.gather(1, baseline_token_prefix[:, None]).squeeze(1)
    baseline_token_rank = (logits > baseline_token_logits[:, None]).sum(dim=-1) + 1
    greedy_token = logits.argmax(dim=-1)
    baseline_greedy_token = baseline_prefix.argmax(dim=-1)
    greedy_agreement = greedy_token == baseline_greedy_token

    rows = []
    for step in range(n_steps):
        rows.append(
            {
                "subset": subset,
                "sample_id": sample_id,
                "alpha": float(alpha),
                "step": step,
                "kl": float(kl[step]),
                "baseline_token_id": int(baseline_token_prefix[step]),
                "baseline_token_prob": float(baseline_token_prob[step]),
                "baseline_token_rank": int(baseline_token_rank[step]),
                "greedy_token_id": int(greedy_token[step]),
                "baseline_greedy_token_id": int(baseline_greedy_token[step]),
                "greedy_agreement": int(greedy_agreement[step]),
            }
        )
    return rows


def extract_vision_attention(generation: dict) -> torch.Tensor | None:
    attentions = generation.get("attentions")
    if attentions is None:
        return None

    prompt_image_mask = generation["prompt_image_mask"].detach().cpu().bool()
    prompt_length = int(generation["prompt_length"])

    by_step = []
    for step_attentions in attentions:
        by_layer = []
        for attn in step_attentions:
            prompt_attn = attn.detach().cpu()[..., -1, :prompt_length]
            by_layer.append(prompt_attn[..., prompt_image_mask].squeeze(0))
        by_step.append(torch.stack(by_layer))

    return torch.stack(by_step).float()


def compute_attention_divergence_rows(
    *,
    subset: str,
    sample_id: str,
    alpha: float,
    attention: torch.Tensor,
    baseline_attention: torch.Tensor,
) -> list[dict]:
    if attention.shape[1:] != baseline_attention.shape[1:]:
        return []

    n_steps = min(attention.shape[0], baseline_attention.shape[0])
    attention = attention[:n_steps]
    baseline_prefix = baseline_attention[:n_steps]

    p = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    q = baseline_prefix / baseline_prefix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    kl = (p * (p.clamp_min(1e-8).log() - q.clamp_min(1e-8).log())).sum(dim=-1)
    cosine = F.cosine_similarity(attention, baseline_prefix, dim=-1)

    rows = []
    for step in range(attention.shape[0]):
        for layer in range(attention.shape[1]):
            rows.append(
                {
                    "subset": subset,
                    "sample_id": sample_id,
                    "alpha": float(alpha),
                    "step": step,
                    "layer": layer,
                    "kl_mean": float(kl[step, layer].mean()),
                    "cosine_mean": float(cosine[step, layer].mean()),
                }
            )
    return rows
