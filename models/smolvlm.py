import torch
import torch.nn.functional as F

from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)


def load_model(
    device: torch.device,
):
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    )
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        dtype=torch.bfloat16,
        _attn_implementation="sdpa",
    ).to(device)

    return processor, model


def prefill(image: Image.Image, prompt: str, processor, model):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(model.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    h = torch.stack(outputs.hidden_states, dim=0).squeeze(1)
    ids = inputs["input_ids"][0]
    fake_id = processor.tokenizer.convert_tokens_to_ids("<fake_token_around_image>")
    fake_pos = (ids == fake_id).nonzero(as_tuple=True)[0]
    vision_mask = torch.zeros(len(ids), dtype=torch.bool, device=ids.device)
    vision_mask[fake_pos[0]:fake_pos[-1] + 1] = True
    return h[:, vision_mask, :], h[:, ~vision_mask, :]


def prefill_text_only(prompt: str, processor, model):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    return torch.stack(outputs.hidden_states, dim=0).squeeze(1)


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


def generate(
    image: Image.Image,
    prompt: str,
    processor,
    model,
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=512,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    generated_ids = outputs.sequences
    h = torch.stack(outputs.hidden_states[0], dim=0).squeeze(1)
    vision_mask = inputs["input_ids"][0] == processor.image_token_id
    vision_hidden_states = h[:, vision_mask, :]
    text_hidden_states = h[:, ~vision_mask, :]
    generated_texts = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return generated_texts[0], vision_hidden_states, text_hidden_states

def generate_text_only(
    prompt: str,
    processor,
    model,
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    gen_outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=512,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    hidden_states = torch.stack(gen_outputs.hidden_states[0], dim=0).squeeze(1)
    generated_text = processor.batch_decode(
        gen_outputs.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]
    return generated_text, hidden_states