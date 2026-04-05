import torch

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