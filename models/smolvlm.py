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


def multimodal_masks(
    processor,
    token_ids,
):
    token_ids_list = token_ids.tolist()
    end_of_utterance_id = processor.tokenizer.convert_tokens_to_ids("<end_of_utterance>")
    end = token_ids_list.index(end_of_utterance_id)
    special_ids = set(processor.tokenizer.all_special_ids)
    i = end - 1
    while i > 0 and token_ids_list[i] not in special_ids:
        i -= 1
    start = i + 1

    image_mask = token_ids == processor.image_token_id
    text_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    text_mask[start:end] = True
    return image_mask, text_mask

def text_only_mask(
    processor,
    token_ids,
):
    token_ids_list = token_ids.tolist()
    im_start_index = token_ids_list.index(processor.tokenizer.convert_tokens_to_ids("<|im_start|>"))
    user_role_ids = processor.tokenizer.encode("User:", add_special_tokens=False)
    text_start = im_start_index + len(user_role_ids) + 1

    end_of_utterance_id = processor.tokenizer.convert_tokens_to_ids("<end_of_utterance>")
    text_end = token_ids_list.index(end_of_utterance_id)
    text_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    text_mask[text_start:text_end] = True
    return text_mask


def prefill(image: Image.Image, prompt: str, processor, model):
    """
    Prefill multimodal input and return the hidden states for image and text.
    """
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

    hidden_states = torch.stack(outputs.hidden_states, dim=0).squeeze(1)
    image_mask, text_mask = multimodal_masks(processor, inputs["input_ids"][0])
    return hidden_states[:, image_mask, :], hidden_states[:, text_mask, :]


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

    hidden_states = torch.stack(outputs.hidden_states, dim=0).squeeze(1)
    text_mask = text_only_mask(processor, inputs["input_ids"][0])
    return hidden_states[:, text_mask, :]


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
