import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def load_model(device: torch.device):
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        dtype=torch.bfloat16,
        _attn_implementation="sdpa",
    ).to(device)
    return processor, model


def multimodal_masks(
    processor,
    token_ids,
):
    image_mask = token_ids == processor.image_token_id

    token_ids_list = token_ids.tolist()
    text_start = token_ids_list.index(processor.vision_end_token_id) + 1
    text_end = token_ids_list.index(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
    text_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    text_mask[text_start:text_end] = True

    return image_mask, text_mask

def text_only_mask(
    processor,
    token_ids,
):
    token_ids_list = token_ids.tolist()
    im_start_index = token_ids_list.index(processor.tokenizer.convert_tokens_to_ids("<|im_start|>"))
    user_role_ids = processor.tokenizer.encode("user\n", add_special_tokens=False)
    text_start = im_start_index + len(user_role_ids) + 1
    text_end = token_ids_list.index(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
    text_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    text_mask[text_start:text_end] = True
    return text_mask



def prefill(
    image: Image.Image,
    prompt: str,
    processor,
    model
):
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
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = torch.stack(outputs.hidden_states, dim=0).squeeze(1)
    image_mask, text_mask = multimodal_masks(processor, inputs["input_ids"][0])
    return hidden_states[:, image_mask, :], hidden_states[:, text_mask, :]


def prefill_text_only(prompt: str, processor, model):
    """
    Prefill text-only input and return the hidden states for text.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = torch.stack(outputs.hidden_states, dim=0).squeeze(1)
    text_mask = text_only_mask(processor, inputs["input_ids"][0])
    return hidden_states[:, text_mask, :]

def register_intervention(model, intervention):
    orig_get_image_features = model.model.get_image_features
    def patch(*args, **kwargs):
        out = orig_get_image_features(*args, **kwargs)
        out.last_hidden_state = intervention.reduce_norm(out.last_hidden_state)
        return out
    model.model.get_image_features = patch

