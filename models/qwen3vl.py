import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


QWEN3VL_MAX_VISUAL_TOKENS = 1024
QWEN3VL_PIXELS_PER_VISUAL_TOKEN = 28 * 28
QWEN3VL_MAX_PIXELS = QWEN3VL_MAX_VISUAL_TOKENS * QWEN3VL_PIXELS_PER_VISUAL_TOKEN


def load_model(device: torch.device):
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        max_pixels=QWEN3VL_MAX_PIXELS,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        dtype=torch.bfloat16,
        _attn_implementation="eager",
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


def generate(
    image: Image.Image,
    prompt: str,
    processor,
    model,
    max_new_tokens: int = 512,
    **generate_kwargs,
):
    """
    Generate from multimodal input and return generation-time artifacts.
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

    generation_args = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "output_attentions": True,
        "output_hidden_states": True,
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    if hasattr(getattr(model, "generation_config", None), "output_logits"):
        generation_args["output_logits"] = False
    generation_args.update(generate_kwargs)

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_args)

    prompt_length = inputs["input_ids"].shape[1]
    generated_token_ids = outputs.sequences[:, prompt_length:]
    generated_text = processor.batch_decode(
        generated_token_ids,
        skip_special_tokens=True,
    )[0]

    image_mask, text_mask = multimodal_masks(processor, inputs["input_ids"][0])
    sequence_image_mask = torch.zeros_like(outputs.sequences[0], dtype=torch.bool)
    sequence_text_mask = torch.zeros_like(outputs.sequences[0], dtype=torch.bool)
    sequence_generated_mask = torch.zeros_like(outputs.sequences[0], dtype=torch.bool)
    sequence_image_mask[:prompt_length] = image_mask
    sequence_text_mask[:prompt_length] = text_mask
    sequence_generated_mask[prompt_length:] = True
    prompt_attention_mask = inputs.get("attention_mask", None)

    return {
        "generated_text": generated_text,
        "generated_token_ids": generated_token_ids[0],
        "sequences": outputs.sequences[0],
        "scores": getattr(outputs, "scores", None),
        "logits": getattr(outputs, "logits", None),
        "attentions": getattr(outputs, "attentions", None),
        "hidden_states": getattr(outputs, "hidden_states", None),
        "prompt_input_ids": inputs["input_ids"][0],
        "prompt_attention_mask": prompt_attention_mask[0]
        if prompt_attention_mask is not None else None,
        "prompt_image_mask": image_mask,
        "prompt_text_mask": text_mask,
        "sequence_image_mask": sequence_image_mask,
        "sequence_text_mask": sequence_text_mask,
        "sequence_generated_mask": sequence_generated_mask,
        "prompt_length": prompt_length,
    }


def generate_text_only(
    prompt: str,
    processor,
    model,
    max_new_tokens: int = 512,
    **generate_kwargs,
):
    """
    Generate from text-only input and return generation-time artifacts.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(model.device)

    generation_args = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "output_attentions": True,
        "output_hidden_states": True,
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    if hasattr(getattr(model, "generation_config", None), "output_logits"):
        generation_args["output_logits"] = False
    generation_args.update(generate_kwargs)

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_args)

    prompt_length = inputs["input_ids"].shape[1]
    generated_token_ids = outputs.sequences[:, prompt_length:]
    generated_text = processor.batch_decode(
        generated_token_ids,
        skip_special_tokens=True,
    )[0]

    text_mask = text_only_mask(processor, inputs["input_ids"][0])
    image_mask = torch.zeros_like(inputs["input_ids"][0], dtype=torch.bool)
    sequence_image_mask = torch.zeros_like(outputs.sequences[0], dtype=torch.bool)
    sequence_text_mask = torch.zeros_like(outputs.sequences[0], dtype=torch.bool)
    sequence_generated_mask = torch.zeros_like(outputs.sequences[0], dtype=torch.bool)
    sequence_text_mask[:prompt_length] = text_mask
    sequence_generated_mask[prompt_length:] = True
    prompt_attention_mask = inputs.get("attention_mask", None)

    return {
        "generated_text": generated_text,
        "generated_token_ids": generated_token_ids[0],
        "sequences": outputs.sequences[0],
        "scores": getattr(outputs, "scores", None),
        "logits": getattr(outputs, "logits", None),
        "attentions": getattr(outputs, "attentions", None),
        "hidden_states": getattr(outputs, "hidden_states", None),
        "prompt_input_ids": inputs["input_ids"][0],
        "prompt_attention_mask": prompt_attention_mask[0]
        if prompt_attention_mask is not None else None,
        "prompt_image_mask": image_mask,
        "prompt_text_mask": text_mask,
        "sequence_image_mask": sequence_image_mask,
        "sequence_text_mask": sequence_text_mask,
        "sequence_generated_mask": sequence_generated_mask,
        "prompt_length": prompt_length,
    }

def register_intervention(model, intervention):
    orig_get_image_features = model.model.get_image_features

    def apply_intervention(features):
        if features is None:
            return None
        if isinstance(features, tuple):
            return tuple(intervention.reduce_norm(feature) for feature in features)
        if isinstance(features, list):
            return [intervention.reduce_norm(feature) for feature in features]
        return intervention.reduce_norm(features)

    def patch(*args, **kwargs):
        out = orig_get_image_features(*args, **kwargs)
        out.pooler_output = apply_intervention(out.pooler_output)
        out.deepstack_features = apply_intervention(out.deepstack_features)
        return out

    model.model.get_image_features = patch
