from PIL import Image
from transformers import AutoProcessor

from models import qwen3vl, smolvlm


IMAGE = Image.new("RGB", (224, 224), color=(0, 0, 0))
PROMPT = "What color is this image?"

def test_input_parsing_qwen3vl():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    messages = [{"role": "user", "content": [
        {"type": "image", "image": IMAGE},
        {"type": "text", "text": PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    token_ids = inputs["input_ids"][0]
    image_mask, text_mask = qwen3vl.multimodal_masks(processor, token_ids)

    assert image_mask.sum() == (token_ids == processor.image_token_id).sum().item()

    text_only_messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    text_only_inputs = processor.apply_chat_template(
        text_only_messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    text_only_token_ids = text_only_inputs["input_ids"][0]
    text_only_mask = qwen3vl.text_only_mask(processor, text_only_token_ids)

    assert text_mask.sum() == text_only_mask.sum()

def test_input_parsing_smolvlm():
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")

    messages = [{"role": "user", "content": [
        {"type": "image", "image": IMAGE},
        {"type": "text", "text": PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    token_ids = inputs["input_ids"][0]
    image_mask, text_mask = smolvlm.multimodal_masks(processor, token_ids)

    assert image_mask.sum() == (token_ids == processor.image_token_id).sum().item()

    text_only_messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    text_only_inputs = processor.apply_chat_template(
        text_only_messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    text_only_token_ids = text_only_inputs["input_ids"][0]
    text_only_mask = smolvlm.text_only_mask(processor, text_only_token_ids)

    assert text_mask.sum() == text_only_mask.sum()