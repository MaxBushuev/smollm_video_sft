from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch

model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to("cuda")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/mnt/d/hse/3ddzkmFPEBU_4-1-rgb_front.mp4"},
            {"type": "text", "text": "Translate the video from american sign language to english."}
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])