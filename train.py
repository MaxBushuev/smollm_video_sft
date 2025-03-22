import os
import os.path as op
import argparse

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset

def main(args):
    ds = load_dataset("csv", delimiter='\t', data_files=args.annotation_path)

    processor = AutoProcessor.from_pretrained(
        args.model_id
    )

    USE_LORA = True
    USE_QLORA = True

    if USE_QLORA or USE_LORA:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
            use_dora=False if USE_QLORA else True,
            init_lora_weights="gaussian"
        )
        lora_config.inference_mode = False
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            quantization_config=bnb_config if USE_QLORA else None,
            device_map="auto"
        )
        model.add_adapter(lora_config)
        model.enable_adapters()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        print(model.get_nb_trainable_parameters())
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        # if you'd like to only fine-tune LLM
        for param in model.model.vision_model.parameters():
            param.requires_grad = False
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]

    def collate_fn(examples):
        instances = []
        for example in examples:
            prompt = example["SENTENCE"]

            user_content = [{"type": "text", "text": "Translate the video from american sign language to english."}]
            user_content.append({"type": "video", "path": op.join(args.video_root, example["SENTENCE_NAME"] + ".mp4")})

            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": f"{prompt}"}]}
            ]

            instance = processor.apply_chat_template(messages, add_generation_prompt=False,
                                                    tokenize=True, return_dict=True, return_tensors="pt").to("cuda").to(model.dtype)
            instances.append(instance)


        input_ids = pad_sequence(
            [inst["input_ids"].squeeze(0) for inst in instances],
            batch_first=True,
            padding_value=processor.tokenizer.pad_token_id
        )
        attention_mask = pad_sequence(
            [inst["attention_mask"].squeeze(0) for inst in instances],
            batch_first=True,
            padding_value=0
        )
        labels = pad_sequence(
            [inst["input_ids"].squeeze(0).clone() for inst in instances],
            batch_first=True,
            padding_value=-100
        )

        labels[labels == image_token_id] = -100

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


        # Step 1: figure out maximum frames, height, width across the batch
        pvs = [inst["pixel_values"].squeeze(0) for inst in instances if "pixel_values" in inst]
        if pvs:  # there is at least one non-None pixel_values
            max_frames = max(pv.shape[0] for pv in pvs)
            max_h = max(pv.shape[-2] for pv in pvs)
            max_w = max(pv.shape[-1] for pv in pvs)
        else:
            max_h = max_w = processor.video_size['longest_edge']
            max_frames = 1

        padded_pixel_values_list = []
        for ex in instances:
            pv = ex.get("pixel_values", None).squeeze(0)

            if pv is None:
                # text-only => fill pixel data + mask with zeros
                shape_pv = (max_frames, 3, max_h, max_w)
                padded_pv = torch.zeros(shape_pv, dtype=torch.float32)
            else:
                f, c, h, w = pv.shape
                # Prepare final storage
                padded_pv = torch.zeros(
                    (max_frames, c, max_h, max_w),
                    dtype=pv.dtype,
                    device=pv.device
                )
                padded_pv[:f, :, :h, :w] = pv
            padded_pixel_values_list.append(padded_pv)

        out["pixel_values"] = torch.stack(padded_pixel_values_list, dim=0)
        return out

    model_name = args.model_id.split("/")[-1]

    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=1,
        optim="adamw_torch",
        bf16=True,
        output_dir=f"{args.output_path}/{model_name}_asl",
        remove_unused_columns=False,
        report_to="tensorboard",
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=ds["train"],
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--output_path", default=".")

    args = parser.parse_args()
    assert op.exists(args.annotation_path)
    assert op.exists(args.video_root)

    main(args)