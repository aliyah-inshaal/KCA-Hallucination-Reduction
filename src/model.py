import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


def load_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    return model


def build_lora_config(lora_cfg):
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )


def prepare_lora_model(model, lora_cfg):
    model = prepare_model_for_kbit_training(model)
    lora_config = build_lora_config(lora_cfg)
    model = get_peft_model(model, lora_config)
    return model


def load_finetuned_model(base_model_name, adapter_dir):
    base_model = load_base_model(base_model_name)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    tokenizer = load_tokenizer(adapter_dir)

    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    instruction,
    max_new_tokens=256,
    temperature=0.3,
    top_p=0.9,
    refusal_aware=True
):
    if refusal_aware:
        prompt = f"""[INST]
Answer accurately. If the answer is uncertain or not supported by known facts, say "I am not sure" instead of guessing.

Question: {instruction}
[/INST]"""
    else:
        prompt = f"[INST] {instruction} [/INST]"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()

    return decoded