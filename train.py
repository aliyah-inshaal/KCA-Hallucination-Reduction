from transformers import TrainingArguments
from trl import SFTTrainer

from src.utils import load_config
from src.dataset import load_training_dataset
from src.model import load_base_model, load_tokenizer, prepare_lora_model


def main():
    config = load_config("config.yaml")

    base_model_name = config["base_model"]
    training_cfg = config["training"]
    lora_cfg = config["lora"]

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(base_model_name)

    print("Loading training dataset...")
    dataset = load_training_dataset(
        training_cfg["train_file"],
        strategy=training_cfg["strategy"],
        samples_per_class=training_cfg["samples_per_class"]
    )

    print("Loading base model...")
    model = load_base_model(base_model_name)

    print("Applying LoRA...")
    model = prepare_lora_model(model, lora_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        warmup_ratio=training_cfg["warmup_ratio"],
        logging_steps=training_cfg["logging_steps"],
        save_steps=training_cfg["save_steps"],
        save_total_limit=1,
        fp16=True,
        bf16=False,
        report_to="none",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=training_cfg["max_seq_length"],
        args=args
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.model.save_pretrained(training_cfg["output_dir"])
    tokenizer.save_pretrained(training_cfg["output_dir"])

    print(f"Training complete. Model saved to {training_cfg['output_dir']}")


if __name__ == "__main__":
    main()