import argparse

from src.utils import load_config
from src.model import load_finetuned_model, generate_answer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_refusal",
        help="Path to fine-tuned LoRA adapter checkpoint"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User question or instruction"
    )

    args = parser.parse_args()

    config = load_config("config.yaml")

    print("Loading model...")
    model, tokenizer = load_finetuned_model(
        base_model_name=config["base_model"],
        adapter_dir=args.checkpoint
    )

    print("Generating answer...")
    answer = generate_answer(
        model=model,
        tokenizer=tokenizer,
        instruction=args.prompt,
        max_new_tokens=config["generation"]["max_new_tokens"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
        refusal_aware=config["generation"]["refusal_aware"]
    )

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()