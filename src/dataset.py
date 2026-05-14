import json
import yaml
from pathlib import Path


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data, path):import json
import pandas as pd
from collections import defaultdict
from datasets import Dataset


def format_instruction_response(instruction, response, knowledge="", strategy="standard"):
    instruction = str(instruction).strip()
    response = str(response).strip()
    knowledge = str(knowledge).strip()

    if strategy == "open_book" and knowledge:
        instruction = f"Reference knowledge: {knowledge}\n\nQuestion: {instruction}"

    instruction = instruction[:1200]
    response = response[:800]

    return {
        "text": f"[INST] {instruction} [/INST] {response} </s>"
    }


def load_csv_dataset(path, strategy="standard"):
    df = pd.read_csv(path)

    formatted = []

    for _, row in df.iterrows():
        formatted.append(
            format_instruction_response(
                instruction=row.get("instruction", ""),
                response=row.get("response", ""),
                knowledge=row.get("knowledge", ""),
                strategy=strategy
            )
        )

    return Dataset.from_list(formatted)


def load_json_dataset(path, strategy="standard", samples_per_class=500):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    buckets = defaultdict(list)

    for item in raw:
        conversations = item.get("conversations", [])

        if len(conversations) < 2:
            continue

        cls = item.get("class", "unknown")
        buckets[cls].append(item)

    sampled = []

    for cls, items in buckets.items():
        sampled.extend(items[:samples_per_class])

    formatted = []

    for item in sampled:
        conversations = item["conversations"]

        instruction = conversations[0]["value"]
        response = conversations[1]["value"]
        knowledge = item.get("knowledge", "")

        formatted.append(
            format_instruction_response(
                instruction=instruction,
                response=response,
                knowledge=knowledge,
                strategy=strategy
            )
        )

    return Dataset.from_list(formatted)


def load_training_dataset(path, strategy="standard", samples_per_class=500):
    if path.endswith(".csv"):
        return load_csv_dataset(path, strategy=strategy)

    if path.endswith(".json"):
        return load_json_dataset(
            path,
            strategy=strategy,
            samples_per_class=samples_per_class
        )

    raise ValueError("Unsupported data format. Use .csv or .json")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_refusal(text):
    refusal_phrases = [
        "i am not sure",
        "i don't know",
        "cannot answer",
        "not enough information",
        "unknown",
        "not supported"
    ]

    text = text.lower()
    return any(phrase in text for phrase in refusal_phrases)