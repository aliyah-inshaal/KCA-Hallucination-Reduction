import json
import yaml
from pathlib import Path


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data, path):
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