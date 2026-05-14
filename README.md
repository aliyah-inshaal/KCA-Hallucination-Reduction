# KCA Hallucination Reduction

This repository contains the reproduction and extension of the paper:

**Knowledge Verification to Nip Hallucination in the Bud**

The project focuses on reducing hallucinations in Large Language Models using the KCA framework and additional refusal-aware improvements.

---

## Project Overview

Large Language Models often generate confident but incorrect answers when they do not have enough factual knowledge. This project reproduces the baseline KCA approach and extends it by adding extra evaluation and inference improvements.

The work is divided into two parts:

1. **Baseline Reproduction**
   - Reproduces the original KCA-based hallucination reduction setup.
   - Uses different training strategies such as standard, open-book, discard, and refusal.

2. **Extension / Improved Work**
   - Adds HaluEval evaluation.
   - Uses refusal-aware prompting.
   - Applies low-temperature decoding.
   - Reports refusal rate and average response length.
   - Compares baseline and improved model behavior.

---

## Repository Structure

```text
project-root/
│── README.md
│── requirements.txt
│── train.py
│── inference.py
│── config.yaml
│── .gitignore
│
├── data/
│   └── sample_data.csv
│
├── notebooks/
│   ├── 01_inference_demo.ipynb
│   ├── 02_reproduction_baseline.ipynb
│   └── 03_extension_improved.ipynb
│
├── src/
│   ├── model.py
│   ├── dataset.py
│   └── utils.py
│
├── results/
│   ├── baseline_metrics.json
│   ├── improved_metrics.json
│   ├── training_log.csv
│   └── eval_outputs/
│
└── checkpoints/
    ├── README.md
    ├── model_standard/
    ├── model_open_book/
    ├── model_discard/
    └── model_refusal/
