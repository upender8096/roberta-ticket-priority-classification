# Fine-Tuning RoBERTa for Customer Support Ticket Priority Classification

## Project Overview

This project investigates the use of a transformer-based Large Language Model (LLM), RoBERTa, for automatically classifying customer support tickets according to their priority level.

The task is formulated as a multi-class text classification problem with four categories:

- Low
- Medium
- High
- Critical

The project compares a traditional machine learning baseline model with pre-trained and fine-tuned RoBERTa models to evaluate the effectiveness of transformer-based approaches for customer support ticket classification.

---

## Objectives

The objectives of this project are:

- Explore and preprocess a customer support ticket dataset.
- Develop a traditional machine learning baseline using TF-IDF and Logistic Regression.
- Fine-tune a pre-trained RoBERTa model for ticket priority classification.
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Analyse the limitations affecting model performance.

---

## Dataset

Dataset:
Customer Support Tickets Dataset from Hugging Face Datasets

Dataset size:
- Total samples: 8,469 customer support tickets

Features used:
- Input: Combined ticket text
- Target: Ticket Priority

Classes:

| Label | Priority |
|---|---|
| 0 | Low |
| 1 | Medium |
| 2 | High |
| 3 | Critical |

---

## Methodology

The project pipeline consists of the following stages:

1. Exploratory Data Analysis (EDA)
2. Data preprocessing
3. Train-test data splitting
4. Baseline model development
5. RoBERTa tokenisation
6. Transformer fine-tuning
7. Performance evaluation

---

## Models Implemented

### Baseline Model

**TF-IDF + Logistic Regression**

TF-IDF was used to convert ticket text into numerical representations before training a Logistic Regression classifier.

### Transformer Model

**RoBERTa-base**

A pre-trained RoBERTa model was fine-tuned using:

- Hugging Face Transformers
- PyTorch
- AdamW optimiser

Configuration:

- Epochs: 4
- Learning rate: 5e-5
- Task: Four-class sequence classification

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

---

## Results Summary

| Model | Accuracy |
|---|---:|
| TF-IDF + Logistic Regression | 0.2556 |
| Pre-trained RoBERTa | 0.2509 |
| Fine-tuned RoBERTa | 0.2586 |

The results indicate that fine-tuning RoBERTa provided limited improvement compared with the baseline. Further analysis suggested that dataset characteristics, including weak relationships between ticket text and priority labels, influenced performance.

---

## Repository Contents
