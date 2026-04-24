from __future__ import annotations

import argparse
import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned DistilBERT directly on GLUE/SST-2."
    )
    parser.add_argument("--split", default="validation", help="Dataset split to evaluate.")
    parser.add_argument("--max-samples", type=int, default=512, help="Limit sample count; use 0 for full split.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    dataset = load_dataset("glue", "sst2", split=args.split)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=args.sequence_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized = tokenized.add_column("labels", dataset["label"])
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    loader = DataLoader(tokenized, batch_size=args.batch_size)
    correct = 0
    total = 0
    loss_sum = 0.0
    start = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch, labels=labels)
            predictions = output.logits.argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
            loss_sum += float(output.loss.item()) * int(labels.numel())

    elapsed = time.perf_counter() - start
    accuracy = correct / total if total else 0.0
    avg_loss = loss_sum / total if total else 0.0

    print(f"model: {MODEL_NAME}")
    print(f"split: {args.split}")
    print(f"device: {device}")
    print(f"samples: {total}")
    print(f"correct: {correct}")
    print(f"accuracy: {accuracy:.6f}")
    print(f"avg_loss: {avg_loss:.6f}")
    print(f"elapsed_seconds: {elapsed:.3f}")
    print(f"samples_per_second: {total / elapsed:.2f}")


if __name__ == "__main__":
    main()
