from __future__ import annotations

from typing import Any

from hetune.models.hf_adapter import HFSequenceClassifierAdapter


def load_tokenized_dataset(
    dataset_config: dict[str, Any],
    adapter: HFSequenceClassifierAdapter,
    split_key: str,
    sample_size: int | None,
    max_length: int,
):
    from datasets import load_dataset

    dataset_name = dataset_config["dataset_name"]
    dataset_subset = dataset_config.get("dataset_subset")
    split = dataset_config[split_key]
    if dataset_subset:
        dataset = load_dataset(dataset_name, dataset_subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    if sample_size is not None:
        sample_size = min(sample_size, len(dataset))
        dataset = dataset.select(range(sample_size))

    text_fields = list(dataset_config["text_fields"])
    label_column = dataset_config.get("label_column", "label")

    def tokenize(examples):
        return adapter.tokenize_batch(examples, text_fields, max_length)

    columns_to_remove = list(dataset.column_names)
    tokenized = dataset.map(tokenize, batched=True, remove_columns=columns_to_remove)
    tokenized = tokenized.add_column("labels", dataset[label_column])
    torch_columns = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized.column_names:
        torch_columns.append("token_type_ids")
    tokenized.set_format("torch", columns=torch_columns)
    return tokenized
