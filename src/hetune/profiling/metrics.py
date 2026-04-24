from __future__ import annotations

import numpy as np


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    predictions = logits.argmax(axis=-1)
    return float((predictions == labels).mean())


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def logit_kl(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> float:
    p = softmax(reference_logits)
    q = softmax(candidate_logits)
    eps = 1e-9
    kl = p * (np.log(p + eps) - np.log(q + eps))
    return float(kl.sum(axis=-1).mean())


def label_flip_rate(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> float:
    ref = reference_logits.argmax(axis=-1)
    cand = candidate_logits.argmax(axis=-1)
    return float((ref != cand).mean())
