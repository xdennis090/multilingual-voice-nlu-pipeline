"""Zero-Shot Intent Recognition Modul auf Basis von XLM-RoBERTa."""

from __future__ import annotations

import re
import time
from typing import Any, Optional

import torch
from transformers import Pipeline, pipeline

INTENT_MODEL_NAME = "joeddav/xlm-roberta-large-xnli"

DEFAULT_INTENT_LABELS: list[str] = [
    "greeting",
    "presentation",
    "report",
    "system failure",
    "question",
    "reassurance",
    "reaction",
    "context",
    "reflection",
]


def load_intent_classifier(model_name: str = INTENT_MODEL_NAME) -> Pipeline:
    """Lädt die Zero-Shot-Klassifikations-Pipeline auf GPU (falls vorhanden) oder CPU."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )


def _fallback_sentence_split(text: str) -> list[str]:
    """Einfache Satztrennung als Fallback, falls keine spaCy-Sätze übergeben wurden."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def classify_intents(
    text: str,
    candidate_labels: Optional[list[str]] = None,
    sentences: Optional[list[str]] = None,
    classifier: Optional[Pipeline] = None,
) -> dict[str, Any]:
    """
    Klassifiziert die Intention jedes einzelnen Satzes per Zero-Shot-Learning.

    Args:
        text: Gesamter Eingabetext.
        candidate_labels: Liste möglicher Intents (Standard: DEFAULT_INTENT_LABELS).
        sentences: Optional bereits segmentierte Einzelsätze (z. B. aus spaCy).
        classifier: Optional bereits vorgeladene Transformers-Pipeline.

    Returns:
        Dictionary mit Satz-Ergebnissen, verwendeten Labels, Modellname und Laufzeit.
    """
    labels = [lbl.strip() for lbl in (candidate_labels or DEFAULT_INTENT_LABELS) if lbl.strip()]
    if not labels:
        raise ValueError("Es muss mindestens ein Intent-Label angegeben werden.")

    sentence_list = sentences if sentences is not None else _fallback_sentence_split(text)
    if not sentence_list:
        return {
            "model_name": INTENT_MODEL_NAME,
            "candidate_labels": labels,
            "predictions": [],
            "duration_sec": 0.0,
        }

    clf = classifier if classifier is not None else load_intent_classifier()

    start_time = time.perf_counter()
    raw_outputs = clf(
        sentence_list,
        labels,
        hypothesis_template="This sentence is about {}.",
    )
    duration_sec = round(time.perf_counter() - start_time, 4)

    if isinstance(raw_outputs, dict):
        raw_outputs = [raw_outputs]

    predictions = []
    for sentence, output in zip(sentence_list, raw_outputs):
        top_label = output["labels"][0]
        top_score = round(float(output["scores"][0]), 4)
        all_scores = {
            lbl: round(float(scr), 4)
            for lbl, scr in zip(output["labels"], output["scores"])
        }
        predictions.append(
            {
                "sentence": sentence,
                "intent": top_label,
                "confidence": top_score,
                "all_scores": all_scores,
            }
        )

    return {
        "model_name": INTENT_MODEL_NAME,
        "candidate_labels": labels,
        "predictions": predictions,
        "duration_sec": duration_sec,
    }