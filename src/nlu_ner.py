"""Named Entity Recognition (NER) Modul auf Basis von spaCy."""

from __future__ import annotations

import time
from typing import Any, Optional

import spacy
from spacy.language import Language

SPACY_MODELS: dict[str, str] = {
    "de": "de_core_news_sm",
    "en": "en_core_web_sm",
}


def load_spacy_model(language: str = "de") -> Language:
    """Lädt das passende spaCy-Sprachmodell für Deutsch ('de') oder Englisch ('en')."""
    lang = language.lower().strip()
    if lang not in SPACY_MODELS:
        raise ValueError(
            f"Nicht unterstützte Sprache für NER: '{language}'. Erlaubt sind: {list(SPACY_MODELS.keys())}"
        )

    model_name = SPACY_MODELS[lang]
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy-Modell '{model_name}' ist nicht installiert. "
            f"Bitte mit 'python -m spacy download {model_name}' herunterladen."
        ) from exc


def extract_entities(
    text: str,
    language: str = "de",
    nlp: Optional[Language] = None,
) -> dict[str, Any]:
    """
    Erkennt Named Entities und Sätze in einem deutschen oder englischen Text.

    Args:
        text: Zu analysierender Eingabetext.
        language: Sprachcode ('de' oder 'en').
        nlp: Optional bereits vorgeladene spaCy-Pipeline.

    Returns:
        Dictionary mit erkannten Entitäten, Einzelsätzen, Modellname und Laufzeit.
    """
    lang = language.lower().strip()
    model_name = SPACY_MODELS.get(lang, "unknown")
    nlp_pipeline = nlp if nlp is not None else load_spacy_model(lang)

    start_time = time.perf_counter()
    doc = nlp_pipeline(text)
    duration_sec = round(time.perf_counter() - start_time, 4)

    entities = [
        {
            "text": ent.text,
            "label": ent.label_,
            "description": spacy.explain(ent.label_) or "Keine Beschreibung verfügbar",
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        }
        for ent in doc.ents
    ]

    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    return {
        "language": lang,
        "model_name": model_name,
        "entities": entities,
        "sentences": sentences,
        "duration_sec": duration_sec,
    }