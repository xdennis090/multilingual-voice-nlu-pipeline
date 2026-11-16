"""End-to-End Voice-NLU Pipeline (Whisper STT -> spaCy NER -> XLM-RoBERTa Intent)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import whisper
from spacy.language import Language
from transformers import Pipeline

from src.nlu_intent import DEFAULT_INTENT_LABELS, classify_intents
from src.nlu_ner import SPACY_MODELS, extract_entities
from src.stt_whisper import transcribe_audio


def run_text_nlu_pipeline(
    text: str,
    language: str = "de",
    candidate_labels: Optional[list[str]] = None,
    spacy_nlp: Optional[Language] = None,
    intent_classifier: Optional[Pipeline] = None,
) -> dict[str, Any]:
    """
    Führt die NLU-Analyse (NER + Zero-Shot Intent) auf einem vorliegenden Text aus.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Der Eingabetext darf nicht leer sein.")

    lang = language.lower().strip()
    language_warning: Optional[str] = None
    if lang not in SPACY_MODELS:
        language_warning = (
            f"Erkannte Sprache '{lang}' wird von den lokalen spaCy-Modellen nicht unterstützt. "
            "Fallback auf Englisch ('en') für NER."
        )
        lang = "en"

    start_total = time.perf_counter()

    ner_result = extract_entities(
        text=cleaned_text,
        language=lang,
        nlp=spacy_nlp,
    )

    intent_result = classify_intents(
        text=cleaned_text,
        candidate_labels=candidate_labels or DEFAULT_INTENT_LABELS,
        sentences=ner_result["sentences"],
        classifier=intent_classifier,
    )

    total_duration = round(time.perf_counter() - start_total, 4)

    return {
        "text": cleaned_text,
        "language": lang,
        "language_warning": language_warning,
        "ner": ner_result,
        "intents": intent_result,
        "timings_sec": {
            "stt": 0.0,
            "ner": ner_result["duration_sec"],
            "intent": intent_result["duration_sec"],
            "total": total_duration,
        },
    }


def run_voice_nlu_pipeline(
    audio_path: str | Path,
    language: Optional[str] = None,
    whisper_size: str = "base",
    candidate_labels: Optional[list[str]] = None,
    whisper_model: Optional[whisper.Whisper] = None,
    spacy_nlp_map: Optional[dict[str, Language]] = None,
    intent_classifier: Optional[Pipeline] = None,
) -> dict[str, Any]:
    """
    Führt die vollständige Audio-to-NLU Pipeline aus:
    1. Audio-Transkription mit Whisper
    2. Named Entity Recognition mit spaCy
    3. Zero-Shot Intent-Klassifikation mit XLM-RoBERTa
    """
    start_total = time.perf_counter()

    stt_result = transcribe_audio(
        audio_path=audio_path,
        language=language,
        model_size=whisper_size,
        model=whisper_model,
    )

    detected_lang = (language or stt_result["language"] or "en").lower().strip()
    spacy_nlp = spacy_nlp_map.get(detected_lang) if spacy_nlp_map else None

    nlu_result = run_text_nlu_pipeline(
        text=stt_result["text"],
        language=detected_lang,
        candidate_labels=candidate_labels,
        spacy_nlp=spacy_nlp,
        intent_classifier=intent_classifier,
    )

    total_duration = round(time.perf_counter() - start_total, 4)
    nlu_result["stt"] = stt_result
    nlu_result["timings_sec"]["stt"] = stt_result["duration_sec"]
    nlu_result["timings_sec"]["total"] = total_duration

    return nlu_result