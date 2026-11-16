"""Backend-Paket für die Multilingual Voice-NLU Pipeline."""

from src.nlu_intent import DEFAULT_INTENT_LABELS, classify_intents, load_intent_classifier
from src.nlu_ner import SPACY_MODELS, extract_entities, load_spacy_model
from src.pipeline import run_text_nlu_pipeline, run_voice_nlu_pipeline
from src.stt_whisper import load_whisper_model, transcribe_audio

__all__ = [
    "DEFAULT_INTENT_LABELS",
    "SPACY_MODELS",
    "classify_intents",
    "extract_entities",
    "load_intent_classifier",
    "load_spacy_model",
    "load_whisper_model",
    "run_text_nlu_pipeline",
    "run_voice_nlu_pipeline",
    "transcribe_audio",
]