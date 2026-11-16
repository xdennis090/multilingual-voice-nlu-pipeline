"""Speech-to-Text (STT) Modul auf Basis von OpenAI Whisper."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import torch
import whisper


def load_whisper_model(model_size: str = "base") -> whisper.Whisper:
    """Lädt ein vortrainiertes Whisper-Modell (z. B. 'tiny', 'base', 'small')."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(model_size, device=device)


def transcribe_audio(
    audio_path: str | Path,
    language: Optional[str] = None,
    model_size: str = "base",
    model: Optional[whisper.Whisper] = None,
) -> dict[str, Any]:
    """
    Transkribiert eine Audiodatei mit OpenAI Whisper.

    Args:
        audio_path: Pfad zur Audiodatei (.wav, .mp3, .m4a, .ogg, ...).
        language: Optionaler Sprachcode ('de' oder 'en'). Bei None erfolgt Auto-Detection.
        model_size: Whisper-Modellgröße (Standard: 'base').
        model: Optional bereits vorgeladene Whisper-Instanz.

    Returns:
        Dictionary mit Transkript, erkannter Sprache, Segmenten und Laufzeit.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audiodatei nicht gefunden: {path}")

    stt_model = model if model is not None else load_whisper_model(model_size)
    use_fp16 = torch.cuda.is_available()

    transcribe_kwargs: dict[str, Any] = {"fp16": use_fp16}
    if language and language.lower() in {"de", "en"}:
        transcribe_kwargs["language"] = language.lower()

    start_time = time.perf_counter()
    raw_result = stt_model.transcribe(str(path), **transcribe_kwargs)
    duration_sec = round(time.perf_counter() - start_time, 4)

    segments = [
        {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        }
        for seg in raw_result.get("segments", [])
    ]

    return {
        "text": raw_result.get("text", "").strip(),
        "language": raw_result.get("language", language or "unknown"),
        "segments": segments,
        "duration_sec": duration_sec,
        "model_size": model_size,
    }