"""Erzeugt die standardisierten Referenz-Audiodateien (16 kHz, Mono, 16-Bit PCM WAV)
aus data/reference_texts.json mithilfe von Edge-TTS und ffmpeg.

Ausführung aus dem Projekt-Root:
    python scripts/generate_sample_audio.py
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import wave
from pathlib import Path

import edge_tts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_TEXTS_PATH = PROJECT_ROOT / "data" / "reference_texts.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sample_audio"

VOICE_CONFIG: dict[str, tuple[str, str]] = {
    "de": ("de-DE-KatjaNeural", "sample_de.wav"),
    "en": ("en-US-AriaNeural", "sample_en.wav"),
}


def resolve_ffmpeg_executable() -> str:
    """Sucht ffmpeg im System-PATH oder optional über imageio-ffmpeg."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg wurde nicht gefunden. Bitte installiere ffmpeg über den System-Paketmanager "
            "oder per 'pip install imageio-ffmpeg'."
        ) from exc


def convert_to_pcm_wav(ffmpeg_exe: str, source_mp3: Path, target_wav: Path) -> None:
    """Konvertiert den Edge-TTS-MP3-Stream in 16 kHz, Mono, 16-Bit PCM WAV."""
    cmd = [
        ffmpeg_exe,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_mp3),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(target_wav),
    ]
    subprocess.run(cmd, check=True)


def verify_wav(wav_path: Path) -> tuple[int, int, float]:
    """Prüft die erzeugte WAV-Datei und gibt (Sample-Rate, Kanäle, Dauer in s) zurück."""
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        duration = round(wf.getnframes() / float(rate), 2)

    if (rate, channels, sampwidth) != (16000, 1, 2):
        raise ValueError(
            f"Unerwartetes WAV-Format in {wav_path}: {rate} Hz, {channels} Kanal/Kanäle, {sampwidth * 8}-Bit"
        )
    return rate, channels, duration


async def generate_all() -> None:
    if not REFERENCE_TEXTS_PATH.exists():
        raise FileNotFoundError(f"Referenzdatei nicht gefunden: {REFERENCE_TEXTS_PATH}")

    with REFERENCE_TEXTS_PATH.open("r", encoding="utf-8") as f:
        ref_data = json.load(f)

    texts: dict[str, str] = ref_data.get("texts", {})
    ffmpeg_exe = resolve_ffmpeg_executable()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for lang_key, (voice_name, target_filename) in VOICE_CONFIG.items():
        raw_entry = texts.get(lang_key)
        if isinstance(raw_entry, dict):
            text = (
                raw_entry.get("text")
                or raw_entry.get("full_text")
                or next((v for v in raw_entry.values() if isinstance(v, str)), "")
            )
        else:
            text = raw_entry or ""

        if not text.strip():
            print(f"Überspringe '{lang_key}': Kein Text in {REFERENCE_TEXTS_PATH.name} gefunden.")
            continue

        tmp_mp3 = OUTPUT_DIR / f".tmp_{lang_key}.mp3"
        target_wav = OUTPUT_DIR / target_filename

        try:
            communicate = edge_tts.Communicate(text.strip(), voice=voice_name)
            await communicate.save(str(tmp_mp3))
            convert_to_pcm_wav(ffmpeg_exe, tmp_mp3, target_wav)
            rate, channels, duration = verify_wav(target_wav)
            size_kb = round(target_wav.stat().st_size / 1024, 1)
            print(
                f"Erstellt: {target_wav.relative_to(PROJECT_ROOT)} "
                f"({voice_name} | {rate} Hz, {channels}ch, {duration}s, {size_kb} KB)"
            )
        finally:
            if tmp_mp3.exists():
                tmp_mp3.unlink()


if __name__ == "__main__":
    asyncio.run(generate_all())