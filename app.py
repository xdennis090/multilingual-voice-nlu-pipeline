"""Streamlit-Demo für die Multilingual Voice-NLU Pipeline."""

from __future__ import annotations

import html
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.nlu_intent import DEFAULT_INTENT_LABELS, load_intent_classifier
from src.nlu_ner import SPACY_MODELS, load_spacy_model
from src.pipeline import run_text_nlu_pipeline, run_voice_nlu_pipeline
from src.stt_whisper import load_whisper_model

BASE_DIR = Path(__file__).resolve().parent
REFERENCE_DATA_PATH = BASE_DIR / "data" / "reference_texts.json"
SAMPLE_AUDIO_DIR = BASE_DIR / "data" / "sample_audio"

ENTITY_COLORS: dict[str, str] = {
    "PER": "#fde68a",
    "PERSON": "#fde68a",
    "ORG": "#bfdbfe",
    "LOC": "#bbf7d0",
    "GPE": "#bbf7d0",
    "DATE": "#ddd6fe",
    "TIME": "#e9d5ff",
    "CARDINAL": "#fed7aa",
    "PERCENT": "#fbcfe8",
    "MONEY": "#a7f3d0",
}


st.set_page_config(
    page_title="Multilingual Voice-NLU Pipeline",
    page_icon="🎙️",
    layout="wide",
)


@st.cache_data
def load_reference_data() -> dict[str, Any]:
    """Lädt Referenztexte und Standard-Intents aus data/reference_texts.json."""
    if REFERENCE_DATA_PATH.exists():
        with REFERENCE_DATA_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "texts": {},
        "default_intent_labels": DEFAULT_INTENT_LABELS,
    }


@st.cache_resource(show_spinner="Lade Whisper STT-Modell...")
def get_cached_whisper(model_size: str):
    return load_whisper_model(model_size)


@st.cache_resource(show_spinner="Lade spaCy NER-Sprachmodelle (DE & EN)...")
def get_cached_spacy_models() -> dict[str, Any]:
    return {lang: load_spacy_model(lang) for lang in SPACY_MODELS}


@st.cache_resource(show_spinner="Lade XLM-RoBERTa Zero-Shot-Klassifikator...")
def get_cached_intent_classifier():
    return load_intent_classifier()


def render_highlighted_entities(text: str, entities: list[dict[str, Any]]) -> str:
    """Erzeugt HTML mit farblich markierten Named Entities im Fließtext."""
    if not entities:
        return html.escape(text)

    sorted_ents = sorted(entities, key=lambda e: e["start_char"])
    chunks: list[str] = []
    last_idx = 0

    for ent in sorted_ents:
        start, end = ent["start_char"], ent["end_char"]
        if start < last_idx:
            continue
        chunks.append(html.escape(text[last_idx:start]))

        label = ent["label"]
        bg_color = ENTITY_COLORS.get(label, "#e5e7eb")
        ent_text = html.escape(text[start:end])
        badge = (
            f'<span style="background-color:{bg_color}; color:#111827; '
            f'padding:2px 6px; border-radius:6px; font-weight:500;">'
            f'{ent_text} <sup style="font-size:0.72em; font-weight:700;">{html.escape(label)}</sup>'
            f"</span>"
        )
        chunks.append(badge)
        last_idx = end

    chunks.append(html.escape(text[last_idx:]))
    return "".join(chunks)


def display_pipeline_results(result: dict[str, Any]) -> None:
    """Stellt Transkript, NER-Entitäten und Intent-Vorhersagen strukturiert dar."""
    if result.get("language_warning"):
        st.warning(result["language_warning"])

    timings = result["timings_sec"]
    entities = result["ner"]["entities"]
    predictions = result["intents"]["predictions"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Erkannte Sprache", result["language"].upper())
    col2.metric("Erkannte Entitäten", len(entities))
    col3.metric("Analysierte Sätze", len(predictions))
    col4.metric("Gesamtlaufzeit", f"{timings['total']:.2f} s")

    st.caption(
        f"Laufzeit-Aufschlüsselung — STT (Whisper): `{timings['stt']:.2f}s` | "
        f"NER (spaCy): `{timings['ner']:.3f}s` | "
        f"Intent (XLM-RoBERTa): `{timings['intent']:.2f}s`"
    )

    st.divider()

    if "stt" in result:
        st.subheader("1. Speech-to-Text Transkription (OpenAI Whisper)")
        st.info(result["text"])
        if result["stt"].get("segments"):
            with st.expander("Zeitstempel-Segmente anzeigen"):
                seg_df = pd.DataFrame(result["stt"]["segments"])
                st.dataframe(seg_df, use_container_width=True, hide_index=True)

    step_ner = "2." if "stt" in result else "1."
    st.subheader(f"{step_ner} Named Entity Recognition (spaCy – `{result['ner']['model_name']}`)")

    highlighted_html = render_highlighted_entities(result["text"], entities)
    st.markdown(
        f'<div style="line-height:1.9; font-size:1.05rem; padding:14px; '
        f'border:1px solid #374151; border-radius:8px; margin-bottom:14px;">'
        f"{highlighted_html}</div>",
        unsafe_allow_html=True,
    )

    if entities:
        ner_df = pd.DataFrame(
            [
                {
                    "Entität": e["text"],
                    "Label": e["label"],
                    "Bedeutung": e["description"],
                }
                for e in entities
            ]
        )
        st.dataframe(ner_df, use_container_width=True, hide_index=True)
    else:
        st.caption("Keine Named Entities im Text erkannt.")

    st.divider()

    step_intent = "3." if "stt" in result else "2."
    st.subheader(f"{step_intent} Zero-Shot Intent Recognition (`{result['intents']['model_name']}`)")

    if predictions:
        intent_df = pd.DataFrame(
            [
                {
                    "Satz": p["sentence"],
                    "Erkannter Intent": p["intent"],
                    "Konfidenz": f"{p['confidence'] * 100:.1f} %",
                }
                for p in predictions
            ]
        )
        st.dataframe(intent_df, use_container_width=True, hide_index=True)

        with st.expander("Detaillierte Wahrscheinlichkeitsverteilung pro Satz"):
            for idx, pred in enumerate(predictions, start=1):
                st.markdown(f"**Satz {idx}:** _{pred['sentence']}_")
                scores_series = pd.Series(pred["all_scores"]).sort_values(ascending=False)
                st.bar_chart(scores_series, height=180)


def main() -> None:
    ref_data = load_reference_data()
    default_labels = ref_data.get("default_intent_labels", DEFAULT_INTENT_LABELS)

    st.title("Multilingual Voice-NLU Pipeline")
    st.markdown(
        "End-to-End Sprachverarbeitung für **Deutsch & Englisch**: "
        "**OpenAI Whisper** (STT) → **spaCy** (NER) → **XLM-RoBERTa** (Zero-Shot Intent Recognition)."
    )

    with st.sidebar:
        st.header("Pipeline-Konfiguration")

        whisper_size = st.selectbox(
            "Whisper Modellgröße",
            options=["tiny", "base", "small"],
            index=1,
            help="'base' bietet den besten Kompromiss aus Geschwindigkeit und Genauigkeit.",
        )

        lang_choice = st.selectbox(
            "Sprachvorgabe",
            options=["Auto-Detect", "de", "en"],
            format_func=lambda x: {
                "Auto-Detect": "Automatisch erkennen",
                "de": "Deutsch (de)",
                "en": "Englisch (en)",
            }[x],
        )
        selected_lang = None if lang_choice == "Auto-Detect" else lang_choice

        st.subheader("Zero-Shot Intent-Labels")
        labels_raw = st.text_area(
            "Kandidaten-Intents (kommagetrennt)",
            value=", ".join(default_labels),
            height=130,
            help="Beliebige englische oder deutsche Intent-Kategorien ohne Nachtraining.",
        )
        candidate_labels = [lbl.strip() for lbl in labels_raw.split(",") if lbl.strip()]

    tab_audio, tab_text = st.tabs(["Audio-Analyse (End-to-End)", "Text-Analyse (NLU-Test)"])

    with tab_audio:
        sample_files = (
            sorted(
                [
                    p
                    for p in SAMPLE_AUDIO_DIR.iterdir()
                    if p.suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
                ]
            )
            if SAMPLE_AUDIO_DIR.exists()
            else []
        )

        uploaded_audio = st.file_uploader(
            "Audiodatei hochladen (.wav, .mp3, .m4a, .ogg)",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
        )

        selected_sample: Path | None = None
        if sample_files:
            sample_names = ["— Keine Beispiel-Datei —"] + [p.name for p in sample_files]
            chosen_sample = st.selectbox("Oder vorhandene Beispiel-Audiodatei wählen:", sample_names)
            if chosen_sample != "— Keine Beispiel-Datei —":
                selected_sample = SAMPLE_AUDIO_DIR / chosen_sample

        if uploaded_audio is not None:
            st.audio(uploaded_audio)
        elif selected_sample is not None:
            st.audio(str(selected_sample))

        if st.button("Voice-NLU Pipeline starten", type="primary", use_container_width=True):
            if uploaded_audio is None and selected_sample is None:
                st.warning("Bitte lade zuerst eine Audiodatei hoch oder wähle ein Beispiel aus.")
            elif not candidate_labels:
                st.warning("Bitte gib mindestens ein Intent-Label in der Seitenleiste an.")
            else:
                whisper_model = get_cached_whisper(whisper_size)
                spacy_models = get_cached_spacy_models()
                intent_clf = get_cached_intent_classifier()

                if uploaded_audio is not None:
                    suffix = Path(uploaded_audio.name).suffix or ".wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_audio.getbuffer())
                        tmp_path = Path(tmp_file.name)
                    try:
                        with st.spinner("Transkribiere Audio und führe NLU-Analyse durch..."):
                            result = run_voice_nlu_pipeline(
                                audio_path=tmp_path,
                                language=selected_lang,
                                whisper_size=whisper_size,
                                candidate_labels=candidate_labels,
                                whisper_model=whisper_model,
                                spacy_nlp_map=spacy_models,
                                intent_classifier=intent_clf,
                            )
                        display_pipeline_results(result)
                    finally:
                        tmp_path.unlink(missing_ok=True)
                elif selected_sample is not None:
                    with st.spinner("Transkribiere Beispiel-Audio und führe NLU-Analyse durch..."):
                        result = run_voice_nlu_pipeline(
                            audio_path=selected_sample,
                            language=selected_lang,
                            whisper_size=whisper_size,
                            candidate_labels=candidate_labels,
                            whisper_model=whisper_model,
                            spacy_nlp_map=spacy_models,
                            intent_classifier=intent_clf,
                        )
                    display_pipeline_results(result)

    with tab_text:
        texts_dict = ref_data.get("texts", {})
        preset_options = {
            "custom": "Eigener Text",
            **{k: v["title"] for k, v in texts_dict.items()},
        }
        chosen_preset = st.selectbox(
            "Beispieltext aus Versuch 4/5 laden:",
            options=list(preset_options.keys()),
            format_func=lambda k: preset_options[k],
            index=1 if "de" in preset_options else 0,
        )

        default_text = (
            texts_dict[chosen_preset]["text"] if chosen_preset in texts_dict else ""
        )
        default_text_lang = chosen_preset if chosen_preset in {"de", "en"} else "de"

        text_lang = st.radio(
            "Sprache für spaCy NER:",
            options=["de", "en"],
            index=0 if default_text_lang == "de" else 1,
            horizontal=True,
        )
        user_text = st.text_area("Eingabetext", value=default_text, height=160)

        if st.button("NLU-Analyse ausführen", type="primary", use_container_width=True):
            if not user_text.strip():
                st.warning("Bitte gib einen Text ein.")
            elif not candidate_labels:
                st.warning("Bitte gib mindestens ein Intent-Label in der Seitenleiste an.")
            else:
                spacy_models = get_cached_spacy_models()
                intent_clf = get_cached_intent_classifier()
                with st.spinner("Analysiere Entitäten und Intents..."):
                    result = run_text_nlu_pipeline(
                        text=user_text,
                        language=text_lang,
                        candidate_labels=candidate_labels,
                        spacy_nlp=spacy_models[text_lang],
                        intent_classifier=intent_clf,
                    )
                display_pipeline_results(result)


if __name__ == "__main__":
    main()