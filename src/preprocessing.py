"""Text preprocessing helpers for medical simplification."""

from __future__ import annotations

import re


MEDICAL_ABBREVS = {
    "ards": "acute respiratory distress syndrome",
    "hx": "history",
    "dx": "diagnosis",
    "tx": "treatment",
    "rx": "prescription",
    "pt": "patient",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "sob": "shortness of breath",
    "cp": "chest pain",
    "bid": "twice daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "prn": "as needed",
    "po": "by mouth",
    "iv": "intravenous",
    "o2": "oxygen",
    "bp": "blood pressure",
    "hr": "heart rate",
    "rr": "respiratory rate",
}

_SENTENCE_ABBR_PATTERN = re.compile(r"(Dr|Mr|Mrs|Ms|Prof|Fig|vs|etc|No)\.\s")


def expand_medical_abbreviations(text: str) -> str:
    expanded = text
    for abbr, expansion in MEDICAL_ABBREVS.items():
        expanded = re.sub(rf"\b{re.escape(abbr)}\b", expansion, expanded, flags=re.IGNORECASE)
    return expanded


def normalize_text(text: str) -> str:
    normalized = expand_medical_abbreviations(text.lower().strip())
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def split_sentences(text: str) -> list[str]:
    protected = _SENTENCE_ABBR_PATTERN.sub(r"\1<PERIOD> ", text)
    sentences = re.split(r"(?<=[.!?])\s+", protected)
    return [sentence.replace("<PERIOD>", ".").strip() for sentence in sentences if sentence.strip()]


def extract_quoted_text(question: str) -> str | None:
    match = re.search(r"'(.+?)'", question)
    return match.group(1).strip() if match else None


def prepare_source_text(note: str, question: str | None, task: str) -> str:
    if task == "Paraphrasing" and question:
        quoted = extract_quoted_text(question)
        if quoted:
            return quoted
    return note
