"""Selectable English voices per TTS provider (shared by the voices stage and the API)."""

from __future__ import annotations


def _v(id: str, gender: str, description: str) -> dict:
    return {"id": id, "gender": gender, "description": description}


VOICES: dict[str, list[dict]] = {
    "gemini": [
        _v("Puck", "male", "Upbeat"),
        _v("Charon", "male", "Informative"),
        _v("Fenrir", "male", "Excitable"),
        _v("Orus", "male", "Firm"),
        _v("Enceladus", "male", "Breathy"),
        _v("Iapetus", "male", "Clear"),
        _v("Algenib", "male", "Gravelly"),
        _v("Umbriel", "male", "Easy-going"),
        _v("Kore", "female", "Firm"),
        _v("Aoede", "female", "Breezy"),
        _v("Leda", "female", "Youthful"),
        _v("Zephyr", "female", "Bright"),
        _v("Callirrhoe", "female", "Easy-going"),
        _v("Despina", "female", "Smooth"),
        _v("Erinome", "female", "Clear"),
        _v("Laomedeia", "female", "Upbeat"),
        _v("Achernar", "female", "Soft"),
        _v("Gacrux", "female", "Mature"),
        _v("Sulafat", "female", "Warm"),
        _v("Vindemiatrix", "female", "Gentle"),
    ],
    "edge": [
        _v("en-US-GuyNeural", "male", "Passionate"),
        _v("en-US-AndrewNeural", "male", "Warm, confident"),
        _v("en-US-BrianNeural", "male", "Approachable, casual"),
        _v("en-US-ChristopherNeural", "male", "Reliable, authoritative"),
        _v("en-US-EricNeural", "male", "Rational"),
        _v("en-US-AriaNeural", "female", "Positive, confident"),
        _v("en-US-JennyNeural", "female", "Friendly, considerate"),
        _v("en-US-AvaNeural", "female", "Expressive, caring"),
        _v("en-US-EmmaNeural", "female", "Cheerful, clear"),
        _v("en-US-MichelleNeural", "female", "Friendly, pleasant"),
    ],
}
