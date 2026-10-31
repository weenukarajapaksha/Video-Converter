"""Selectable English voices per TTS provider (shared by the voices stage and the API).

``f0_hz`` is each voice's median pitch, used to cast a deep-voiced speaker with a
deep English voice. Edge values were measured on 2026-10-31 (pYIN median over a
two-sentence sample). Gemini values are gender-typical estimates, not measurements:
measuring them would spend the limited free-tier TTS quota.
"""

from __future__ import annotations

MALE_F0_ESTIMATE = 120.0
FEMALE_F0_ESTIMATE = 195.0


def _v(id: str, gender: str, description: str, f0_hz: float | None = None) -> dict:
    if f0_hz is None:
        f0_hz = MALE_F0_ESTIMATE if gender == "male" else FEMALE_F0_ESTIMATE
    return {"id": id, "gender": gender, "description": description, "f0_hz": f0_hz}


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
        _v("en-US-GuyNeural", "male", "Passionate, higher", 142.6),
        _v("en-US-AndrewNeural", "male", "Warm, confident", 124.5),
        _v("en-US-BrianNeural", "male", "Approachable, casual", 117.2),
        _v("en-US-ChristopherNeural", "male", "Reliable, authoritative, deep", 103.2),
        _v("en-US-EricNeural", "male", "Rational, deep", 100.2),
        _v("en-US-AriaNeural", "female", "Positive, confident", 194.8),
        _v("en-US-JennyNeural", "female", "Friendly, considerate", 176.6),
        _v("en-US-AvaNeural", "female", "Expressive, caring, bright", 197.0),
        _v("en-US-EmmaNeural", "female", "Cheerful, clear, lower", 169.1),
        _v("en-US-MichelleNeural", "female", "Friendly, pleasant", 186.0),
    ],
}


# Native Edge voices for the non-English targets offered in the UI (Gemini voices are
# multilingual, so the same Gemini catalog serves every language). Pitch values use
# the gender-typical estimates.
EDGE_NATIVE: dict[str, list[dict]] = {
    "Spanish": [_v("es-ES-AlvaroNeural", "male", "Spain"), _v("es-ES-ElviraNeural", "female", "Spain"),
                _v("es-MX-JorgeNeural", "male", "Mexico"), _v("es-MX-DaliaNeural", "female", "Mexico")],
    "French": [_v("fr-FR-HenriNeural", "male", "France"), _v("fr-FR-DeniseNeural", "female", "France")],
    "German": [_v("de-DE-ConradNeural", "male", "Germany"), _v("de-DE-KatjaNeural", "female", "Germany"),
               _v("de-DE-KillianNeural", "male", "Germany"), _v("de-DE-AmalaNeural", "female", "Germany")],
    "Japanese": [_v("ja-JP-KeitaNeural", "male", "Japan"), _v("ja-JP-NanamiNeural", "female", "Japan")],
    "Korean": [_v("ko-KR-InJoonNeural", "male", "Korea"), _v("ko-KR-SunHiNeural", "female", "Korea")],
    "Sinhala": [_v("si-LK-SameeraNeural", "male", "Sri Lanka"), _v("si-LK-ThiliniNeural", "female", "Sri Lanka")],
    "Hindi": [_v("hi-IN-MadhurNeural", "male", "India"), _v("hi-IN-SwaraNeural", "female", "India")],
}
for _lang, _voices in EDGE_NATIVE.items():
    VOICES["edge"] += [v | {"language": _lang} for v in _voices]


def catalog_for(provider: str, language: str = "English") -> list[dict]:
    """Voices that can speak ``language``: Edge voices are per-locale, Gemini's are
    multilingual. Unknown languages fall back to the English Edge voices."""
    voices = VOICES[provider]
    if provider != "edge":
        return voices
    native = [v for v in voices if v.get("language", "English") == language]
    return native or [v for v in voices if v.get("language", "English") == "English"]
