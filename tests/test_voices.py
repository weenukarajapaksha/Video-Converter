"""Pitch-aware voice casting."""

from dubber.stages.voices import classify_gender, pick_voices, voice_f0


def test_deep_and_high_male_speakers_get_matching_voices():
    profiles = [("deep", "male", 10.0), ("high", "male", 8.0)]
    chosen = pick_voices(profiles, "edge", {"deep": 98.0, "high": 140.0})
    assert chosen["deep"] == "en-US-EricNeural"  # 100.2 Hz, the deepest Edge male
    assert chosen["high"] == "en-US-GuyNeural"  # 142.6 Hz


def test_female_speaker_gets_closest_female_voice_and_voices_are_distinct():
    profiles = [("a", "female", 9.0), ("b", "female", 5.0)]
    chosen = pick_voices(profiles, "edge", {"a": 170.0, "b": 172.0})
    assert chosen["a"] == "en-US-EmmaNeural"  # 169.1 Hz: the closest, for the bigger part
    assert chosen["b"] == "en-US-JennyNeural"  # next closest, never a duplicate
    assert chosen["a"] != chosen["b"]


def test_without_pitch_falls_back_to_catalog_order_deterministically():
    profiles = [("x", "unknown", 3.0), ("y", "female", 3.0)]
    first, second = pick_voices(profiles, "edge"), pick_voices(profiles, "edge")
    assert first == second
    assert first["x"] == "en-US-GuyNeural" and first["y"] == "en-US-AriaNeural"


def test_gender_split_and_catalog_pitch_lookup():
    assert classify_gender(120.0) == "male" and classify_gender(200.0) == "female"
    assert classify_gender(None) == "unknown"
    assert voice_f0("edge", "en-US-AvaNeural") == 197.0
    assert voice_f0("edge", "nope") is None


def test_light_male_voice_is_judged_relative_to_the_other_speaker():
    # Evaluation finding: the male test voice measured 179-206 Hz and a fixed 160 Hz
    # split called him female. Next to a 260 Hz woman he is clearly the lower voice.
    from dubber.stages.voices import classify_genders

    assert classify_genders({"m": 190.0, "f": 260.0}) == {"m": "male", "f": "female"}
    # Clear cases ignore context; unknown pitch stays unknown.
    assert classify_genders({"a": 110.0, "b": 125.0}) == {"a": "male", "b": "male"}
    assert classify_genders({"a": 230.0, "b": None}) == {"a": "female", "b": "unknown"}
    # A lone ambiguous speaker falls back to the fixed split.
    assert classify_genders({"solo": 190.0}) == {"solo": "female"}
