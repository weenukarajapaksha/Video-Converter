"""File names of every artifact a job directory can contain (single source of truth)."""

INPUT_VIDEO = "input.mp4"
METADATA = "metadata.json"  # VideoMetadata
ORIGINAL_AUDIO = "audio_original.wav"  # 44.1 kHz stereo
VOCALS = "vocals.wav"  # Demucs speech stem
BACKGROUND = "background.wav"  # Demucs music + SFX stem
TRANSCRIPT = "transcript.json"  # Transcript (Chinese, diarized)
VOICES = "voices.json"  # VoiceMap
TRANSLATION = "translation.json"  # Translation
TTS_DIR = "tts"  # per-line synthesized clips
DUB_TRACK = "dub_track.json"  # DubTrack
DUB_VOICE = "dub_voice.wav"  # English speech only, on the video timeline
DUB_MIX = "dub_mix.wav"  # English speech + background
SUBTITLES_EN = "subtitles_en.srt"
SUBTITLES_ZH = "subtitles_zh.srt"
OUTPUT_VIDEO = "output.mp4"
LIPSYNC_VIDEO = "output_lipsync.mp4"
REPORT = "report.json"  # timing + cost summary
