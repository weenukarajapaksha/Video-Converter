"""Stage registry: the canonical pipeline order.

Heavy ML stages are imported lazily so that the API server and the unit tests can
start without loading torch.
"""

from __future__ import annotations

from importlib import import_module

from dubber.pipeline.stage import Stage

# (stage name, "module:Class")
_REGISTRY: list[tuple[str, str]] = [
    ("ingest", "dubber.stages.ingest:IngestStage"),
    ("separate", "dubber.stages.separate:SeparateStage"),
    ("transcribe", "dubber.stages.transcribe:TranscribeStage"),
    ("diarize", "dubber.stages.diarize:DiarizeStage"),
    ("voices", "dubber.stages.voices:VoicesStage"),
    ("translate", "dubber.stages.translate:TranslateStage"),
    ("synthesize", "dubber.stages.synthesize:SynthesizeStage"),
    ("mix", "dubber.stages.mix:MixStage"),
    ("render", "dubber.stages.render:RenderStage"),
    ("lipsync", "dubber.stages.lipsync:LipSyncStage"),
]

STAGE_NAMES = [name for name, _ in _REGISTRY]
# With review enabled, the pipeline pauses here so the user can edit the translation
# and the speaker->voice mapping before any speech is generated.
REVIEW_STAGE = "translate"


def _load(target: str) -> type[Stage]:
    module, _, cls = target.partition(":")
    return getattr(import_module(module), cls)


def build_stages(lipsync: bool = False) -> list[Stage]:
    stages = []
    for name, target in _REGISTRY:
        if name == "lipsync" and not lipsync:
            continue
        try:
            stages.append(_load(target)())
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("dubber.stages."):
                continue  # stage not implemented yet
            raise
    return stages


def __getattr__(name: str):  # keeps `from dubber.stages import IngestStage` working
    for _, target in _REGISTRY:
        if target.endswith(f":{name}"):
            return _load(target)
    raise AttributeError(name)
