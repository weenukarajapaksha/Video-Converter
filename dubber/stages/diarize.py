"""Speaker diarization with pyannote 3.1, merged with ASR word timings.

pyannote tends to over-split speakers on film audio (5 clusters for ~3 people on a
real test clip), which makes one actor switch English voices mid-scene. After
diarization, clusters are merged agglomeratively when their speaker embeddings are
similar AND their pitch is close. Embeddings alone don't separate the cases: two
genuinely different speakers (one man, one woman) scored cosine 0.44, the same as
over-split pairs of one person (0.41-0.44), but their pitch differed by ~8
semitones vs ~2-3. Every merge decision is saved to diarization_meta.json.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from dubber import ml
from dubber.align import Turn, assign_speakers
from dubber.artifacts import TRANSCRIPT, VOCALS
from dubber.config import get_settings
from dubber.errors import DubberError
from dubber.pipeline.job import Job
from dubber.pipeline.stage import ProgressFn, Stage
from dubber.schemas import Transcript
from dubber.stages.transcribe import SR, TranscribeStage, load_mono_16k

log = logging.getLogger(__name__)

DIARIZATION = "diarization.json"
DIARIZATION_META = "diarization_meta.json"
PIPELINE_ID = "pyannote/speaker-diarization-3.1"

MERGE_SIMILARITY = 0.40  # cosine between duration-weighted cluster centroids
MAX_PITCH_DISTANCE = 0.20  # |ln(f0a / f0b)|, ~3.5 semitones
UNKNOWN_PITCH_SIMILARITY = 0.50  # stricter when a cluster has no reliable pitch
MIN_SPEECH_S = 1.5  # smaller clusters are absorbed by a similar same-gender one
GENDER_SPLIT_HZ = 160.0  # same split as the voices stage


@dataclass
class Cluster:
    label: str
    duration: float
    centroid: np.ndarray | None  # unit vector; None if pyannote gave no embedding
    f0: float | None

    @property
    def gender(self) -> str | None:
        return None if self.f0 is None else ("male" if self.f0 < GENDER_SPLIT_HZ else "female")


def _cosine(a: Cluster, b: Cluster) -> float:
    if a.centroid is None or b.centroid is None:
        return -1.0
    return float(a.centroid @ b.centroid)


def _pitch_distance(a: Cluster, b: Cluster) -> float | None:
    return None if a.f0 is None or b.f0 is None else abs(float(np.log(a.f0 / b.f0)))


def _combine(a: Cluster, b: Cluster) -> Cluster:
    """Duration-weighted centroid (re-normalized) and log-mean pitch."""
    centroid = None
    if a.centroid is not None and b.centroid is not None:
        c = a.centroid * a.duration + b.centroid * b.duration
        centroid = c / (np.linalg.norm(c) or 1.0)
    else:
        centroid = a.centroid if a.centroid is not None else b.centroid
    voiced = [(c.f0, c.duration) for c in (a, b) if c.f0]
    f0 = (float(np.exp(sum(np.log(f) * d for f, d in voiced) / sum(d for _, d in voiced)))
          if voiced else None)
    return Cluster(a.label, a.duration + b.duration, centroid, f0)


def merge_speakers(clusters: list[Cluster], min_speakers: int = 1,
                   similarity: float = MERGE_SIMILARITY,
                   max_pitch: float = MAX_PITCH_DISTANCE,
                   min_speech_s: float = MIN_SPEECH_S) -> tuple[dict[str, str], list[dict]]:
    """Agglomerative merge of over-split speaker clusters.

    Returns ``(mapping original label -> final label, decision log)``. The final
    label is that of the longest-speaking member, so the mapping is stable.
    """
    originals = {c.label: c for c in clusters}
    groups = {c.label: c for c in clusters}  # aggregate duration / pitch per group
    members = {c.label: [c.label] for c in clusters}
    log_: list[dict] = []

    def linkage(x: str, y: str) -> float:
        """Average linkage: duration-weighted mean similarity of original members.
        Unlike a running centroid it doesn't depend on merge order."""
        pairs = [(originals[a], originals[b]) for a in members[x] for b in members[y]]
        weights = [a.duration * b.duration for a, b in pairs]
        return float(np.average([_cosine(a, b) for a, b in pairs], weights=weights))

    def mergeable(x: str, y: str) -> bool:
        pitch = _pitch_distance(groups[x], groups[y])
        if pitch is None:
            return linkage(x, y) >= UNKNOWN_PITCH_SIMILARITY
        return linkage(x, y) >= similarity and pitch <= max_pitch

    def do_merge(x: str, y: str, reason: str) -> None:
        keep, drop = (x, y) if groups[x].duration >= groups[y].duration else (y, x)
        log_.append({"merge": drop, "into": keep, "reason": reason,
                     "similarity": round(linkage(keep, drop), 3),
                     "pitch_distance": _round(_pitch_distance(groups[keep], groups[drop]))})
        groups[keep] = _combine(groups[keep], groups[drop])
        members[keep] += members.pop(drop)
        del groups[drop]

    # 1. Similar voice + similar pitch: most similar pair first; ties go to the pair
    #    with the closer pitch, then by label, so the result is deterministic.
    while len(groups) > min_speakers:
        labels = sorted(groups)
        pairs = [(round(linkage(x, y), 2), -(_pitch_distance(groups[x], groups[y]) or 0.0), x, y)
                 for i, x in enumerate(labels) for y in labels[i + 1:] if mergeable(x, y)]
        if not pairs:
            break
        *_, x, y = max(pairs)
        do_merge(x, y, "similar voice and pitch")

    # 2. Tiny clusters: absorb into the most similar cluster of the same gender.
    for label in sorted(groups, key=lambda k: (groups[k].duration, k)):
        if len(groups) <= min_speakers or label not in groups:
            continue
        tiny = groups[label]
        if tiny.duration >= min_speech_s:
            continue
        hosts = [k for k in sorted(groups) if k != label and groups[k].duration >= min_speech_s
                 and (tiny.gender is None or groups[k].gender in (None, tiny.gender))]
        if hosts:
            host = max(hosts, key=lambda k: linkage(k, label))
            do_merge(host, label, f"tiny cluster ({tiny.duration:.1f}s)")

    mapping = {orig: final for final, origs in members.items() for orig in origs}
    return mapping, log_


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 3)
# Gated repos the pipeline pulls in; each needs its terms accepted on huggingface.co.
GATED_MODELS = {"pyannote/speaker-diarization-3.1": "config.yaml",
                "pyannote/segmentation-3.0": "config.yaml"}


def _allow_pyannote_checkpoints() -> None:
    """PyTorch >= 2.6 loads checkpoints with ``weights_only=True``, which rejects any
    class not explicitly allow-listed. pyannote 3.x checkpoints pickle a few of their
    own metadata classes; allow exactly those instead of disabling the safety check."""
    import torch
    from pyannote.audio.core.task import Problem, Resolution, Specifications

    torch.serialization.add_safe_globals(
        [torch.torch_version.TorchVersion, Specifications, Problem, Resolution])


def check_model_access(token: str) -> None:
    """Fail fast with an actionable message: pyannote itself swallows 403s and
    later crashes with "'NoneType' object has no attribute 'eval'"."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    for repo, filename in GATED_MODELS.items():
        try:
            hf_hub_download(repo, filename, token=token)
        except GatedRepoError:
            raise DubberError(f"HF_TOKEN's account has not accepted the terms of {repo}: "
                              f"open https://huggingface.co/{repo}, click 'Agree', then "
                              "retry") from None
        except RepositoryNotFoundError:
            raise DubberError(f"HF_TOKEN is invalid or cannot see {repo}") from None


def _hf_hub_compat() -> None:
    """pyannote 3.x passes ``use_auth_token=`` to ``hf_hub_download``; huggingface_hub
    >= 1.0 only accepts ``token=``. Translate the argument instead of pinning an old
    hub version that the other ML libraries (faster-whisper, tokenizers) have outgrown.
    Must run before pyannote is imported, since it binds the function at import time."""
    import huggingface_hub

    original = huggingface_hub.hf_hub_download
    if getattr(original, "_dubber_compat", False):
        return

    def hf_hub_download(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None:
            kwargs.setdefault("token", use_auth_token)
        return original(*args, **kwargs)

    hf_hub_download._dubber_compat = True
    huggingface_hub.hf_hub_download = hf_hub_download


def _clusters(turns: list[Turn], labels: list[str], embeddings, audio) -> list[Cluster]:
    from dubber.stages.voices import median_f0

    out = []
    for i, label in enumerate(labels):
        spans = [(t.start, t.end) for t in turns if t.speaker == label]
        vec = None
        if embeddings is not None and i < len(embeddings) and np.all(np.isfinite(embeddings[i])):
            norm = np.linalg.norm(embeddings[i])
            vec = embeddings[i] / norm if norm else None
        out.append(Cluster(label, sum(e - s for s, e in spans), vec, median_f0(audio, spans)))
    return out


class DiarizeStage(Stage):
    name = "diarize"
    outputs = (DIARIZATION, TRANSCRIPT)
    weight = 1.0

    def run(self, job: Job, progress: ProgressFn) -> None:
        raw = TranscribeStage.raw(job)
        turns, meta = self._diarize(job, progress) if raw.segments else ([], {})
        job.write_json(DIARIZATION, [t.__dict__ for t in turns])
        job.write_json(DIARIZATION_META, meta)

        progress(0.9, "merging speakers with words")
        lines = assign_speakers(raw.segments, turns)
        transcript = Transcript(segments=lines, speech_ratio=raw.speech_ratio)
        job.write_model(TRANSCRIPT, transcript)
        progress(1.0, f"{len(transcript.speakers)} speaker(s), {len(lines)} lines")

    def _diarize(self, job: Job, progress: ProgressFn) -> tuple[list[Turn], dict]:
        token = get_settings().hf_token
        if not token:
            raise DubberError("HF_TOKEN is not set - required for pyannote diarization")
        check_model_access(token)
        _hf_hub_compat()  # must precede the first pyannote import
        _allow_pyannote_checkpoints()
        import torch
        from pyannote.audio import Pipeline as PyannotePipeline

        audio = load_mono_16k(job.path(VOCALS))
        with ml.gpu_scope():
            progress(0.05, "loading pyannote")
            pipe = PyannotePipeline.from_pretrained(PIPELINE_ID, use_auth_token=token)
            if pipe is None:
                raise DubberError(f"could not load {PIPELINE_ID}: accept its terms on "
                                  "huggingface.co and check HF_TOKEN")
            pipe.to(torch.device(ml.device()))
            # Passing the waveform in memory avoids torchaudio file decoding entirely.
            waveform = torch.from_numpy(audio)[None]
            options = {key: int(v) for key in ("num_speakers", "min_speakers", "max_speakers")
                       if (v := job.option(key)) is not None}
            progress(0.2, "diarizing")
            diarization, embeddings = pipe({"waveform": waveform, "sample_rate": SR},
                                           return_embeddings=True, **options)
            del pipe

        raw_turns = [Turn(round(seg.start, 3), round(seg.end, 3), speaker)
                     for seg, _, speaker in diarization.itertracks(yield_label=True)]
        clusters = _clusters(raw_turns, diarization.labels(), embeddings, audio)
        meta = {"clusters": [{"label": c.label, "speech_s": round(c.duration, 2),
                              "f0_hz": _round(c.f0)} for c in clusters],
                "similarity": {a.label: {b.label: round(_cosine(a, b), 3) for b in clusters}
                               for a in clusters}}
        if "num_speakers" in options:  # the user fixed the count: trust it
            meta.update(merges=[], note="num_speakers given; auto-merge skipped")
            return raw_turns, meta

        progress(0.8, "merging over-split speakers")
        mapping, merges = merge_speakers(clusters, min_speakers=options.get("min_speakers", 1))
        meta.update(merges=merges, final_speakers=len(set(mapping.values())))
        if merges:
            log.info("diarization: %d -> %d speakers", len(clusters), meta["final_speakers"])
        return [Turn(t.start, t.end, mapping[t.speaker]) for t in raw_turns], meta
