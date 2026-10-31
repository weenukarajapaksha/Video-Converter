class DubberError(Exception):
    """Base class for all pipeline errors."""


class InputValidationError(DubberError):
    """The input video is unusable (wrong format, too short/long, no audio...). Never retried."""


class TransientError(DubberError):
    """A failure worth retrying: rate limits, timeouts, flaky network."""


class MediaError(DubberError):
    """FFmpeg / ffprobe failed."""
