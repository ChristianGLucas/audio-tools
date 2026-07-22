import vendor.librosa as librosa
from gen.messages_pb2 import Audio, TempoResult
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio, to_mono_1d


def estimate_tempo(ax: AxiomContext, input: Audio) -> TempoResult:
    """Estimate the global tempo (beats per minute) of a caller-supplied
    audio clip via onset-strength-based beat tracking. Multi-channel audio
    is averaged to mono first. Malformed, or empty input
    returns a structured error rather than crashing. Wraps librosa's beat
    tracker (ISC-licensed, vendored); see DetectBeats for individual beat
    timestamps.
    """
    try:
        y, sr, _channels, _bit_depth, _sf = decode_audio(
            input.data, input.format, input.sample_rate, input.channels, input.sample_format
        )
    except AudioDecodeError as e:
        return TempoResult(error=str(e))

    y_mono = to_mono_1d(y)
    try:
        tempo, _beats = librosa.beat.beat_track(y=y_mono, sr=sr)
    except Exception as e:  # noqa: BLE001
        return TempoResult(error=f"tempo estimation failed: {e}")

    return TempoResult(bpm=float(tempo))
