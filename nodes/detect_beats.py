import vendor.librosa as librosa
from gen.messages_pb2 import Audio, BeatsResult
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio, to_mono_1d


def detect_beats(ax: AxiomContext, input: Audio) -> BeatsResult:
    """Detect individual beat positions (in seconds from the start of the
    clip) in a caller-supplied audio clip, plus the global tempo estimate
    that beat tracking produces as a byproduct. Multi-channel audio is
    averaged to mono first. Malformed, or empty input
    returns a structured error rather than crashing. Wraps librosa's beat
    tracker (ISC-licensed, vendored); see EstimateTempo for the tempo alone.
    """
    try:
        y, sr, _channels, _bit_depth, _sf = decode_audio(
            input.data, input.format, input.sample_rate, input.channels, input.sample_format
        )
    except AudioDecodeError as e:
        return BeatsResult(error=str(e))

    y_mono = to_mono_1d(y)
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    except Exception as e:  # noqa: BLE001
        return BeatsResult(error=f"beat detection failed: {e}")

    return BeatsResult(
        beat_times_seconds=[float(v) for v in beat_times],
        count=int(len(beat_times)),
        bpm=float(tempo),
    )
