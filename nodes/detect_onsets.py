import vendor.librosa as librosa
from gen.messages_pb2 import Audio, OnsetsResult
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio, to_mono_1d


def detect_onsets(ax: AxiomContext, input: Audio) -> OnsetsResult:
    """Detect onset events (the start of new sonic events — notes, drum
    hits, transients) in a caller-supplied audio clip, returned as times in
    seconds from the start of the clip. Multi-channel audio is averaged to
    mono first. Malformed, empty, or oversized (>3 MiB) input returns a
    structured error rather than crashing. Wraps librosa's onset-strength-
    based onset detector (ISC-licensed, vendored).
    """
    try:
        y, sr, _channels, _bit_depth, _sf = decode_audio(
            input.data, input.format, input.sample_rate, input.channels, input.sample_format
        )
    except AudioDecodeError as e:
        return OnsetsResult(error=str(e))

    y_mono = to_mono_1d(y)
    try:
        onset_frames = librosa.onset.onset_detect(y=y_mono, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    except Exception as e:  # noqa: BLE001
        return OnsetsResult(error=f"onset detection failed: {e}")

    return OnsetsResult(
        onset_times_seconds=[float(v) for v in onset_times],
        count=int(len(onset_times)),
    )
