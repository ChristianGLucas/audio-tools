import vendor.librosa as librosa
from gen.messages_pb2 import Audio, TonnetzResult
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio, to_mono_1d


def compute_tonnetz(ax: AxiomContext, input: Audio) -> TonnetzResult:
    """Compute the tonnetz (tonal centroid features) of a caller-supplied
    audio clip — six dimensions representing the harmonic content projected
    onto the Tonnetz (perfect-fifth, minor-third, and major-third
    relationships) — aggregated (mean/std) over all frames. Multi-channel
    audio is averaged to mono first. Malformed, empty, or oversized (>3 MiB)
    input returns a structured error rather than crashing. Wraps librosa's
    tonnetz implementation (ISC-licensed, vendored).
    """
    try:
        y, sr, _channels, _bit_depth, _sf = decode_audio(
            input.data, input.format, input.sample_rate, input.channels, input.sample_format
        )
    except AudioDecodeError as e:
        return TonnetzResult(error=str(e))

    y_mono = to_mono_1d(y)
    try:
        tonnetz = librosa.feature.tonnetz(y=y_mono, sr=sr)
    except Exception as e:  # noqa: BLE001
        return TonnetzResult(error=f"tonnetz computation failed: {e}")

    return TonnetzResult(
        dimension_mean=[float(v) for v in tonnetz.mean(axis=1)],
        dimension_std=[float(v) for v in tonnetz.std(axis=1)],
        n_frames=int(tonnetz.shape[1]),
    )
