import vendor.librosa as librosa
from gen.messages_pb2 import Audio, ZcrResult
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio, to_mono_1d


def compute_zero_crossing_rate(ax: AxiomContext, input: Audio) -> ZcrResult:
    """Compute the zero-crossing rate (fraction of sign changes per frame) of
    a caller-supplied audio clip, aggregated (mean/std) over all frames — a
    cheap proxy for noisiness/pitch used in speech and percussion detection.
    Multi-channel audio is averaged to mono first. Malformed or empty
    input returns a structured error rather than crashing. Wraps librosa's zero-crossing-rate implementation
    (ISC-licensed, vendored).
    """
    try:
        y, sr, _channels, _bit_depth, _sf = decode_audio(
            input.data, input.format, input.sample_rate, input.channels, input.sample_format
        )
    except AudioDecodeError as e:
        return ZcrResult(error=str(e))

    y_mono = to_mono_1d(y)
    try:
        zcr = librosa.feature.zero_crossing_rate(y_mono)[0]
    except Exception as e:  # noqa: BLE001
        return ZcrResult(error=f"zero-crossing-rate computation failed: {e}")

    return ZcrResult(
        mean=float(zcr.mean()),
        std=float(zcr.std()),
        n_frames=int(zcr.shape[0]),
    )
