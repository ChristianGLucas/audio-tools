import vendor.librosa as librosa
from gen.messages_pb2 import Audio, RmsResult
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio, to_mono_1d


def compute_rms_energy(ax: AxiomContext, input: Audio) -> RmsResult:
    """Compute the root-mean-square (RMS) energy envelope of a caller-
    supplied audio clip, aggregated (mean/std/max) over all frames — a
    measure of loudness over time. Multi-channel audio is averaged to mono
    first. Malformed, empty, or oversized (>3 MiB) input returns a
    structured error rather than crashing. Wraps librosa's RMS
    implementation (ISC-licensed, vendored).
    """
    try:
        y, sr, _channels, _bit_depth, _sf = decode_audio(
            input.data, input.format, input.sample_rate, input.channels, input.sample_format
        )
    except AudioDecodeError as e:
        return RmsResult(error=str(e))

    y_mono = to_mono_1d(y)
    try:
        rms = librosa.feature.rms(y=y_mono)[0]
    except Exception as e:  # noqa: BLE001
        return RmsResult(error=f"RMS computation failed: {e}")

    return RmsResult(
        mean=float(rms.mean()),
        std=float(rms.std()),
        max=float(rms.max()),
        n_frames=int(rms.shape[0]),
    )
