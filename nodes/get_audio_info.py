from gen.messages_pb2 import Audio, AudioInfo
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio


def get_audio_info(ax: AxiomContext, input: Audio) -> AudioInfo:
    """Decode a caller-supplied audio clip (WAV file bytes, or raw PCM
    samples + sample_rate/channels/sample_format) and report its basic
    properties: sample rate, channel count, total sample count, duration in
    seconds, bit depth, and the sample encoding used to decode it. Malformed,
    empty, or oversized (>3 MiB) input returns a structured error rather than
    crashing.
    """
    try:
        y, sr, channels, bit_depth, sample_format_used = decode_audio(
            input.data, input.format, input.sample_rate, input.channels, input.sample_format
        )
    except AudioDecodeError as e:
        return AudioInfo(error=str(e))

    num_samples = int(y.shape[-1])
    duration_seconds = float(num_samples) / sr if sr else 0.0

    return AudioInfo(
        sample_rate=sr,
        channels=channels,
        num_samples=num_samples,
        duration_seconds=duration_seconds,
        bit_depth=bit_depth,
        sample_format=sample_format_used,
    )
