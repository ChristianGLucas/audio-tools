import vendor.librosa as librosa
from gen.messages_pb2 import Audio
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio, encode_wav


def convert_to_mono(ax: AxiomContext, input: Audio) -> Audio:
    """Convert a caller-supplied stereo/multi-channel audio clip to mono by
    averaging its channels, returning the result as 16-bit PCM WAV bytes
    (base64-encoded). Already-mono input is returned unchanged (still
    re-encoded as WAV). Malformed, or empty input
    returns a structured error rather than crashing. Wraps librosa's
    to_mono implementation (ISC-licensed, vendored).
    """
    try:
        y, sr, _channels, _bit_depth, _sf = decode_audio(
            input.data, input.format, input.sample_rate, input.channels, input.sample_format
        )
    except AudioDecodeError as e:
        return Audio(error=str(e))

    try:
        y_mono = librosa.to_mono(y)
    except Exception as e:  # noqa: BLE001
        return Audio(error=f"mono conversion failed: {e}")

    wav_bytes = encode_wav(y_mono, sr)
    return Audio(data=wav_bytes, format="wav", sample_rate=sr, channels=1)
