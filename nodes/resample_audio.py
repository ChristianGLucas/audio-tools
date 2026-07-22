import vendor.librosa as librosa
from gen.messages_pb2 import ResampleInput, Audio
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio, encode_wav


def resample_audio(ax: AxiomContext, input: ResampleInput) -> Audio:
    """Resample a caller-supplied audio clip to a target sample rate,
    returning the result as 16-bit PCM WAV bytes (base64-encoded).
    Channel count is preserved. Malformed, empty, or non-positive
    target_sample_rate input returns a structured error rather than
    crashing. Wraps librosa's resampler (ISC-licensed, vendored, resampy
    backend).
    """
    audio = input.audio
    try:
        y, sr, channels, _bit_depth, _sf = decode_audio(
            audio.data, audio.format, audio.sample_rate, audio.channels, audio.sample_format
        )
    except AudioDecodeError as e:
        return Audio(error=str(e))

    target_sr = input.target_sample_rate
    if target_sr <= 0:
        return Audio(error="target_sample_rate must be positive")

    try:
        if target_sr == sr:
            y_resampled = y
        else:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    except Exception as e:  # noqa: BLE001
        return Audio(error=f"resampling failed: {e}")

    wav_bytes = encode_wav(y_resampled, target_sr)
    return Audio(data=wav_bytes, format="wav", sample_rate=target_sr, channels=channels)
