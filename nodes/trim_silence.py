import numpy as np

import vendor.librosa as librosa
from gen.messages_pb2 import TrimSilenceInput, TrimSilenceResult, Audio
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio, encode_wav

# Absolute-amplitude floor below which a clip is treated as true silence.
# librosa.effects.trim measures loudness RELATIVE TO THE CLIP'S OWN PEAK, so
# a genuinely all-zero (or near-zero) clip has no dynamic range and — with no
# special-casing — reads as entirely "at the peak" (i.e. NOT silent) rather
# than entirely silent. We special-case true silence explicitly rather than
# rely on that degenerate relative-threshold behavior.
_ABSOLUTE_SILENCE_THRESHOLD = 1e-6


def trim_silence(ax: AxiomContext, input: TrimSilenceInput) -> TrimSilenceResult:
    """Trim leading and trailing silence from a caller-supplied audio clip
    (any audio below top_db decibels relative to the clip's peak). Returns
    the trimmed audio as 16-bit PCM WAV bytes plus the start/end boundaries
    (in seconds) of the retained region in the original clip. Channel count
    is preserved. If the entire clip is below the silence threshold,
    all_silent is set and trimmed_audio is empty. Malformed, empty, or
    oversized (>3 MiB) input returns a structured error rather than
    crashing. Wraps librosa's silence-trimming implementation (ISC-licensed,
    vendored).
    """
    audio = input.audio
    try:
        y, sr, channels, _bit_depth, _sf = decode_audio(
            audio.data, audio.format, audio.sample_rate, audio.channels, audio.sample_format
        )
    except AudioDecodeError as e:
        return TrimSilenceResult(error=str(e))

    if input.top_db < 0:
        return TrimSilenceResult(error="top_db must be >= 0 (0 = default 60 dB)")
    top_db = input.top_db if input.top_db > 0 else 60.0

    if float(np.max(np.abs(y))) < _ABSOLUTE_SILENCE_THRESHOLD:
        empty = np.zeros((channels, 0), dtype=np.float32) if channels > 1 else np.zeros(0, dtype=np.float32)
        wav_bytes = encode_wav(empty, sr)
        return TrimSilenceResult(
            trimmed_audio=Audio(data=wav_bytes, format="wav", sample_rate=sr, channels=channels),
            start_seconds=0.0,
            end_seconds=0.0,
            all_silent=True,
        )

    try:
        y_trimmed, index = librosa.effects.trim(y, top_db=top_db)
    except Exception as e:  # noqa: BLE001
        return TrimSilenceResult(error=f"silence trimming failed: {e}")

    start_sample, end_sample = int(index[0]), int(index[1])
    all_silent = start_sample == end_sample

    wav_bytes = encode_wav(y_trimmed, sr)
    trimmed_audio = Audio(data=wav_bytes, format="wav", sample_rate=sr, channels=channels)

    return TrimSilenceResult(
        trimmed_audio=trimmed_audio,
        start_seconds=float(start_sample) / sr,
        end_seconds=float(end_sample) / sr,
        all_silent=all_silent,
    )
