"""Shared audio decode/encode helpers for christiangeorgelucas/audio-tools.

Deliberately does NOT use `soundfile`/`audioread`/librosa.load() — see the
"christiangeorgelucas/audio-tools patch" comments in vendor/librosa/core/audio.py
and vendor/librosa/util/files.py, and the license audit at the top of
requirements.txt, for why. WAV decode/encode uses only Python's stdlib `wave`
module (PSF license); raw PCM decode uses only numpy. Both are fully
permissive and keep this package's runtime dependency tree free of copyleft
(GPL/AGPL/LGPL/MPL) code.

Every node in this package is a pure, deterministic, single-input ->
single-output transform: no network calls, no wall-clock reads, no
randomness. Size/memory/DoS/cost bounds are the platform's concern, not
this package's — nodes validate only genuine audio-format correctness.
"""

import io
import wave

import numpy as np


def resolve_frame_params(n_fft: int, hop_length: int, n_samples: int):
    """Validate and default caller-supplied n_fft/hop_length.

    Returns (effective_n_fft, effective_hop_length). Raises AudioDecodeError
    on a domain-invalid combination (negative values).
    """
    if n_fft < 0:
        raise AudioDecodeError("n_fft must be >= 0 (0 = default 2048)")
    if hop_length < 0:
        raise AudioDecodeError("hop_length must be >= 0 (0 = default 512)")

    effective_n_fft = n_fft if n_fft > 0 else 2048
    effective_hop = hop_length if hop_length > 0 else 512

    return effective_n_fft, effective_hop


def validate_coefficient_count(n: int, field_name: str) -> int:
    """Validate a caller-supplied coefficient/band count (n_mfcc/n_mels)."""
    if n < 0:
        raise AudioDecodeError(f"{field_name} must be >= 0 (0 = default)")
    return n


class AudioDecodeError(ValueError):
    """Raised for any malformed or unsupported audio input.

    Node handlers catch this and translate it into a structured error
    response rather than letting the process crash.
    """


def decode_audio(data: bytes, fmt: str, sample_rate: int, channels: int, sample_format: str):
    """Decode caller-supplied audio bytes into (y, sr, decoded_channels, bit_depth, sample_format_used).

    y is a float32 numpy array in approximately [-1, 1]: shape (n,) for mono,
    or (channels, n) for multi-channel (librosa's convention — leading axes
    are channels, the last axis is time).
    """
    if not data:
        raise AudioDecodeError("audio data is empty")

    fmt_norm = (fmt or "").strip().lower()
    if not fmt_norm:
        fmt_norm = "wav" if _looks_like_wav(data) else ""
    if fmt_norm not in ("wav", "pcm"):
        raise AudioDecodeError(
            "format must be 'wav' or 'pcm' (or omitted for WAV auto-detection "
            "via the RIFF/WAVE header)"
        )

    if fmt_norm == "wav":
        y, sr, ch, bit_depth = _decode_wav(data)
        sample_format_used = "wav-pcm"
    else:
        y, sr, ch, bit_depth, sample_format_used = _decode_pcm(
            data, sample_rate, channels, sample_format
        )

    n_per_channel = y.shape[-1]
    if n_per_channel == 0:
        raise AudioDecodeError("decoded audio has zero samples")
    if ch <= 0:
        raise AudioDecodeError(f"channel count {ch} is invalid; must be positive")
    if sr <= 0:
        raise AudioDecodeError(f"sample_rate {sr} is invalid; must be positive")

    return y, int(sr), int(ch), int(bit_depth), sample_format_used


def _looks_like_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WAVE"


def _decode_wav(data: bytes):
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
    except (wave.Error, EOFError, RuntimeError) as e:
        raise AudioDecodeError(f"malformed WAV data: {e}") from e

    if sampwidth == 1:
        # WAV 8-bit PCM is unsigned, centered at 128.
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        bit_depth = 8
    elif sampwidth == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        bit_depth = 16
    elif sampwidth == 3:
        if len(raw) % 3 != 0:
            raise AudioDecodeError("malformed WAV data: truncated 24-bit sample")
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        as_i32 = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        as_i32 = np.where(as_i32 & 0x800000, as_i32 - 0x1000000, as_i32)
        arr = as_i32.astype(np.float32) / 8388608.0
        bit_depth = 24
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        bit_depth = 32
    else:
        raise AudioDecodeError(f"unsupported WAV sample width: {sampwidth * 8} bits")

    if channels < 1:
        raise AudioDecodeError("malformed WAV data: non-positive channel count")
    if len(arr) % channels != 0:
        raise AudioDecodeError("malformed WAV data: sample count is not a multiple of channel count")

    if channels > 1:
        arr = arr.reshape(-1, channels).T  # (channels, n)

    return arr, framerate, channels, bit_depth


_PCM_DTYPES = {
    "int16": ("<i2", 32768.0, 16),
    "int32": ("<i4", 2147483648.0, 32),
    "float32": ("<f4", 1.0, 32),
}


def _decode_pcm(data: bytes, sample_rate, channels, sample_format):
    if sample_rate is None or sample_rate <= 0:
        raise AudioDecodeError("sample_rate is required and must be positive for format='pcm'")
    if channels is None or channels <= 0:
        raise AudioDecodeError("channels is required and must be positive for format='pcm'")

    sf_norm = (sample_format or "int16").strip().lower()
    if sf_norm not in _PCM_DTYPES:
        raise AudioDecodeError("sample_format must be one of: int16, int32, float32")
    dtype, scale, bit_depth = _PCM_DTYPES[sf_norm]

    itemsize = np.dtype(dtype).itemsize
    frame_bytes = itemsize * channels
    if frame_bytes == 0 or len(data) % frame_bytes != 0:
        raise AudioDecodeError(
            "pcm data length is not a whole number of frames for the given "
            "channels/sample_format"
        )

    arr = np.frombuffer(data, dtype=dtype).astype(np.float32) / scale
    if channels > 1:
        arr = arr.reshape(-1, channels).T  # (channels, n)

    return arr, sample_rate, channels, bit_depth, sf_norm


def to_mono_1d(y: np.ndarray) -> np.ndarray:
    """Collapse a (channels, n) or (n,) array to mono (n,) by averaging channels."""
    y = np.asarray(y)
    if y.ndim == 1:
        return y
    return np.mean(y, axis=0)


def encode_wav(y: np.ndarray, sr: int) -> bytes:
    """Encode a float32 array (mono (n,) or multi-channel (channels, n)),
    approximately in [-1, 1], as 16-bit PCM WAV bytes."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        channels = 1
        interleaved = y
    else:
        channels = y.shape[0]
        interleaved = y.T.reshape(-1)

    clipped = np.clip(interleaved, -1.0, 1.0)
    int_samples = np.round(clipped * 32767.0).astype("<i2")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(max(channels, 1))
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()
