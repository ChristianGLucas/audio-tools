import vendor.librosa as librosa
from gen.messages_pb2 import MelSpectrogramInput, MelSpectrogramResult
from gen.axiom_context import AxiomContext

from nodes._audio_io import (
    MAX_N_MELS,
    AudioDecodeError,
    decode_audio,
    resolve_frame_params,
    to_mono_1d,
    validate_coefficient_count,
)


def compute_mel_spectrogram(ax: AxiomContext, input: MelSpectrogramInput) -> MelSpectrogramResult:
    """Compute a mel-scaled power spectrogram for a caller-supplied audio
    clip. Returns per-mel-band mean and standard deviation power aggregated
    over all frames (not the full n_mels x n_frames matrix — a raw
    time-resolved spectrogram can exceed the transport size cap for longer
    clips). n_mels is capped at 512 and n_fft/hop_length are bounded so the
    resulting frame count cannot exceed 100,000 (an ordinary-looking small
    hop_length would otherwise drive an unbounded allocation). Multi-channel
    audio is averaged to mono first. Malformed, empty, oversized (>3 MiB), or
    out-of-range input returns a structured error rather than crashing.
    Wraps librosa's mel-spectrogram implementation (ISC-licensed, vendored).
    """
    audio = input.audio
    try:
        y, sr, _channels, _bit_depth, _sf = decode_audio(
            audio.data, audio.format, audio.sample_rate, audio.channels, audio.sample_format
        )
        n_mels = validate_coefficient_count(input.n_mels, MAX_N_MELS, "n_mels") or 128
        n_fft, hop_length = resolve_frame_params(input.frame.n_fft, input.frame.hop_length, y.shape[-1])
    except AudioDecodeError as e:
        return MelSpectrogramResult(error=str(e))

    y_mono = to_mono_1d(y)
    kwargs = {"n_fft": n_fft, "hop_length": hop_length}

    try:
        mel = librosa.feature.melspectrogram(y=y_mono, sr=sr, n_mels=n_mels, **kwargs)
    except Exception as e:  # noqa: BLE001
        return MelSpectrogramResult(error=f"mel-spectrogram computation failed: {e}")

    return MelSpectrogramResult(
        band_mean=[float(v) for v in mel.mean(axis=1)],
        band_std=[float(v) for v in mel.std(axis=1)],
        n_mels=int(mel.shape[0]),
        n_frames=int(mel.shape[1]),
    )
