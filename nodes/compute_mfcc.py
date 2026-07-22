import vendor.librosa as librosa
from gen.messages_pb2 import MfccInput, MfccResult
from gen.axiom_context import AxiomContext

from nodes._audio_io import (
    AudioDecodeError,
    decode_audio,
    resolve_frame_params,
    to_mono_1d,
    validate_coefficient_count,
)


def compute_mfcc(ax: AxiomContext, input: MfccInput) -> MfccResult:
    """Compute Mel-Frequency Cepstral Coefficients (MFCCs) — the standard
    speech/audio timbral feature — for a caller-supplied audio clip. Returns
    per-coefficient mean and standard deviation aggregated over all frames
    (not the full n_mfcc x n_frames matrix; frame count scales with clip
    duration). Multi-channel audio is averaged to mono first. Malformed,
    empty, or out-of-range input returns a structured error rather than
    crashing. Wraps librosa's MFCC implementation (ISC-licensed, vendored).
    """
    audio = input.audio
    try:
        y, sr, _channels, _bit_depth, _sf = decode_audio(
            audio.data, audio.format, audio.sample_rate, audio.channels, audio.sample_format
        )
        n_mfcc = validate_coefficient_count(input.n_mfcc, "n_mfcc") or 13
        n_fft, hop_length = resolve_frame_params(input.frame.n_fft, input.frame.hop_length, y.shape[-1])
    except AudioDecodeError as e:
        return MfccResult(error=str(e))

    y_mono = to_mono_1d(y)
    kwargs = {"n_fft": n_fft, "hop_length": hop_length}

    try:
        mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=n_mfcc, **kwargs)
    except Exception as e:  # noqa: BLE001 - translate any DSP-library failure to a structured error
        return MfccResult(error=f"MFCC computation failed: {e}")

    return MfccResult(
        coefficient_mean=[float(v) for v in mfcc.mean(axis=1)],
        coefficient_std=[float(v) for v in mfcc.std(axis=1)],
        n_mfcc=int(mfcc.shape[0]),
        n_frames=int(mfcc.shape[1]),
    )
