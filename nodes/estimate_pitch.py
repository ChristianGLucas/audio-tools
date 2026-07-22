import numpy as np

import vendor.librosa as librosa
from gen.messages_pb2 import PitchInput, PitchResult
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio, to_mono_1d


def estimate_pitch(ax: AxiomContext, input: PitchInput) -> PitchResult:
    """Estimate the fundamental frequency (f0/pitch) track of a
    caller-supplied audio clip via the pYIN algorithm, plus a per-frame
    voiced/unvoiced flag and the mean/median f0 over voiced frames. The full
    per-frame track is returned (frame count is bounded by clip duration, not
    a large 2D matrix). Unvoiced frames report 0 Hz in f0_hz — use
    voiced_flag to distinguish them from a genuine 0 Hz estimate. Multi-
    channel audio is averaged to mono first. Malformed, empty, or oversized
    (>3 MiB) input returns a structured error rather than crashing. Wraps
    librosa's pYIN implementation (ISC-licensed, vendored).
    """
    audio = input.audio
    try:
        y, sr, _channels, _bit_depth, _sf = decode_audio(
            audio.data, audio.format, audio.sample_rate, audio.channels, audio.sample_format
        )
    except AudioDecodeError as e:
        return PitchResult(error=str(e))

    y_mono = to_mono_1d(y)
    fmin = input.fmin_hz if input.fmin_hz > 0 else librosa.note_to_hz("C2")
    fmax = input.fmax_hz if input.fmax_hz > 0 else librosa.note_to_hz("C7")
    if fmin >= fmax:
        return PitchResult(error="fmin_hz must be less than fmax_hz")

    try:
        f0, voiced_flag, _voiced_prob = librosa.pyin(y=y_mono, sr=sr, fmin=fmin, fmax=fmax)
    except Exception as e:  # noqa: BLE001
        return PitchResult(error=f"pitch estimation failed: {e}")

    voiced_mask = ~np.isnan(f0)
    voiced_values = f0[voiced_mask]
    mean_v = float(np.mean(voiced_values)) if voiced_values.size else 0.0
    median_v = float(np.median(voiced_values)) if voiced_values.size else 0.0
    f0_out = np.nan_to_num(f0, nan=0.0)

    return PitchResult(
        f0_hz=[float(v) for v in f0_out],
        voiced_flag=[bool(v) for v in voiced_flag],
        mean_voiced_f0_hz=mean_v,
        median_voiced_f0_hz=median_v,
        n_frames=int(len(f0_out)),
        n_voiced_frames=int(voiced_mask.sum()),
    )
