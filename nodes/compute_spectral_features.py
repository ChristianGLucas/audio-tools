import vendor.librosa as librosa
from gen.messages_pb2 import Audio, SpectralFeaturesResult
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio, to_mono_1d


def compute_spectral_features(ax: AxiomContext, input: Audio) -> SpectralFeaturesResult:
    """Compute standard timbral spectral descriptors for a caller-supplied
    audio clip — spectral centroid, bandwidth, rolloff, flatness, and
    per-octave-band contrast — each aggregated (mean/std) over all frames.
    Multi-channel audio is averaged to mono first. Malformed or empty
    input returns a structured error rather than crashing. Wraps librosa's spectral-feature implementations
    (ISC-licensed, vendored).
    """
    try:
        y, sr, _channels, _bit_depth, _sf = decode_audio(
            input.data, input.format, input.sample_rate, input.channels, input.sample_format
        )
    except AudioDecodeError as e:
        return SpectralFeaturesResult(error=str(e))

    y_mono = to_mono_1d(y)
    try:
        centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y_mono, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y_mono, sr=sr)[0]
        flatness = librosa.feature.spectral_flatness(y=y_mono)[0]
        contrast = librosa.feature.spectral_contrast(y=y_mono, sr=sr)
    except Exception as e:  # noqa: BLE001
        return SpectralFeaturesResult(error=f"spectral feature computation failed: {e}")

    return SpectralFeaturesResult(
        centroid_mean_hz=float(centroid.mean()),
        centroid_std_hz=float(centroid.std()),
        bandwidth_mean_hz=float(bandwidth.mean()),
        bandwidth_std_hz=float(bandwidth.std()),
        rolloff_mean_hz=float(rolloff.mean()),
        rolloff_std_hz=float(rolloff.std()),
        flatness_mean=float(flatness.mean()),
        flatness_std=float(flatness.std()),
        contrast_mean=[float(v) for v in contrast.mean(axis=1)],
        contrast_std=[float(v) for v in contrast.std(axis=1)],
        n_frames=int(centroid.shape[0]),
    )
