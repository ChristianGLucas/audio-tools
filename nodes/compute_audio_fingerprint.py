import vendor.librosa as librosa
from gen.messages_pb2 import Audio, FingerprintResult
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio, to_mono_1d


def compute_audio_fingerprint(ax: AxiomContext, input: Audio) -> FingerprintResult:
    """Compute a fixed-length, deterministic summary feature vector for a
    caller-supplied audio clip — mean and standard deviation of 13 MFCCs, 12
    chroma bins, spectral centroid/bandwidth/rolloff/flatness, zero-crossing
    rate, and RMS energy, plus the estimated tempo — 63 dimensions total, in
    a fixed field order, suitable for nearest-neighbor similarity search or
    as a classifier input. Multi-channel audio is averaged to mono first.
    Malformed, empty, or oversized (>3 MiB) input returns a structured error
    rather than crashing. Wraps librosa's feature-extraction implementations
    (ISC-licensed, vendored).
    """
    try:
        y, sr, _channels, _bit_depth, _sf = decode_audio(
            input.data, input.format, input.sample_rate, input.channels, input.sample_format
        )
    except AudioDecodeError as e:
        return FingerprintResult(error=str(e))

    y_mono = to_mono_1d(y)

    try:
        mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y_mono, sr=sr)
        centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y_mono, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y_mono, sr=sr)[0]
        flatness = librosa.feature.spectral_flatness(y=y_mono)[0]
        zcr = librosa.feature.zero_crossing_rate(y_mono)[0]
        rms = librosa.feature.rms(y=y_mono)[0]
        tempo, _beats = librosa.beat.beat_track(y=y_mono, sr=sr)
    except Exception as e:  # noqa: BLE001
        return FingerprintResult(error=f"fingerprint computation failed: {e}")

    vector = []
    vector.extend(float(v) for v in mfcc.mean(axis=1))
    vector.extend(float(v) for v in mfcc.std(axis=1))
    vector.extend(float(v) for v in chroma.mean(axis=1))
    vector.extend(float(v) for v in chroma.std(axis=1))
    vector.append(float(centroid.mean()))
    vector.append(float(centroid.std()))
    vector.append(float(bandwidth.mean()))
    vector.append(float(bandwidth.std()))
    vector.append(float(rolloff.mean()))
    vector.append(float(rolloff.std()))
    vector.append(float(flatness.mean()))
    vector.append(float(flatness.std()))
    vector.append(float(zcr.mean()))
    vector.append(float(zcr.std()))
    vector.append(float(rms.mean()))
    vector.append(float(rms.std()))
    vector.append(float(tempo))

    return FingerprintResult(vector=vector, dimension=len(vector))
