from gen.messages_pb2 import Audio
from nodes.compute_spectral_features import compute_spectral_features
from nodes._test_fixtures import make_context, sine_wav_bytes


def test_compute_spectral_features_centroid_matches_pure_tone():
    """Independent oracle: a pure tone's spectral centroid should sit close
    to its own frequency."""
    ax = make_context()
    wav = sine_wav_bytes(freq=440.0, sr=22050, duration=1.0)
    result = compute_spectral_features(ax, Audio(data=wav, format="wav"))
    assert result.error == ""
    assert abs(result.centroid_mean_hz - 440.0) < 40.0
    # A pure tone is highly non-flat (concentrated energy) -> low flatness.
    assert 0.0 <= result.flatness_mean < 0.2
    assert result.bandwidth_mean_hz >= 0.0
    assert result.rolloff_mean_hz >= 0.0
    assert len(result.contrast_mean) == 7
    assert len(result.contrast_std) == 7
    assert result.n_frames > 0


def test_compute_spectral_features_error_on_malformed_input():
    ax = make_context()
    result = compute_spectral_features(ax, Audio(data=b"junk", format=""))
    assert result.error != ""
