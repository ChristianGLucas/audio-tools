from gen.messages_pb2 import Audio
from nodes.compute_tonnetz import compute_tonnetz
from nodes._test_fixtures import make_context, sine_wav_bytes


def test_compute_tonnetz_shape_and_determinism():
    ax = make_context()
    wav = sine_wav_bytes(freq=440.0, sr=22050, duration=1.0)
    r1 = compute_tonnetz(ax, Audio(data=wav, format="wav"))
    assert r1.error == ""
    assert len(r1.dimension_mean) == 6
    assert len(r1.dimension_std) == 6
    assert r1.n_frames > 0
    # Tonnetz coordinates are bounded in [-1, 1] by construction.
    assert all(-1.0001 <= v <= 1.0001 for v in r1.dimension_mean)

    r2 = compute_tonnetz(ax, Audio(data=wav, format="wav"))
    assert list(r1.dimension_mean) == list(r2.dimension_mean)


def test_compute_tonnetz_error_on_malformed_input():
    ax = make_context()
    result = compute_tonnetz(ax, Audio(data=b"junk", format=""))
    assert result.error != ""
