from gen.messages_pb2 import Audio
from nodes.compute_audio_fingerprint import compute_audio_fingerprint
from nodes._test_fixtures import make_context, sine_wav_bytes


def test_compute_audio_fingerprint_shape_and_determinism():
    ax = make_context()
    wav = sine_wav_bytes(freq=440.0, sr=22050, duration=1.0)
    r1 = compute_audio_fingerprint(ax, Audio(data=wav, format="wav"))
    assert r1.error == ""
    assert r1.dimension == 63
    assert len(r1.vector) == 63

    r2 = compute_audio_fingerprint(ax, Audio(data=wav, format="wav"))
    assert list(r1.vector) == list(r2.vector)


def test_compute_audio_fingerprint_differs_for_different_clips():
    ax = make_context()
    wav_a = sine_wav_bytes(freq=440.0, sr=22050, duration=1.0)
    wav_b = sine_wav_bytes(freq=220.0, sr=22050, duration=1.0)
    r_a = compute_audio_fingerprint(ax, Audio(data=wav_a, format="wav"))
    r_b = compute_audio_fingerprint(ax, Audio(data=wav_b, format="wav"))
    assert list(r_a.vector) != list(r_b.vector)


def test_compute_audio_fingerprint_error_on_malformed_input():
    ax = make_context()
    result = compute_audio_fingerprint(ax, Audio(data=b"junk", format=""))
    assert result.error != ""
    assert result.dimension == 0
