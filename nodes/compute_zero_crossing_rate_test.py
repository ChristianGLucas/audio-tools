from gen.messages_pb2 import Audio
from nodes.compute_zero_crossing_rate import compute_zero_crossing_rate
from nodes._test_fixtures import make_context, sine_wav_bytes, silence_wav_bytes


def test_compute_zero_crossing_rate_matches_formula():
    """Independent oracle: a sine at frequency f, sample rate sr, crosses
    zero approximately 2f/sr times per sample (two crossings per period)."""
    ax = make_context()
    freq, sr = 440.0, 22050
    wav = sine_wav_bytes(freq=freq, sr=sr, duration=1.0)
    result = compute_zero_crossing_rate(ax, Audio(data=wav, format="wav"))
    assert result.error == ""
    expected = 2.0 * freq / sr
    assert abs(result.mean - expected) < expected * 0.25
    assert result.n_frames > 0


def test_compute_zero_crossing_rate_silence_is_zero():
    ax = make_context()
    wav = silence_wav_bytes(sr=22050, duration=0.5)
    result = compute_zero_crossing_rate(ax, Audio(data=wav, format="wav"))
    assert result.error == ""
    assert result.mean == 0.0


def test_compute_zero_crossing_rate_error_on_malformed_input():
    ax = make_context()
    result = compute_zero_crossing_rate(ax, Audio(data=b"junk", format=""))
    assert result.error != ""
