from gen.messages_pb2 import Audio
from nodes.estimate_tempo import estimate_tempo
from nodes._test_fixtures import make_context, click_track_wav_bytes


def test_estimate_tempo_matches_known_click_track_bpm():
    """Independent oracle: a click track synthesized at an exact 120 BPM
    interval should be estimated close to 120 BPM."""
    ax = make_context()
    wav, _expected_times = click_track_wav_bytes(bpm=120.0, sr=22050, duration=4.0)
    result = estimate_tempo(ax, Audio(data=wav, format="wav"))
    assert result.error == ""
    assert abs(result.bpm - 120.0) < 12.0


def test_estimate_tempo_deterministic():
    ax = make_context()
    wav, _ = click_track_wav_bytes(bpm=100.0, sr=22050, duration=4.0)
    r1 = estimate_tempo(ax, Audio(data=wav, format="wav"))
    r2 = estimate_tempo(ax, Audio(data=wav, format="wav"))
    assert r1.bpm == r2.bpm


def test_estimate_tempo_error_on_malformed_input():
    ax = make_context()
    result = estimate_tempo(ax, Audio(data=b"not audio", format=""))
    assert result.error != ""
    assert result.bpm == 0.0
