from gen.messages_pb2 import Audio
from nodes.detect_beats import detect_beats
from nodes._test_fixtures import make_context, click_track_wav_bytes


def test_detect_beats_on_known_click_track():
    ax = make_context()
    wav, expected_times = click_track_wav_bytes(bpm=120.0, sr=22050, duration=4.0)
    result = detect_beats(ax, Audio(data=wav, format="wav"))
    assert result.error == ""
    assert result.count == len(result.beat_times_seconds)
    assert result.count >= 4  # a 4s clip at 120 BPM has 8 clicks
    assert abs(result.bpm - 120.0) < 12.0
    # Beat times should be monotonically increasing and within the clip.
    times = list(result.beat_times_seconds)
    assert times == sorted(times)
    assert all(0.0 <= t <= 4.0 for t in times)


def test_detect_beats_error_on_empty_input():
    ax = make_context()
    result = detect_beats(ax, Audio(data=b"", format="wav"))
    assert result.error != ""
    assert result.count == 0
