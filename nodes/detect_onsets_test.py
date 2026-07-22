from gen.messages_pb2 import Audio
from nodes.detect_onsets import detect_onsets
from nodes._test_fixtures import make_context, click_track_wav_bytes


def test_detect_onsets_matches_known_click_times():
    """Independent oracle: onsets should land near the exact times the click
    track's impulses were placed at."""
    ax = make_context()
    interval = 0.5  # 120 BPM
    wav, expected_times = click_track_wav_bytes(bpm=120.0, sr=22050, duration=4.0)
    result = detect_onsets(ax, Audio(data=wav, format="wav"))
    assert result.error == ""
    assert result.count == len(result.onset_times_seconds)
    assert result.count >= 5

    for t in result.onset_times_seconds:
        nearest_multiple = round(t / interval) * interval
        assert abs(t - nearest_multiple) < 0.1


def test_detect_onsets_error_on_malformed_input():
    ax = make_context()
    result = detect_onsets(ax, Audio(data=b"junk bytes not audio", format=""))
    assert result.error != ""
    assert result.count == 0
