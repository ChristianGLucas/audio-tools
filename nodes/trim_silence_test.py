from gen.messages_pb2 import Audio, TrimSilenceInput
from nodes.trim_silence import trim_silence
from nodes._test_fixtures import make_context, padded_tone_wav_bytes, silence_wav_bytes


def test_trim_silence_finds_known_boundaries():
    """Independent oracle: the fixture pads a tone with exact, known amounts
    of true-zero silence before and after — the detected boundaries should
    land close to those known padding lengths."""
    ax = make_context()
    wav, expected_start, expected_end = padded_tone_wav_bytes(
        freq=440.0, sr=22050, tone_duration=1.0, lead_silence=0.5, trail_silence=0.7
    )
    result = trim_silence(ax, TrimSilenceInput(audio=Audio(data=wav, format="wav")))
    assert result.error == ""
    assert result.all_silent is False
    assert abs(result.start_seconds - expected_start) < 0.15
    assert abs(result.end_seconds - expected_end) < 0.15
    assert result.trimmed_audio.error == ""
    assert result.trimmed_audio.format == "wav"


def test_trim_silence_all_silent_clip():
    ax = make_context()
    wav = silence_wav_bytes(sr=22050, duration=1.0)
    result = trim_silence(ax, TrimSilenceInput(audio=Audio(data=wav, format="wav")))
    assert result.error == ""
    assert result.all_silent is True


def test_trim_silence_error_on_malformed_input():
    ax = make_context()
    result = trim_silence(ax, TrimSilenceInput(audio=Audio(data=b"junk", format="")))
    assert result.error != ""
