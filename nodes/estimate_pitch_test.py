from gen.messages_pb2 import Audio, PitchInput
from nodes.estimate_pitch import estimate_pitch
from nodes._test_fixtures import make_context, sine_wav_bytes


def test_estimate_pitch_matches_known_frequency():
    """Independent oracle: a pure 440 Hz sine's estimated f0 should be
    (near-)exactly 440 Hz."""
    ax = make_context()
    wav = sine_wav_bytes(freq=440.0, sr=22050, duration=1.0)
    result = estimate_pitch(ax, PitchInput(audio=Audio(data=wav, format="wav")))
    assert result.error == ""
    assert result.n_voiced_frames > 0
    assert abs(result.mean_voiced_f0_hz - 440.0) < 3.0
    assert abs(result.median_voiced_f0_hz - 440.0) < 3.0
    assert len(result.f0_hz) == result.n_frames
    assert len(result.voiced_flag) == result.n_frames


def test_estimate_pitch_low_frequency():
    ax = make_context()
    wav = sine_wav_bytes(freq=110.0, sr=22050, duration=1.0)  # A2
    result = estimate_pitch(ax, PitchInput(audio=Audio(data=wav, format="wav")))
    assert result.error == ""
    assert abs(result.mean_voiced_f0_hz - 110.0) < 3.0


def test_estimate_pitch_invalid_range_error():
    ax = make_context()
    wav = sine_wav_bytes()
    result = estimate_pitch(
        ax, PitchInput(audio=Audio(data=wav, format="wav"), fmin_hz=2000.0, fmax_hz=100.0)
    )
    assert result.error != ""


def test_estimate_pitch_error_on_malformed_input():
    ax = make_context()
    result = estimate_pitch(ax, PitchInput(audio=Audio(data=b"junk", format="")))
    assert result.error != ""
