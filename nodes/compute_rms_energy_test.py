from gen.messages_pb2 import Audio
from nodes.compute_rms_energy import compute_rms_energy
from nodes._test_fixtures import make_context, sine_wav_bytes, silence_wav_bytes


def test_compute_rms_energy_matches_formula():
    """Independent oracle: a sine wave with amplitude A has RMS = A/sqrt(2)."""
    ax = make_context()
    amplitude = 0.5
    wav = sine_wav_bytes(freq=440.0, sr=22050, duration=1.0, amplitude=amplitude)
    result = compute_rms_energy(ax, Audio(data=wav, format="wav"))
    assert result.error == ""
    expected = amplitude / (2.0 ** 0.5)
    assert abs(result.mean - expected) < expected * 0.1
    assert result.max >= result.mean
    assert result.n_frames > 0


def test_compute_rms_energy_silence_is_zero():
    ax = make_context()
    wav = silence_wav_bytes(sr=22050, duration=0.5)
    result = compute_rms_energy(ax, Audio(data=wav, format="wav"))
    assert result.error == ""
    assert result.mean == 0.0
    assert result.max == 0.0


def test_compute_rms_energy_error_on_malformed_input():
    ax = make_context()
    result = compute_rms_energy(ax, Audio(data=b"junk", format=""))
    assert result.error != ""
