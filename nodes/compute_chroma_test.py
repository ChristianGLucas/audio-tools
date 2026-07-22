from gen.messages_pb2 import Audio, ChromaInput
from nodes.compute_chroma import compute_chroma
from nodes._test_fixtures import make_context, sine_wav_bytes


def test_compute_chroma_identifies_pitch_class_a():
    """Independent oracle: A4 = 440 Hz. librosa's chroma bin ordering starts
    at C (index 0), so A is index 9 — the dominant pitch class for a pure
    440 Hz sine should be A."""
    ax = make_context()
    wav = sine_wav_bytes(freq=440.0, sr=22050, duration=1.0)
    result = compute_chroma(ax, ChromaInput(audio=Audio(data=wav, format="wav")))
    assert result.error == ""
    assert len(result.pitch_class_mean) == 12
    assert len(result.pitch_class_std) == 12
    assert result.n_frames > 0

    dominant = max(range(12), key=lambda i: result.pitch_class_mean[i])
    assert dominant == 9  # pitch class A


def test_compute_chroma_error_on_malformed_input():
    ax = make_context()
    result = compute_chroma(ax, ChromaInput(audio=Audio(data=b"junk", format="")))
    assert result.error != ""


