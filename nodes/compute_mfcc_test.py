from gen.messages_pb2 import Audio, FrameParams, MfccInput
from nodes.compute_mfcc import compute_mfcc
from nodes._test_fixtures import make_context, sine_wav_bytes


def test_compute_mfcc_default_shape():
    ax = make_context()
    wav = sine_wav_bytes(freq=440.0, sr=22050, duration=1.0)
    result = compute_mfcc(ax, MfccInput(audio=Audio(data=wav, format="wav")))
    assert result.error == ""
    assert result.n_mfcc == 13
    assert len(result.coefficient_mean) == 13
    assert len(result.coefficient_std) == 13
    assert result.n_frames > 0
    assert all(s >= 0.0 for s in result.coefficient_std)


def test_compute_mfcc_custom_n_mfcc():
    ax = make_context()
    wav = sine_wav_bytes(sr=16000, duration=0.5)
    result = compute_mfcc(ax, MfccInput(audio=Audio(data=wav, format="wav"), n_mfcc=20))
    assert result.error == ""
    assert result.n_mfcc == 20
    assert len(result.coefficient_mean) == 20


def test_compute_mfcc_deterministic():
    ax = make_context()
    wav = sine_wav_bytes()
    r1 = compute_mfcc(ax, MfccInput(audio=Audio(data=wav, format="wav")))
    r2 = compute_mfcc(ax, MfccInput(audio=Audio(data=wav, format="wav")))
    assert list(r1.coefficient_mean) == list(r2.coefficient_mean)
    assert r1.n_frames == r2.n_frames


def test_compute_mfcc_error_on_malformed_input():
    ax = make_context()
    result = compute_mfcc(ax, MfccInput(audio=Audio(data=b"not audio", format="")))
    assert result.error != ""
    assert result.n_frames == 0


def test_compute_mfcc_rejects_tiny_hop_length_that_would_explode_frame_count():
    """Regression test: see the audio-tools 2026-07-21 adversarial review finding."""
    ax = make_context()
    wav = sine_wav_bytes(freq=440.0, sr=22050, duration=5.0)
    result = compute_mfcc(
        ax, MfccInput(audio=Audio(data=wav, format="wav"), frame=FrameParams(hop_length=1))
    )
    assert result.error != ""


def test_compute_mfcc_rejects_oversized_n_mfcc():
    ax = make_context()
    wav = sine_wav_bytes(sr=22050, duration=0.5)
    result = compute_mfcc(ax, MfccInput(audio=Audio(data=wav, format="wav"), n_mfcc=100000))
    assert result.error != ""
