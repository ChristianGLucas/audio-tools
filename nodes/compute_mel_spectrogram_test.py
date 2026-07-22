from gen.messages_pb2 import Audio, MelSpectrogramInput
from nodes.compute_mel_spectrogram import compute_mel_spectrogram
from nodes._test_fixtures import make_context, sine_wav_bytes


def test_compute_mel_spectrogram_default_shape():
    ax = make_context()
    wav = sine_wav_bytes(freq=440.0, sr=22050, duration=1.0)
    result = compute_mel_spectrogram(ax, MelSpectrogramInput(audio=Audio(data=wav, format="wav")))
    assert result.error == ""
    assert result.n_mels == 128
    assert len(result.band_mean) == 128
    assert len(result.band_std) == 128
    assert result.n_frames > 0
    # Power spectrogram values are non-negative by construction.
    assert all(v >= 0.0 for v in result.band_mean)


def test_compute_mel_spectrogram_custom_n_mels():
    ax = make_context()
    wav = sine_wav_bytes(sr=16000, duration=0.5)
    result = compute_mel_spectrogram(
        ax, MelSpectrogramInput(audio=Audio(data=wav, format="wav"), n_mels=40)
    )
    assert result.error == ""
    assert result.n_mels == 40
    assert len(result.band_mean) == 40


def test_compute_mel_spectrogram_error_on_empty_input():
    ax = make_context()
    result = compute_mel_spectrogram(ax, MelSpectrogramInput(audio=Audio(data=b"", format="wav")))
    assert result.error != ""
    assert result.n_frames == 0
