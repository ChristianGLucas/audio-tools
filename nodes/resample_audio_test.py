from gen.messages_pb2 import Audio, ResampleInput
from nodes.resample_audio import resample_audio
from nodes.get_audio_info import get_audio_info
from nodes._audio_io import decode_audio
from nodes._test_fixtures import make_context, sine_wav_bytes


def test_resample_audio_changes_sample_rate_and_preserves_pitch():
    ax = make_context()
    wav = sine_wav_bytes(freq=440.0, sr=22050, duration=1.0)
    result = resample_audio(ax, ResampleInput(audio=Audio(data=wav, format="wav"), target_sample_rate=11025))
    assert result.error == ""
    assert result.format == "wav"
    assert result.sample_rate == 11025
    assert result.channels == 1

    info = get_audio_info(ax, result)
    assert info.error == ""
    assert info.sample_rate == 11025
    assert abs(info.duration_seconds - 1.0) < 0.01

    # Independent oracle: pitch should still read ~440 Hz after resampling.
    import vendor.librosa as librosa
    y, sr, _ch, _bd, _sf = decode_audio(result.data, result.format, result.sample_rate, result.channels, result.sample_format)
    f0, _vf, _vp = librosa.pyin(y=y, sr=sr, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    import numpy as np
    voiced = f0[~np.isnan(f0)]
    assert abs(voiced.mean() - 440.0) < 5.0


def test_resample_audio_same_rate_is_identity_length():
    ax = make_context()
    wav = sine_wav_bytes(sr=22050, duration=0.5)
    result = resample_audio(ax, ResampleInput(audio=Audio(data=wav, format="wav"), target_sample_rate=22050))
    assert result.error == ""
    assert result.sample_rate == 22050


def test_resample_audio_invalid_target_rate_error():
    ax = make_context()
    wav = sine_wav_bytes()
    result = resample_audio(ax, ResampleInput(audio=Audio(data=wav, format="wav"), target_sample_rate=0))
    assert result.error != ""


def test_resample_audio_error_on_malformed_input():
    ax = make_context()
    result = resample_audio(ax, ResampleInput(audio=Audio(data=b"junk", format=""), target_sample_rate=16000))
    assert result.error != ""
