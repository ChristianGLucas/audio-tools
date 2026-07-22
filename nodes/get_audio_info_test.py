from gen.messages_pb2 import Audio
from nodes.get_audio_info import get_audio_info
from nodes._test_fixtures import make_context, sine_wav_bytes, stereo_sine_wav_bytes


def test_get_audio_info_mono_wav():
    ax = make_context()
    wav = sine_wav_bytes(freq=440.0, sr=22050, duration=1.0)
    result = get_audio_info(ax, Audio(data=wav, format="wav"))
    assert result.error == ""
    assert result.sample_rate == 22050
    assert result.channels == 1
    assert result.num_samples == 22050
    assert abs(result.duration_seconds - 1.0) < 1e-6
    assert result.bit_depth == 16
    assert result.sample_format == "wav-pcm"


def test_get_audio_info_stereo_wav():
    ax = make_context()
    wav = stereo_sine_wav_bytes(sr=16000, duration=0.5)
    result = get_audio_info(ax, Audio(data=wav, format="wav"))
    assert result.error == ""
    assert result.sample_rate == 16000
    assert result.channels == 2
    assert result.num_samples == 8000


def test_get_audio_info_pcm_input():
    ax = make_context()
    import numpy as np
    y = (0.4 * np.sin(2 * 3.14159265 * 220 * np.arange(0, 1.0, 1 / 8000))).astype(np.float32)
    pcm = (y * 32767).astype("<i2").tobytes()
    result = get_audio_info(ax, Audio(data=pcm, format="pcm", sample_rate=8000, channels=1, sample_format="int16"))
    assert result.error == ""
    assert result.sample_rate == 8000
    assert result.channels == 1
    assert result.num_samples == 8000


def test_get_audio_info_format_auto_detect():
    ax = make_context()
    wav = sine_wav_bytes(sr=22050, duration=0.25)
    # format left empty -> auto-detected from RIFF/WAVE header
    result = get_audio_info(ax, Audio(data=wav, format=""))
    assert result.error == ""
    assert result.sample_rate == 22050


def test_get_audio_info_empty_input_error():
    ax = make_context()
    result = get_audio_info(ax, Audio(data=b"", format="wav"))
    assert result.error != ""
    assert result.sample_rate == 0


def test_get_audio_info_malformed_input_error():
    ax = make_context()
    result = get_audio_info(ax, Audio(data=b"this is not audio data at all, just plain junk bytes", format=""))
    assert result.error != ""


def test_get_audio_info_deterministic():
    ax = make_context()
    wav = sine_wav_bytes()
    r1 = get_audio_info(ax, Audio(data=wav, format="wav"))
    r2 = get_audio_info(ax, Audio(data=wav, format="wav"))
    assert r1.sample_rate == r2.sample_rate
    assert r1.num_samples == r2.num_samples
    assert r1.duration_seconds == r2.duration_seconds
