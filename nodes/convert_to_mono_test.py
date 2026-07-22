from gen.messages_pb2 import Audio
from nodes.convert_to_mono import convert_to_mono
from nodes._audio_io import decode_audio
from nodes._test_fixtures import make_context, sine_wav_bytes, stereo_sine_wav_bytes
import numpy as np


def test_convert_to_mono_averages_channels():
    ax = make_context()
    wav = stereo_sine_wav_bytes(freq=440.0, sr=22050, duration=0.5, amplitude=0.6)
    # Independently decode the original stereo input to compute the expected
    # mono result ourselves (formula oracle: mono = mean over channels).
    y_orig, sr_orig, ch_orig, _bd, _sf = decode_audio(wav, "wav", 0, 0, "")
    expected_mono = y_orig.mean(axis=0)

    result = convert_to_mono(ax, Audio(data=wav, format="wav"))
    assert result.error == ""
    assert result.channels == 1
    assert result.sample_rate == sr_orig

    y_out, sr_out, ch_out, _bd2, _sf2 = decode_audio(result.data, result.format, result.sample_rate, result.channels, result.sample_format)
    assert ch_out == 1
    assert y_out.shape == expected_mono.shape
    # 16-bit PCM round trip introduces small quantization error.
    assert np.max(np.abs(y_out - expected_mono)) < 0.01


def test_convert_to_mono_already_mono_is_unchanged_in_value():
    ax = make_context()
    wav = sine_wav_bytes(freq=440.0, sr=22050, duration=0.5)
    y_orig, sr_orig, _ch, _bd, _sf = decode_audio(wav, "wav", 0, 0, "")

    result = convert_to_mono(ax, Audio(data=wav, format="wav"))
    assert result.error == ""
    assert result.channels == 1

    y_out, _sr, _ch2, _bd2, _sf2 = decode_audio(result.data, result.format, result.sample_rate, result.channels, result.sample_format)
    assert np.max(np.abs(y_out - y_orig)) < 0.01


def test_convert_to_mono_error_on_malformed_input():
    ax = make_context()
    result = convert_to_mono(ax, Audio(data=b"junk", format=""))
    assert result.error != ""
