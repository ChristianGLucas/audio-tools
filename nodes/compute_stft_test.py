from gen.messages_pb2 import Audio, StftInput
from nodes.compute_stft import compute_stft
from nodes._test_fixtures import make_context, sine_wav_bytes


def test_compute_stft_peak_frequency_matches_pure_tone():
    """Independent oracle: a pure 440 Hz sine's STFT energy should
    concentrate in the bin closest to 440 Hz."""
    ax = make_context()
    wav = sine_wav_bytes(freq=440.0, sr=22050, duration=1.0)
    result = compute_stft(ax, StftInput(audio=Audio(data=wav, format="wav")))
    assert result.error == ""
    assert result.n_freq_bins == 1025  # n_fft=2048 default -> n_fft//2 + 1
    assert result.n_frames > 0
    assert len(result.bin_frequencies_hz) == result.n_freq_bins

    peak_idx = max(range(len(result.magnitude_mean)), key=lambda i: result.magnitude_mean[i])
    peak_freq = result.bin_frequencies_hz[peak_idx]
    assert abs(peak_freq - 440.0) < 25.0  # within ~2 FFT bins (~10.7 Hz/bin)


def test_compute_stft_error_on_malformed_input():
    ax = make_context()
    result = compute_stft(ax, StftInput(audio=Audio(data=b"junk", format="")))
    assert result.error != ""
    assert result.n_frames == 0
