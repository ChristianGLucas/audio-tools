import vendor.librosa as librosa
from gen.messages_pb2 import StftInput, StftResult
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio, to_mono_1d


def compute_stft(ax: AxiomContext, input: StftInput) -> StftResult:
    """Compute the Short-Time Fourier Transform (STFT) magnitude spectrogram
    for a caller-supplied audio clip. Returns per-frequency-bin mean and
    standard deviation magnitude aggregated over all frames, plus the center
    frequency (Hz) of each bin — not the full n_freq_bins x n_frames matrix,
    which can exceed the transport size cap for longer clips. Multi-channel
    audio is averaged to mono first. Malformed, empty, or oversized (>3 MiB)
    input returns a structured error rather than crashing. Wraps librosa's
    STFT implementation (ISC-licensed, vendored).
    """
    audio = input.audio
    try:
        y, sr, _channels, _bit_depth, _sf = decode_audio(
            audio.data, audio.format, audio.sample_rate, audio.channels, audio.sample_format
        )
    except AudioDecodeError as e:
        return StftResult(error=str(e))

    y_mono = to_mono_1d(y)
    n_fft = input.frame.n_fft if input.frame.n_fft > 0 else 2048
    hop_length = input.frame.hop_length if input.frame.hop_length > 0 else 512

    try:
        stft = librosa.stft(y_mono, n_fft=n_fft, hop_length=hop_length)
        magnitude = abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    except Exception as e:  # noqa: BLE001
        return StftResult(error=f"STFT computation failed: {e}")

    return StftResult(
        magnitude_mean=[float(v) for v in magnitude.mean(axis=1)],
        magnitude_std=[float(v) for v in magnitude.std(axis=1)],
        n_freq_bins=int(magnitude.shape[0]),
        n_frames=int(magnitude.shape[1]),
        bin_frequencies_hz=[float(v) for v in freqs],
    )
