import vendor.librosa as librosa
from gen.messages_pb2 import StftInput, StftResult
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio, resolve_frame_params, to_mono_1d


def compute_stft(ax: AxiomContext, input: StftInput) -> StftResult:
    """Compute the Short-Time Fourier Transform (STFT) magnitude spectrogram
    for a caller-supplied audio clip. Returns per-frequency-bin mean and
    standard deviation magnitude aggregated over all frames, plus the center
    frequency (Hz) of each bin — not the full n_freq_bins x n_frames matrix,
    which can exceed the transport size cap for longer clips. n_fft/hop_length
    are bounded so the resulting frame count cannot exceed 100,000 (an
    ordinary-looking small hop_length would otherwise drive an unbounded
    allocation). Multi-channel audio is averaged to mono first. Malformed,
    empty, oversized (>3 MiB), or out-of-range input returns a structured
    error rather than crashing. Wraps librosa's STFT implementation
    (ISC-licensed, vendored).
    """
    audio = input.audio
    try:
        y, sr, _channels, _bit_depth, _sf = decode_audio(
            audio.data, audio.format, audio.sample_rate, audio.channels, audio.sample_format
        )
        n_fft, hop_length = resolve_frame_params(input.frame.n_fft, input.frame.hop_length, y.shape[-1])
    except AudioDecodeError as e:
        return StftResult(error=str(e))

    y_mono = to_mono_1d(y)

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
