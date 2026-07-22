import vendor.librosa as librosa
from gen.messages_pb2 import MelSpectrogramInput, MelSpectrogramResult
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio, to_mono_1d


def compute_mel_spectrogram(ax: AxiomContext, input: MelSpectrogramInput) -> MelSpectrogramResult:
    """Compute a mel-scaled power spectrogram for a caller-supplied audio
    clip. Returns per-mel-band mean and standard deviation power aggregated
    over all frames (not the full n_mels x n_frames matrix — a raw
    time-resolved spectrogram can exceed the transport size cap for longer
    clips). Multi-channel audio is averaged to mono first. Malformed, empty,
    or oversized (>3 MiB) input returns a structured error rather than
    crashing. Wraps librosa's mel-spectrogram implementation (ISC-licensed,
    vendored).
    """
    audio = input.audio
    try:
        y, sr, _channels, _bit_depth, _sf = decode_audio(
            audio.data, audio.format, audio.sample_rate, audio.channels, audio.sample_format
        )
    except AudioDecodeError as e:
        return MelSpectrogramResult(error=str(e))

    y_mono = to_mono_1d(y)
    n_mels = input.n_mels if input.n_mels > 0 else 128
    kwargs = {}
    if input.frame.n_fft > 0:
        kwargs["n_fft"] = input.frame.n_fft
    if input.frame.hop_length > 0:
        kwargs["hop_length"] = input.frame.hop_length

    try:
        mel = librosa.feature.melspectrogram(y=y_mono, sr=sr, n_mels=n_mels, **kwargs)
    except Exception as e:  # noqa: BLE001
        return MelSpectrogramResult(error=f"mel-spectrogram computation failed: {e}")

    return MelSpectrogramResult(
        band_mean=[float(v) for v in mel.mean(axis=1)],
        band_std=[float(v) for v in mel.std(axis=1)],
        n_mels=int(mel.shape[0]),
        n_frames=int(mel.shape[1]),
    )
