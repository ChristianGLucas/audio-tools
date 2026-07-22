import vendor.librosa as librosa
from gen.messages_pb2 import ChromaInput, ChromaResult
from gen.axiom_context import AxiomContext

from nodes._audio_io import AudioDecodeError, decode_audio, to_mono_1d


def compute_chroma(ax: AxiomContext, input: ChromaInput) -> ChromaResult:
    """Compute a 12-bin chromagram (pitch-class energy: C, C#, D, ... B) for
    a caller-supplied audio clip — useful for music/key analysis and chord
    detection. Returns per-pitch-class mean and standard deviation energy
    aggregated over all frames (not the full 12 x n_frames matrix). Multi-
    channel audio is averaged to mono first. Malformed, empty, or oversized
    (>3 MiB) input returns a structured error rather than crashing. Wraps
    librosa's chroma-STFT implementation (ISC-licensed, vendored).
    """
    audio = input.audio
    try:
        y, sr, _channels, _bit_depth, _sf = decode_audio(
            audio.data, audio.format, audio.sample_rate, audio.channels, audio.sample_format
        )
    except AudioDecodeError as e:
        return ChromaResult(error=str(e))

    y_mono = to_mono_1d(y)
    kwargs = {}
    if input.frame.n_fft > 0:
        kwargs["n_fft"] = input.frame.n_fft
    if input.frame.hop_length > 0:
        kwargs["hop_length"] = input.frame.hop_length

    try:
        chroma = librosa.feature.chroma_stft(y=y_mono, sr=sr, **kwargs)
    except Exception as e:  # noqa: BLE001
        return ChromaResult(error=f"chroma computation failed: {e}")

    return ChromaResult(
        pitch_class_mean=[float(v) for v in chroma.mean(axis=1)],
        pitch_class_std=[float(v) for v in chroma.std(axis=1)],
        n_frames=int(chroma.shape[1]),
    )
