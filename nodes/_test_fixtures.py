"""Shared test fixtures for christiangeorgelucas/audio-tools node tests.

Synthetic audio with KNOWN properties, used as independent oracles: e.g. a
pure 440 Hz sine's pitch/spectral-centroid/zero-crossing-rate/chroma class
are all predictable from closed-form acoustics/music-theory formulas
independent of librosa's own implementation, and a click track at a known
BPM has a known tempo and known onset times.
"""

import numpy as np

from gen.axiom_context import SecretStatus
from nodes._audio_io import encode_wav
import vendor.librosa as librosa


class TestContext:
    """Minimal AxiomContext implementation for unit tests."""

    class _Logger:
        def debug(self, msg: str, **attrs) -> None: pass
        def info(self, msg: str, **attrs) -> None: pass
        def warn(self, msg: str, **attrs) -> None: pass
        def error(self, msg: str, **attrs) -> None: pass

    class _Secrets:
        def __init__(self, m: dict, revoked: set) -> None:
            self._m = m or {}
            self._revoked = revoked or set()
        def get(self, name: str):
            v = self._m.get(name)
            return (v, True) if v is not None else ("", False)
        def status(self, name: str) -> SecretStatus:
            if name in self._m:
                return SecretStatus.AVAILABLE
            if name in self._revoked:
                return SecretStatus.REVOKED
            return SecretStatus.UNSET

    def __init__(self, secrets_map=None, revoked_names=None) -> None:
        self.log = self._Logger()
        self.secrets = self._Secrets(secrets_map or {}, revoked_names)
        self.execution_id = "test-execution-id"
        self.flow_id = "test-flow-id"
        self.tenant_id = "test-tenant-id"


def make_context() -> TestContext:
    return TestContext()


def sine_wav_bytes(freq=440.0, sr=22050, duration=1.0, amplitude=0.5) -> bytes:
    """A pure sine tone at `freq` Hz — the pitch/spectral-centroid/chroma/ZCR
    oracle: its f0, spectral centroid, and zero-crossing-rate are all
    predictable from closed-form formulas independent of librosa."""
    t = np.arange(0, duration, 1.0 / sr)
    y = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return encode_wav(y, sr)


def stereo_sine_wav_bytes(freq=440.0, sr=22050, duration=1.0, amplitude=0.5) -> bytes:
    t = np.arange(0, duration, 1.0 / sr)
    left = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    right = (amplitude * 0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return encode_wav(np.stack([left, right]), sr)


def silence_wav_bytes(sr=22050, duration=1.0) -> bytes:
    y = np.zeros(int(sr * duration), dtype=np.float32)
    return encode_wav(y, sr)


def click_track_wav_bytes(bpm=120.0, sr=22050, duration=4.0):
    """A click track with clicks at exact `bpm`-derived intervals — the
    tempo/beat/onset oracle. Returns (wav_bytes, expected_click_times)."""
    interval = 60.0 / bpm
    times = np.arange(0, duration, interval)
    click_sig = librosa.clicks(times=times, sr=sr, length=int(duration * sr)).astype(np.float32)
    return encode_wav(click_sig, sr), times


def padded_tone_wav_bytes(freq=440.0, sr=22050, tone_duration=1.0, lead_silence=0.5, trail_silence=0.7, amplitude=0.6):
    """A tone with KNOWN leading/trailing true-zero silence padding — the
    TrimSilence boundary oracle."""
    lead = np.zeros(int(sr * lead_silence), dtype=np.float32)
    trail = np.zeros(int(sr * trail_silence), dtype=np.float32)
    t = np.arange(0, tone_duration, 1.0 / sr)
    tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    y = np.concatenate([lead, tone, trail])
    return encode_wav(y, sr), lead_silence, lead_silence + tone_duration
