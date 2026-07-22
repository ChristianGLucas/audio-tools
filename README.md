# audio-tools

Composable Axiom nodes for deterministic audio-feature extraction from
caller-supplied waveforms — the operations audio-ML, music-information-
retrieval, and transcription-prep agents need. Distinct from
`music-theory-tools` (music theory over abstract notes/chords),
`musicxml-tools`/`midi-tools` (symbolic notation formats), and
`subtitle-tools` (caption files) — this package works on the actual
waveform/samples.

Built for the Axiom marketplace. License: MIT (see `LICENSE`).

## What it does

Every node takes a caller-supplied `Audio` clip — either raw WAV
(RIFF/WAVE) file bytes, or raw interleaved PCM samples plus
`sample_rate`/`channels`/`sample_format` — capped at 3 MiB, and returns a
deterministic feature or transform. There is no network access, no
wall-clock reads, and no randomness: the same input always produces the
same output.

### Nodes

- **GetAudioInfo** — decode and report sample rate, duration, channel
  count, sample count, and bit depth.
- **ComputeMfcc** — Mel-Frequency Cepstral Coefficients (per-coefficient
  mean/std).
- **ComputeMelSpectrogram** — mel-scaled power spectrogram (per-band
  mean/std).
- **ComputeStft** — Short-Time Fourier Transform magnitude spectrogram
  (per-bin mean/std + bin frequencies).
- **ComputeChroma** — 12-bin pitch-class chromagram (per-class mean/std).
- **EstimateTempo** — global tempo (BPM) via beat tracking.
- **DetectBeats** — individual beat timestamps + tempo.
- **DetectOnsets** — onset event timestamps.
- **EstimatePitch** — fundamental-frequency (f0) track via pYIN, with a
  voiced/unvoiced flag per frame.
- **ComputeSpectralFeatures** — centroid, bandwidth, rolloff, flatness,
  and octave-band contrast.
- **ComputeZeroCrossingRate** — zero-crossing rate (mean/std).
- **ComputeRmsEnergy** — RMS loudness envelope (mean/std/max).
- **ComputeTonnetz** — tonal-centroid (Tonnetz) features.
- **ResampleAudio** — resample to a target sample rate, returned as WAV.
- **ConvertToMono** — average multi-channel audio to mono, returned as WAV.
- **TrimSilence** — trim leading/trailing silence, returning the trimmed
  clip plus the retained region's boundaries.
- **ComputeAudioFingerprint** — a fixed-length (63-dim), deterministic
  summary feature vector for similarity search/classification.

Large 2D feature matrices (MFCC, mel-spectrogram, STFT, chroma) are
returned as per-coefficient/band/bin **mean and standard deviation**
aggregated over time, not the raw time-resolved matrix — this keeps every
response well under the platform's transport size cap regardless of clip
length.

## Implementation notes

This package wraps a **vendored, patched copy of librosa 0.9.2** (ISC
license) under `vendor/librosa/` rather than a plain PyPI dependency. See
`requirements.txt` for the full license audit and the "christiangeorgelucas/
audio-tools patch" comments in `vendor/librosa/util/files.py` and
`vendor/librosa/core/audio.py` for why: librosa's default PyPI install pulls
in `pooch` (which hard-depends on `requests` -> `certifi`, MPL-2.0) and, in
recent versions, `soxr` (LGPL-2.1-or-later) — both copyleft, and both
avoidable. The vendored copy drops librosa's demo-file-fetching code
(`pooch`-dependent, and unused — this package never touches the network)
and its own file-decoding path (`soundfile`, whose wheel bundles a compiled
libsndfile.so under LGPL-2.1). Audio decoding/encoding instead uses only
Python's stdlib `wave` module (`nodes/_audio_io.py`) — WAV and raw PCM in,
16-bit PCM WAV out. FLAC is not currently supported for the same reason
(no clean permissively-licensed decoder was available without pulling in
libsndfile); this is a deliberate scope cut, not an oversight.

Every runtime dependency in the resulting closure (numpy, scipy,
scikit-learn, joblib, decorator, resampy, numba, llvmlite, threadpoolctl,
narwhals) is MIT/BSD/ISC/Apache-2.0/PSF — no copyleft anywhere in the tree.
