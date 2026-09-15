"""Shared ML contract for the legacy DeepBreath pipeline.

These constants document the MSc implementation and should be changed only when a
new experiment is intentionally introduced.
"""

from pathlib import Path


AUDIO_DURATION_SECONDS = 6.0

# The original notebooks used librosa.load(path) without passing sr, so librosa's
# default 22050 Hz resampling is part of the legacy preprocessing contract.
LIBROSA_DEFAULT_SAMPLE_RATE = 22050
EXPECTED_SAMPLE_RATE = LIBROSA_DEFAULT_SAMPLE_RATE

MEL_N_MELS = 128
MEL_TIME_FRAMES = 259
MEL_SPECTROGRAM_SHAPE = (MEL_N_MELS, MEL_TIME_FRAMES)
CNN_CHANNELS = 1
CNN_INPUT_SHAPE = (MEL_N_MELS, MEL_TIME_FRAMES, CNN_CHANNELS)

CLASS_NAMES = (
    "Bronchiectasis",
    "Bronchiolitis",
    "COPD",
    "Healthy",
    "Pneumonia",
    "URTI",
)
NUM_CLASSES = len(CLASS_NAMES)

# Preserves the original notebook misspelling for artifact compatibility.
LEGACY_MODEL_FILENAME = "melspectogramTrained.h5"
LEGACY_MODEL_PATH = Path("models") / LEGACY_MODEL_FILENAME

# The notebooks relied on librosa defaults for melspectrogram extraction.
MEL_SPECTROGRAM_PARAMS = {
    "n_mels": MEL_N_MELS,
    "sr": EXPECTED_SAMPLE_RATE,
    "n_fft": 2048,
    "hop_length": 512,
    "power": 2.0,
}
