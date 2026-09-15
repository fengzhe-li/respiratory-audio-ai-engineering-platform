# DeepBreath Architecture

DeepBreath is organized as a two-stage respiratory AI system:

1. CNN acoustic perception: preprocess WAV files, extract Mel spectrograms, and predict respiratory disease probabilities.
2. LLM clinical reporting: combine CNN confidence, Top-2 differential diagnosis, and patient-level crackle/wheeze aggregation into structured patient-facing and ward-facing reports.

The current codebase preserves the MSc experiment logic. Later phases should improve evaluation with patient-level splits and formal model validation.

## ML Contract

- Audio duration: 6 seconds.
- Librosa sample rate: 22050 Hz, matching the notebook default.
- Mel-spectrogram shape: `128 x 259`.
- CNN input shape: `128 x 259 x 1`.
- Class order: Bronchiectasis, Bronchiolitis, COPD, Healthy, Pneumonia, URTI.
- Output layer: six-unit softmax.

The original artifact name `melspectogramTrained.h5` is misspelled and retained
only for provenance.

Runtime note: the archived notebook sometimes passes Mel arrays as
`(batch, 128, 259)`. With TensorFlow 2.21 / Keras 3.15, inference requires the
explicit channel dimension `(batch, 128, 259, 1)`. The refactored inference path
adds that channel axis without changing the saved model architecture or weights.
