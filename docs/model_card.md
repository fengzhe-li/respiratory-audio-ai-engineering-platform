# Model Card Draft

## Model

DeepBreath Mel-spectrogram CNN.

Legacy artifact name: `melspectogramTrained.h5`. The misspelling is preserved
from the original MSc notebook workflow.

## Intended Use

Research and portfolio demonstration of respiratory sound classification and AI-assisted report generation.

## Not Intended For

Clinical diagnosis, triage, treatment planning, or autonomous medical decision-making.

## Inputs

Six-second mono respiratory sound segments converted to Mel spectrograms with shape `128 x 259 x 1`.

## Outputs

Six-class probability distribution:

- Bronchiectasis
- Bronchiolitis
- COPD
- Healthy
- Pneumonia
- URTI

The saved model and refactored architecture use a six-unit softmax output. Any
reference to an eight-neuron output layer in the MSc paper should be treated as a
documentation inconsistency, not as the implemented model.

## Evaluation

The original MSc result is preserved as a legacy sample-level evaluation artifact
and should be treated as research provenance. That split can place samples from
the same patient, including augmented variants of the same respiratory cycle,
across train and validation.

The refactored project adds a separate patient-level grouped evaluation path.
It uses patient ID as the primary grouping key and keeps augmented versions of a
base respiratory cycle in the same split.

Grouped holdout summary:

- Split method: deterministic class-aware patient-level grouped split.
- Random seed: `42`.
- Patients: train `22`, validation `5`, test `6`.
- Patient overlap: `0`.
- Base-cycle overlap: `0`.
- Test samples: `774`.
- Test accuracy: `0.4535`.
- Test balanced accuracy: `0.4391`.
- Test macro F1: `0.3483`.
- Test weighted F1: `0.4442`.

Repeated grouped holdout evaluation across five fixed seeds shows that the
patient-independent result is not yet stable:

- Accuracy mean/std: `0.3695` / `0.0515`.
- Accuracy range: `0.2971` to `0.4148`.
- Balanced accuracy mean/std: `0.4157` / `0.0874`.
- Macro F1 mean/std: `0.3295` / `0.0687`.

The Phase 3B single-run accuracy of `0.4535` is above the repeated deterministic
holdout range and should be treated as optimistic. The repeated grouped metrics
are the preferred reliability signal for the refactored portfolio project, but
they are still not clinical validation.

## Known Limitations

- The legacy evaluation split is sample-level, not patient-level.
- Augmented variants of the same base respiratory cycle can appear across training and validation.
- The grouped split is constrained by very small patient counts in some classes;
  Pneumonia has no validation patient and one test patient in the current split.
- Equal sample balancing hides unequal patient diversity: Pneumonia has only two
  patients and COPD has only three patients in the current balanced subset.
- The model has not been externally validated.
- LLM-generated reports depend on prompt design and must not be treated as medical advice.
- The original notebooks explored multiple LLM providers; the refactored module
  uses generic OpenAI-compatible configuration.
