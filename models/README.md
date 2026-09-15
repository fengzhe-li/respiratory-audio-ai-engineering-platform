# Models

This directory stores lightweight model artifacts for the portfolio version of DeepBreath.

The current artifact, `melspectogramTrained.h5`, is copied from the MSc notebook workflow and is kept with its original misspelled filename to preserve research provenance.

Contract:

- input shape: `128 x 259 x 1`
- output shape: six-class softmax
- class order: Bronchiectasis, Bronchiolitis, COPD, Healthy, Pneumonia, URTI

Future work should export the same architecture to the modern `.keras` format and record the training configuration next to the artifact.
