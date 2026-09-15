# Research Summary

The MSc research paper frames DeepBreath as a hybrid CNN and LLM respiratory sound analysis system.

Key research components:

- ICBHI respiratory sound data ingestion.
- Respiratory cycle segmentation from annotation files.
- Six-second audio standardization.
- Audio augmentation for class balancing.
- Mel-spectrogram feature extraction.
- CNN-based disease classification.
- Patient-level crackle and wheeze aggregation.
- LLM-generated patient and clinical reports.

The Phase 1 repository restructuring keeps this research architecture intact while making the implementation easier to inspect and extend.

## Public Repository Notes

The notebooks in `notebooks/archive/research/` are cleaned public-facing copies:
outputs are cleared, exposed API keys are redacted, and machine-specific local
paths are replaced with placeholders. Untouched raw originals should be kept
outside the public repository when provenance is required.

The MSc paper PDF is kept as a private local reference only; it is not committed
or published in this repository because its cover page identifies the student
group by student number. It contains research-context wording that does not
always match the final code exactly, including a textual reference to an
eight-neuron output layer. The implemented CNN is a six-class model.
