# Grouped Repeated Evaluation Reliability Report

This report preserves the legacy MSc result and the Phase 3B grouped result. It investigates patient-independent generalisation stability across fixed grouped holdout seeds.

## Grouped CV Feasibility

Grouped cross-validation is not used because several diseases have too few patient groups for statistically meaningful StratifiedGroupKFold. Pneumonia has only two patients, so repeated grouped holdout is a clearer reliability probe without pretending fold-level class balance is robust.

## Metric Stability

| metric | mean | std | min | max | ci95_half_width |
| --- | --- | --- | --- | --- | --- |
| accuracy | 0.3695 | 0.0515 | 0.2971 | 0.4148 | 0.0452 |
| balanced_accuracy | 0.4157 | 0.0874 | 0.2686 | 0.4750 | 0.0766 |
| macro_f1 | 0.3295 | 0.0687 | 0.2476 | 0.4126 | 0.0603 |
| weighted_f1 | 0.3451 | 0.0480 | 0.2733 | 0.3917 | 0.0421 |

## Per-Run Metrics

| seed | train_patients | validation_patients | test_patients | train_samples | validation_samples | test_samples | accuracy | balanced_accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 22 | 5 | 6 | 2244 | 480 | 1020 | 0.2971 | 0.2686 | 0.2476 | 0.2733 |
| 21 | 22 | 5 | 6 | 2484 | 450 | 810 | 0.4123 | 0.4630 | 0.3423 | 0.3917 |
| 42 | 22 | 5 | 6 | 2052 | 918 | 774 | 0.3876 | 0.4750 | 0.3727 | 0.3868 |
| 84 | 22 | 5 | 6 | 2382 | 294 | 1068 | 0.4148 | 0.4702 | 0.4126 | 0.3381 |
| 168 | 22 | 5 | 6 | 2172 | 630 | 942 | 0.3355 | 0.4019 | 0.2724 | 0.3356 |

## Patient Distribution Before Balancing

| disease | patients | base_cycles | samples | augmented_samples |
| --- | --- | --- | --- | --- |
| Bronchiectasis | 7 | 104 | 624 | 520 |
| Bronchiolitis | 4 | 104 | 624 | 520 |
| COPD | 3 | 104 | 624 | 520 |
| Healthy | 9 | 104 | 624 | 520 |
| Pneumonia | 2 | 104 | 624 | 520 |
| URTI | 8 | 104 | 624 | 520 |

## Error Analysis

Strongest mean-F1 classes: Pneumonia, Bronchiectasis.
Weakest mean-F1 classes: COPD, Bronchiolitis.

Most common off-diagonal confusion pairs:

| true_label | predicted_label | count |
| --- | --- | --- |
| COPD | Pneumonia | 598 |
| Bronchiolitis | Healthy | 302 |
| Bronchiolitis | URTI | 263 |
| Pneumonia | URTI | 245 |
| Pneumonia | Healthy | 231 |
| COPD | Bronchiectasis | 166 |
| URTI | Healthy | 143 |
| URTI | Bronchiolitis | 128 |
| Healthy | URTI | 100 |
| Pneumonia | Bronchiolitis | 93 |

## Interpretation

Equal sample counts do not imply equal patient diversity. The current balancing strategy creates 624 samples per disease, but some classes are represented by only two or three patients. Augmentation expands respiratory cycles from the same patients, so it can help the CNN learn audio transformations while still leaving patient-specific acoustic patterns as a likely source of overfitting.

The Phase 3B single grouped result reported accuracy 0.4535, balanced accuracy 0.4391, macro F1 0.3483, and weighted F1 0.4442. In this repeated evaluation, accuracy ranges from 0.2971 to 0.4148. The Phase 3B accuracy is therefore above the repeated deterministic holdout range and should be treated as an optimistic single-run outcome rather than a stable estimate.

These results should not be presented as clinical reliability claims.
