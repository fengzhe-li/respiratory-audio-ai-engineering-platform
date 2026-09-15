import unittest

import pandas as pd

from deepbreath.data.splits import (
    assert_no_group_leakage,
    base_cycle_id,
    patient_level_grouped_split,
    prepare_grouped_metadata,
    validate_grouped_split,
)


class GroupedPatientSplitTests(unittest.TestCase):
    def test_base_cycle_id_groups_augmented_variants(self):
        self.assertEqual(base_cycle_id("101_1b1_Al_sc_Meditron_0.wav"), "101_1b1_Al_sc_Meditron_0")
        self.assertEqual(base_cycle_id("101_1b1_Al_sc_Meditron_0_noise.wav"), "101_1b1_Al_sc_Meditron_0")
        self.assertEqual(
            base_cycle_id("101_1b1_Al_sc_Meditron_0_pitch_down.wav"),
            "101_1b1_Al_sc_Meditron_0",
        )
        self.assertEqual(
            base_cycle_id("101_1b1_Al_sc_Meditron_0_speed_up.wav"),
            "101_1b1_Al_sc_Meditron_0",
        )

    def test_real_metadata_has_expected_grouping_columns(self):
        metadata = prepare_grouped_metadata("data/metadata_sample/label_mapping_new.csv")
        self.assertTrue({"pid", "base_cycle_id", "disease", "is_augmented"}.issubset(metadata.columns))
        self.assertEqual(metadata["pid"].nunique(), 33)
        self.assertEqual(metadata["base_cycle_id"].nunique(), 624)
        self.assertFalse(metadata.groupby("pid")["disease"].nunique().gt(1).any())

    def test_grouped_split_has_no_patient_or_base_cycle_leakage(self):
        metadata = prepare_grouped_metadata("data/metadata_sample/label_mapping_new.csv")
        split_metadata = patient_level_grouped_split(metadata, random_state=42)
        summary = validate_grouped_split(split_metadata)

        self.assertEqual(sum(summary["patient_overlap_counts"].values()), 0)
        self.assertEqual(sum(summary["base_cycle_overlap_counts"].values()), 0)
        assert_no_group_leakage(split_metadata)

    def test_grouped_split_is_deterministic_with_fixed_seed(self):
        metadata = prepare_grouped_metadata("data/metadata_sample/label_mapping_new.csv")
        first = patient_level_grouped_split(metadata, random_state=42).sort_values("filename")
        second = patient_level_grouped_split(metadata, random_state=42).sort_values("filename")

        pd.testing.assert_series_equal(
            first["split"].reset_index(drop=True),
            second["split"].reset_index(drop=True),
        )

    def test_metadata_rejects_patient_with_multiple_labels(self):
        bad_metadata = pd.DataFrame(
            {
                "filename": ["101_a_0.wav", "101_a_1.wav"],
                "disease": ["Healthy", "URTI"],
                "pid": [101, 101],
                "base_cycle_id": ["101_a_0", "101_a_1"],
            }
        )
        with self.assertRaises(ValueError):
            patient_level_grouped_split(bad_metadata)


if __name__ == "__main__":
    unittest.main()
