import importlib
import sys
import types
import unittest


class PreprocessingCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self._original_modules = {
            name: sys.modules.get(name) for name in ("librosa", "pandas", "soundfile")
        }

        def pad_center(data, size):
            pad_total = size - len(data)
            left = pad_total // 2
            right = pad_total - left
            return [0] * left + list(data) + [0] * right

        fake_librosa = types.SimpleNamespace(util=types.SimpleNamespace(pad_center=pad_center))
        sys.modules["librosa"] = fake_librosa
        fake_pandas = types.ModuleType("pandas")
        fake_pandas.DataFrame = object
        sys.modules["pandas"] = fake_pandas
        sys.modules["soundfile"] = types.ModuleType("soundfile")
        sys.modules.pop("deepbreath.data.segmentation", None)

    def tearDown(self):
        for name, module in self._original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.modules.pop("deepbreath.data.segmentation", None)

    def test_pad_or_clip_center_pads_short_audio_like_notebook(self):
        segmentation = importlib.import_module("deepbreath.data.segmentation")
        self.assertEqual(segmentation.pad_or_clip([1, 2], sample_rate=1, duration_seconds=5), [0, 1, 2, 0, 0])

    def test_pad_or_clip_clips_long_audio_like_notebook(self):
        segmentation = importlib.import_module("deepbreath.data.segmentation")
        self.assertEqual(segmentation.pad_or_clip([1, 2, 3, 4], sample_rate=1, duration_seconds=2), [1, 2])


if __name__ == "__main__":
    unittest.main()
