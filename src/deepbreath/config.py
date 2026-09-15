from dataclasses import dataclass
from pathlib import Path

from deepbreath.ml_contract import LEGACY_MODEL_FILENAME


@dataclass(frozen=True)
class DeepBreathPaths:
    """Filesystem locations used by the legacy MSc workflow."""

    project_root: Path
    icbhi_audio_dir: Path
    patient_diagnosis_csv: Path
    processed_audio_dir: Path
    metadata_dir: Path
    model_path: Path

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "DeepBreathPaths":
        root = Path(project_root)
        icbhi_root = root / "Respiratory_Sound_Database" / "Respiratory_Sound_Database"
        return cls(
            project_root=root,
            icbhi_audio_dir=icbhi_root / "audio_and_txt_files",
            patient_diagnosis_csv=icbhi_root / "patient_diagnosis.csv",
            processed_audio_dir=root / "processed_audio_files",
            metadata_dir=root / "csv_data",
            model_path=root / LEGACY_MODEL_FILENAME,
        )
