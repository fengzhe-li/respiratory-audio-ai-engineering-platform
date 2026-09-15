from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from deepbreath.ml_contract import AUDIO_DURATION_SECONDS, CLASS_NAMES


DEFAULT_CLASSES = list(CLASS_NAMES)


def add_noise(audio, noise_level: float = 0.005):
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise


def pitch_shift(audio, sample_rate: int, n_steps: int):
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)


def time_stretch(audio, speed_factor: float):
    return librosa.effects.time_stretch(audio, rate=speed_factor)


def adjust_audio_duration(audio, sample_rate: int, max_duration: float = AUDIO_DURATION_SECONDS):
    max_length = int(sample_rate * max_duration)
    if len(audio) < max_length:
        return librosa.util.fix_length(data=audio, size=max_length)
    return audio[:max_length]


def augment_processed_audio(
    files_df: pd.DataFrame,
    processed_audio_dir: str | Path,
    target_classes: list[str] | None = None,
    max_duration: float = AUDIO_DURATION_SECONDS,
) -> pd.DataFrame:
    """Create the same five augmentation variants used in the MSc notebook."""

    target_classes = target_classes or DEFAULT_CLASSES
    processed_audio_dir = Path(processed_audio_dir)
    filtered = files_df[files_df["disease"].isin(target_classes)]
    augmented_rows: list[dict[str, object]] = []

    for _, row in filtered.iterrows():
        filename = row["filename"]
        file_path = processed_audio_dir / filename
        if not file_path.exists():
            continue

        audio, sample_rate = librosa.load(file_path)
        base_filename = file_path.stem
        variants = {
            "noise": add_noise(audio),
            "pitch_up": pitch_shift(audio, sample_rate, n_steps=2),
            "pitch_down": pitch_shift(audio, sample_rate, n_steps=-2),
            "speed_up": time_stretch(audio, speed_factor=1.2),
            "speed_down": time_stretch(audio, speed_factor=0.8),
        }

        for suffix, variant_audio in variants.items():
            adjusted = adjust_audio_duration(variant_audio, sample_rate, max_duration)
            output_name = f"{base_filename}_{suffix}.wav"
            sf.write(processed_audio_dir / output_name, adjusted, sample_rate)
            augmented_rows.append(
                {
                    "filename": output_name,
                    "disease": row["disease"],
                    "pid": int(output_name.split("_")[0]),
                }
            )

    return pd.DataFrame(augmented_rows)
