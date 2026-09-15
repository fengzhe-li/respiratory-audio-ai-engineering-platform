from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf

from deepbreath.ml_contract import AUDIO_DURATION_SECONDS


def get_pure_sample(raw_data, start: float, end: float, sample_rate: int):
    """Return the audio samples between start/end seconds."""

    max_index = len(raw_data)
    start_index = min(int(start * sample_rate), max_index)
    end_index = min(int(end * sample_rate), max_index)
    return raw_data[start_index:end_index]


def pad_or_clip(audio, sample_rate: int, duration_seconds: float = AUDIO_DURATION_SECONDS):
    """Match the notebook behavior: center-pad short cycles and clip long cycles."""

    required_length = int(duration_seconds * sample_rate)
    if len(audio) < required_length:
        return librosa.util.pad_center(data=audio, size=required_length)
    return audio[:required_length]


def segment_cycles(
    cycle_df: pd.DataFrame,
    source_audio_dir: str | Path,
    output_dir: str | Path,
    duration_seconds: float = AUDIO_DURATION_SECONDS,
) -> int:
    """Create one fixed-length WAV file per annotated respiratory cycle."""

    source_audio_dir = Path(source_audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cycle_df = cycle_df.copy()
    cycle_df["start"] = cycle_df["start"].astype(float)
    cycle_df["end"] = cycle_df["end"].astype(float)

    count = 0
    current_cycle_index = 0
    previous_filename = None

    for _, row in cycle_df.iterrows():
        filename = row["filename"]
        current_cycle_index = current_cycle_index + 1 if filename == previous_filename else 0
        previous_filename = filename

        start = row["start"]
        end = min(row["end"], start + duration_seconds)
        audio_path = source_audio_dir / f"{filename}.wav"
        audio, sample_rate = librosa.load(audio_path)
        sample = get_pure_sample(audio, start, end, sample_rate)
        fixed = pad_or_clip(sample, sample_rate, duration_seconds)

        output_path = output_dir / f"{filename}_{current_cycle_index}.wav"
        sf.write(file=output_path, data=fixed, samplerate=sample_rate)
        count += 1

    return count
