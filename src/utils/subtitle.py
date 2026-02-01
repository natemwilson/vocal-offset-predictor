"""
Subtitle parsing and timing utilities.

Parses SRT subtitle files and applies word-by-word timing predictions.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path


@dataclass
class SubtitleWord:
    """A single word with its timing information."""
    text: str
    start_time: float  # seconds
    end_time: float    # seconds
    duration: float    # seconds

    @property
    def display_time(self) -> str:
        """Format time for display."""
        return f"{self.start_time:.2f}s - {self.end_time:.2f}s"


@dataclass
class SubtitleEntry:
    """A subtitle entry (one line/block in the subtitle file)."""
    index: int
    start_time: float  # seconds
    end_time: float    # seconds
    text: str
    words: List[SubtitleWord] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def parse_time(time_str: str) -> float:
    """
    Parse SRT timestamp to seconds.

    Format: HH:MM:SS,mmm or HH:MM:SS.mmm
    """
    # Replace comma with period for consistency
    time_str = time_str.replace(",", ".")

    parts = time_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {time_str}")

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])

    return hours * 3600 + minutes * 60 + seconds


def format_time(seconds: float) -> str:
    """
    Format seconds as SRT timestamp.

    Returns: HH:MM:SS,mmm
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def parse_srt(content: str) -> List[SubtitleEntry]:
    """
    Parse SRT subtitle content.

    Args:
        content: Raw SRT file content

    Returns:
        List of SubtitleEntry objects
    """
    entries = []

    # Split into blocks (entries are separated by blank lines)
    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        try:
            # First line: index
            index = int(lines[0].strip())

            # Second line: timestamps
            time_match = re.match(
                r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
                lines[1].strip()
            )
            if not time_match:
                continue

            start_time = parse_time(time_match.group(1))
            end_time = parse_time(time_match.group(2))

            # Remaining lines: text
            text = " ".join(lines[2:]).strip()
            # Clean up HTML tags and formatting
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"\{[^}]+\}", "", text)

            entries.append(SubtitleEntry(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text
            ))

        except (ValueError, IndexError):
            continue

    return entries


def parse_srt_file(path: str) -> List[SubtitleEntry]:
    """
    Parse SRT file from path.

    Args:
        path: Path to SRT file

    Returns:
        List of SubtitleEntry objects
    """
    path = Path(path)

    # Try different encodings
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            content = path.read_text(encoding=encoding)
            return parse_srt(content)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode file: {path}")


def apply_word_timing(
    entries: List[SubtitleEntry],
    predictor,
    device: str = "cpu"
) -> List[SubtitleEntry]:
    """
    Apply word-by-word timing predictions to subtitle entries.

    Args:
        entries: List of SubtitleEntry objects
        predictor: Trained duration prediction model
        device: Device for inference

    Returns:
        Updated entries with word timing information
    """
    for entry in entries:
        if not entry.text.strip():
            continue

        # Get predictions for this subtitle line
        word_durations = predictor.predict_sentence(entry.text, device=device)

        # Normalize durations to fit within the entry's time window
        total_predicted = sum(d for _, d in word_durations)
        entry_duration = entry.duration

        if total_predicted > 0:
            scale_factor = entry_duration / total_predicted
        else:
            scale_factor = 1.0

        # Create timed words
        current_time = entry.start_time
        entry.words = []

        for word, duration in word_durations:
            scaled_duration = duration * scale_factor

            entry.words.append(SubtitleWord(
                text=word,
                start_time=current_time,
                end_time=current_time + scaled_duration,
                duration=scaled_duration
            ))

            current_time += scaled_duration

    return entries


def apply_uniform_timing(entries: List[SubtitleEntry]) -> List[SubtitleEntry]:
    """
    Apply uniform timing to words (baseline comparison).

    Each word gets equal time within the subtitle duration.

    Args:
        entries: List of SubtitleEntry objects

    Returns:
        Updated entries with uniform word timing
    """
    for entry in entries:
        if not entry.text.strip():
            continue

        words = entry.text.split()
        if not words:
            continue

        duration_per_word = entry.duration / len(words)
        current_time = entry.start_time

        entry.words = []
        for word in words:
            entry.words.append(SubtitleWord(
                text=word,
                start_time=current_time,
                end_time=current_time + duration_per_word,
                duration=duration_per_word
            ))
            current_time += duration_per_word

    return entries


def entries_to_json(entries: List[SubtitleEntry]) -> List[dict]:
    """
    Convert entries to JSON-serializable format.

    Args:
        entries: List of SubtitleEntry objects

    Returns:
        List of dictionaries
    """
    result = []

    for entry in entries:
        entry_dict = {
            "index": entry.index,
            "start_time": entry.start_time,
            "end_time": entry.end_time,
            "text": entry.text,
            "words": [
                {
                    "text": w.text,
                    "start_time": w.start_time,
                    "end_time": w.end_time,
                    "duration": w.duration
                }
                for w in entry.words
            ]
        }
        result.append(entry_dict)

    return result


def write_timed_srt(
    entries: List[SubtitleEntry],
    output_path: str,
    word_by_word: bool = True
):
    """
    Write entries to SRT file.

    If word_by_word is True, creates one subtitle entry per word.
    Otherwise, writes standard SRT format.

    Args:
        entries: List of SubtitleEntry objects
        output_path: Path for output file
        word_by_word: Whether to create per-word entries
    """
    path = Path(output_path)

    lines = []
    index = 1

    for entry in entries:
        if word_by_word and entry.words:
            # One entry per word
            for word in entry.words:
                lines.append(str(index))
                lines.append(f"{format_time(word.start_time)} --> {format_time(word.end_time)}")
                lines.append(word.text)
                lines.append("")
                index += 1
        else:
            # Standard format
            lines.append(str(index))
            lines.append(f"{format_time(entry.start_time)} --> {format_time(entry.end_time)}")
            lines.append(entry.text)
            lines.append("")
            index += 1

    path.write_text("\n".join(lines), encoding="utf-8")
