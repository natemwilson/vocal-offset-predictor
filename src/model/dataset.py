"""
Dataset handling for word duration prediction.

Loads and preprocesses data from CSV files containing word timing information.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import numpy as np


class WordDurationDataset(Dataset):
    """
    Dataset for word duration prediction.

    Expected CSV format:
        sentence_id, word_id, word, start_time, end_time, offset
    or:
        index, word, word_id, sentence_id, sentence_length, label, prediction

    The 'offset' or 'label' column contains the word duration in seconds.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to CSV file
            tokenizer_name: HuggingFace tokenizer to use
            max_length: Maximum sequence length
            split: One of 'train', 'val', 'test'
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            seed: Random seed for splitting
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Load and preprocess data
        df = pd.read_csv(data_path, header=None if self._is_headerless(data_path) else 0)
        df = self._normalize_columns(df)

        # Group words by sentence
        self.sentences = self._group_by_sentence(df)

        # Split data
        np.random.seed(seed)
        sentence_ids = list(self.sentences.keys())
        np.random.shuffle(sentence_ids)

        n_train = int(len(sentence_ids) * train_ratio)
        n_val = int(len(sentence_ids) * val_ratio)

        if split == "train":
            self.sentence_ids = sentence_ids[:n_train]
        elif split == "val":
            self.sentence_ids = sentence_ids[n_train:n_train + n_val]
        else:  # test
            self.sentence_ids = sentence_ids[n_train + n_val:]

    def _is_headerless(self, path: str) -> bool:
        """Check if CSV has no header (first row is data)."""
        with open(path) as f:
            first_line = f.readline()
            # If first element looks like a number, probably no header
            first_col = first_line.split(",")[0].strip()
            try:
                int(first_col)
                return True
            except ValueError:
                return False

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to standard format."""
        if df.columns.dtype == "int64":
            # Headerless CSV - assign names
            if len(df.columns) == 6:
                # Format: sentence_id, word_id, word, start_time, end_time, offset
                df.columns = ["sentence_id", "word_id", "word", "start_time", "end_time", "offset"]
            elif len(df.columns) == 7:
                # Format: index, word, word_id, sentence_id, sentence_length, label, prediction
                df.columns = ["index", "word", "word_id", "sentence_id", "sentence_length", "label", "prediction"]
                df["offset"] = df["label"]

        # Handle named columns
        if "label" in df.columns and "offset" not in df.columns:
            df["offset"] = df["label"]

        return df

    def _group_by_sentence(self, df: pd.DataFrame) -> Dict[int, List[Tuple[str, float]]]:
        """Group words by sentence ID."""
        sentences = {}

        for sentence_id in df["sentence_id"].unique():
            sentence_df = df[df["sentence_id"] == sentence_id].sort_values("word_id")
            words_and_durations = list(zip(
                sentence_df["word"].astype(str).tolist(),
                sentence_df["offset"].astype(float).tolist()
            ))
            sentences[sentence_id] = words_and_durations

        return sentences

    def __len__(self) -> int:
        return len(self.sentence_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sentence_id = self.sentence_ids[idx]
        words_and_durations = self.sentences[sentence_id]

        words = [w for w, _ in words_and_durations]
        durations = [d for _, d in words_and_durations]

        sentence = " ".join(words)

        # Tokenize
        encoding = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offsets = encoding["offset_mapping"].squeeze(0)

        # Create target durations aligned with tokens
        target_durations = torch.zeros(self.max_length)
        word_mask = torch.zeros(self.max_length)

        # Map tokens to words and assign durations
        word_boundaries = []
        pos = 0
        for word in words:
            start = sentence.find(word, pos)
            if start == -1:
                start = pos
            end = start + len(word)
            word_boundaries.append((start, end))
            pos = end

        for i, (start, end) in enumerate(offsets.tolist()):
            if start == 0 and end == 0:  # Special token
                continue

            # Find which word this token belongs to
            for word_idx, (w_start, w_end) in enumerate(word_boundaries):
                if start >= w_start and start < w_end:
                    if word_idx < len(durations):
                        target_durations[i] = durations[word_idx]
                        word_mask[i] = 1.0
                    break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "durations": target_durations,
            "word_mask": word_mask,
        }


class SimpleDataset(Dataset):
    """
    Simpler dataset for the SimpleDurationPredictor model.
    """

    def __init__(
        self,
        data_path: str,
        max_words: int = 128,
        max_word_len: int = 20,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ):
        self.max_words = max_words
        self.max_word_len = max_word_len

        # Load data
        df = pd.read_csv(data_path, header=None if self._is_headerless(data_path) else 0)
        df = self._normalize_columns(df)
        self.sentences = self._group_by_sentence(df)

        # Split
        np.random.seed(seed)
        sentence_ids = list(self.sentences.keys())
        np.random.shuffle(sentence_ids)

        n_train = int(len(sentence_ids) * train_ratio)
        n_val = int(len(sentence_ids) * val_ratio)

        if split == "train":
            self.sentence_ids = sentence_ids[:n_train]
        elif split == "val":
            self.sentence_ids = sentence_ids[n_train:n_train + n_val]
        else:
            self.sentence_ids = sentence_ids[n_train + n_val:]

        # Build vocab from all data
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        word_counts = {}
        for words_durations in self.sentences.values():
            for word, _ in words_durations:
                word = word.lower()
                word_counts[word] = word_counts.get(word, 0) + 1

        for word, count in sorted(word_counts.items(), key=lambda x: -x[1])[:29998]:
            if count >= 2:
                self.word2idx[word] = len(self.word2idx)

    def _is_headerless(self, path: str) -> bool:
        with open(path) as f:
            first_col = f.readline().split(",")[0].strip()
            try:
                int(first_col)
                return True
            except ValueError:
                return False

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.columns.dtype == "int64":
            if len(df.columns) == 6:
                df.columns = ["sentence_id", "word_id", "word", "start_time", "end_time", "offset"]
            elif len(df.columns) == 7:
                df.columns = ["index", "word", "word_id", "sentence_id", "sentence_length", "label", "prediction"]
                df["offset"] = df["label"]
        if "label" in df.columns and "offset" not in df.columns:
            df["offset"] = df["label"]
        return df

    def _group_by_sentence(self, df: pd.DataFrame) -> Dict[int, List[Tuple[str, float]]]:
        sentences = {}
        for sid in df["sentence_id"].unique():
            sdf = df[df["sentence_id"] == sid].sort_values("word_id")
            sentences[sid] = list(zip(
                sdf["word"].astype(str).tolist(),
                sdf["offset"].astype(float).tolist()
            ))
        return sentences

    def __len__(self) -> int:
        return len(self.sentence_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sid = self.sentence_ids[idx]
        words_durations = self.sentences[sid]

        # Prepare tensors
        word_ids = torch.zeros(self.max_words, dtype=torch.long)
        char_ids = torch.zeros(self.max_words, self.max_word_len, dtype=torch.long)
        lengths = torch.zeros(self.max_words, dtype=torch.long)
        durations = torch.zeros(self.max_words)
        mask = torch.zeros(self.max_words, dtype=torch.bool)

        for i, (word, duration) in enumerate(words_durations[:self.max_words]):
            word_lower = word.lower()

            # Word ID
            word_ids[i] = self.word2idx.get(word_lower, 1)

            # Char IDs
            for j, c in enumerate(word[:self.max_word_len]):
                char_ids[i, j] = min(ord(c), 127)

            # Length
            lengths[i] = min(len(word), 49)

            # Duration
            durations[i] = duration

            # Mask
            mask[i] = True

        return {
            "word_ids": word_ids,
            "char_ids": char_ids,
            "lengths": lengths,
            "durations": durations,
            "mask": mask,
        }


def create_dataloader(
    data_path: str,
    batch_size: int = 32,
    split: str = "train",
    model_type: str = "transformer",
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for training/evaluation.

    Args:
        data_path: Path to CSV data
        batch_size: Batch size
        split: One of 'train', 'val', 'test'
        model_type: 'transformer' or 'simple'
        **kwargs: Additional arguments for dataset

    Returns:
        DataLoader instance
    """
    if model_type == "transformer":
        dataset = WordDurationDataset(data_path, split=split, **kwargs)
    else:
        dataset = SimpleDataset(data_path, split=split, **kwargs)

    shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
