"""
Word Duration Prediction Model

Uses a transformer encoder to predict the speaking duration of each word
in a sentence. This enables word-by-word subtitle display synchronized
with audio playback.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Optional
import json
from pathlib import Path


class WordDurationPredictor(nn.Module):
    """
    Predicts the speaking duration (in seconds) for each word in a sentence.

    Uses a pre-trained transformer encoder with a regression head to predict
    word durations based on the word itself and its surrounding context.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        hidden_size: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()

        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Regression head for duration prediction
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensures positive output (duration can't be negative)
        )

        self._hidden_size = hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to predict word durations.

        Args:
            input_ids: Token IDs from tokenizer [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            word_ids: Mapping from tokens to words [batch, seq_len]
                      Used to aggregate subword predictions

        Returns:
            durations: Predicted duration for each token [batch, seq_len]
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get token-level representations
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Predict duration for each token
        durations = self.duration_head(hidden_states).squeeze(-1)  # [batch, seq_len]

        return durations

    def predict_sentence(self, sentence: str, device: str = "cpu") -> List[Tuple[str, float]]:
        """
        Predict durations for each word in a sentence.

        Args:
            sentence: Input sentence
            device: Device to run inference on

        Returns:
            List of (word, duration) tuples
        """
        self.eval()
        self.to(device)

        # Tokenize with word mapping
        words = sentence.split()
        encoding = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        offsets = encoding["offset_mapping"][0].tolist()

        # Get predictions
        with torch.no_grad():
            token_durations = self(input_ids, attention_mask)[0].cpu().numpy()

        # Map token predictions back to words
        word_durations = []
        current_word_idx = 0
        current_word_duration = 0.0
        current_word_start = 0

        # Find word boundaries in the sentence
        word_boundaries = []
        pos = 0
        for word in words:
            start = sentence.find(word, pos)
            end = start + len(word)
            word_boundaries.append((start, end))
            pos = end

        # Aggregate token durations per word
        for i, (start, end) in enumerate(offsets):
            if start == 0 and end == 0:  # Special tokens
                continue

            # Find which word this token belongs to
            for word_idx, (w_start, w_end) in enumerate(word_boundaries):
                if start >= w_start and end <= w_end:
                    # This token belongs to this word
                    while len(word_durations) < word_idx:
                        # Fill in any skipped words
                        word_durations.append(0.0)

                    if len(word_durations) == word_idx:
                        word_durations.append(token_durations[i])
                    else:
                        # Add duration of subword token to word total
                        word_durations[word_idx] += token_durations[i]
                    break

        # Pad if necessary
        while len(word_durations) < len(words):
            word_durations.append(0.3)  # Default duration

        return list(zip(words, word_durations[:len(words)]))

    def predict_batch(
        self,
        sentences: List[str],
        device: str = "cpu"
    ) -> List[List[Tuple[str, float]]]:
        """
        Predict durations for multiple sentences.

        Args:
            sentences: List of sentences
            device: Device for inference

        Returns:
            List of (word, duration) lists for each sentence
        """
        return [self.predict_sentence(s, device) for s in sentences]

    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.state_dict(), path / "model.pt")

        # Save config
        config = {
            "model_name": self.model_name,
            "hidden_size": self._hidden_size,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "WordDurationPredictor":
        """Load model from disk."""
        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config = json.load(f)

        # Create model
        model = cls(**config)

        # Load weights
        model.load_state_dict(
            torch.load(path / "model.pt", map_location=device)
        )

        return model


class SimpleDurationPredictor(nn.Module):
    """
    A simpler, faster model for word duration prediction.

    Uses character-level and word-level features instead of a full
    transformer, making it much faster for inference.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        char_vocab_size: int = 128,
        embedding_dim: int = 128,
        char_embedding_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)

        # Character-level CNN for word representation
        self.char_cnn = nn.Sequential(
            nn.Conv1d(char_embedding_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        # Length feature embedding
        self.length_embedding = nn.Embedding(50, 16)  # Words up to 50 chars

        # Context encoder (BiLSTM)
        self.context_encoder = nn.LSTM(
            input_size=embedding_dim + 64 + 16,  # word + char + length
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Duration prediction head
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Softplus()
        )

        # Word to index mapping (built during training)
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self._vocab_size = vocab_size
        self._char_vocab_size = char_vocab_size
        self._embedding_dim = embedding_dim
        self._char_embedding_dim = char_embedding_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._dropout = dropout

    def build_vocab(self, sentences: List[str], min_freq: int = 2):
        """Build vocabulary from training sentences."""
        from collections import Counter

        word_counts = Counter()
        for sentence in sentences:
            for word in sentence.lower().split():
                word_counts[word] += 1

        # Add frequent words to vocab
        for word, count in word_counts.most_common(self._vocab_size - 2):
            if count >= min_freq:
                self.word2idx[word] = len(self.word2idx)

    def encode_words(
        self,
        sentences: List[str],
        max_len: int = 128,
        max_word_len: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode sentences to tensors.

        Returns:
            word_ids: [batch, max_len]
            char_ids: [batch, max_len, max_word_len]
            lengths: [batch, max_len]
            mask: [batch, max_len]
        """
        batch_size = len(sentences)

        word_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        char_ids = torch.zeros(batch_size, max_len, max_word_len, dtype=torch.long)
        lengths = torch.zeros(batch_size, max_len, dtype=torch.long)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()[:max_len]

            for j, word in enumerate(words):
                # Word ID
                word_ids[i, j] = self.word2idx.get(word, 1)  # 1 = <unk>

                # Character IDs
                for k, char in enumerate(word[:max_word_len]):
                    char_ids[i, j, k] = min(ord(char), self._char_vocab_size - 1)

                # Length
                lengths[i, j] = min(len(word), 49)

                # Mask
                mask[i, j] = True

        return word_ids, char_ids, lengths, mask

    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            word_ids: [batch, seq_len]
            char_ids: [batch, seq_len, word_len]
            lengths: [batch, seq_len]
            mask: [batch, seq_len]

        Returns:
            durations: [batch, seq_len]
        """
        batch_size, seq_len, word_len = char_ids.shape

        # Word embeddings
        word_emb = self.word_embedding(word_ids)  # [batch, seq, emb]

        # Character embeddings + CNN
        char_emb = self.char_embedding(char_ids)  # [batch, seq, word_len, char_emb]
        char_emb = char_emb.view(batch_size * seq_len, word_len, -1)
        char_emb = char_emb.permute(0, 2, 1)  # [batch*seq, char_emb, word_len]
        char_features = self.char_cnn(char_emb).squeeze(-1)  # [batch*seq, 64]
        char_features = char_features.view(batch_size, seq_len, -1)

        # Length embeddings
        length_emb = self.length_embedding(lengths)  # [batch, seq, 16]

        # Combine features
        combined = torch.cat([word_emb, char_features, length_emb], dim=-1)

        # Context encoding
        context, _ = self.context_encoder(combined)

        # Predict durations
        durations = self.duration_head(context).squeeze(-1)

        # Mask out padding
        durations = durations * mask.float()

        return durations

    def predict_sentence(self, sentence: str, device: str = "cpu") -> List[Tuple[str, float]]:
        """Predict durations for a single sentence."""
        self.eval()
        self.to(device)

        words = sentence.split()
        word_ids, char_ids, lengths, mask = self.encode_words([sentence])

        word_ids = word_ids.to(device)
        char_ids = char_ids.to(device)
        lengths = lengths.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            durations = self(word_ids, char_ids, lengths, mask)[0].cpu().numpy()

        return [(word, float(durations[i])) for i, word in enumerate(words)]

    def save(self, path: str):
        """Save model and vocab."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), path / "model.pt")

        config = {
            "vocab_size": self._vocab_size,
            "char_vocab_size": self._char_vocab_size,
            "embedding_dim": self._embedding_dim,
            "char_embedding_dim": self._char_embedding_dim,
            "hidden_dim": self._hidden_dim,
            "num_layers": self._num_layers,
            "dropout": self._dropout,
        }

        with open(path / "config.json", "w") as f:
            json.dump(config, f)

        with open(path / "vocab.json", "w") as f:
            json.dump(self.word2idx, f)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SimpleDurationPredictor":
        """Load model from disk."""
        path = Path(path)

        with open(path / "config.json") as f:
            config = json.load(f)

        model = cls(**config)
        model.load_state_dict(
            torch.load(path / "model.pt", map_location=device)
        )

        with open(path / "vocab.json") as f:
            model.word2idx = json.load(f)

        return model
