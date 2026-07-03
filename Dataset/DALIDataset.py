# Implementation of the DALI dataset for PyTorch.
import json
import os
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

from .utils.audio_utils import load_audio_segment

# Default source has BOTH the raw lyric ("line") and the LLM-generated
# emotional-sentiment description ("sentiment") for each ~10s segment, so the
# sentiment-vs-lyric ablation runs on the SAME segments — just flip use_sentiment.
DEFAULT_SEGMENTS = "segments_with_descriptions.json"


class DALIDataset(Dataset):
    """DALI dataset for audio-lyric alignment.

    Each item is a ``(text, (waveform, sample_rate))`` pair, where ``text`` is
    either the segment's raw lyric or its emotional-sentiment description,
    selected by ``use_sentiment``. Returning a single text field (rather than
    both) keeps one collate path for both ablation arms.
    """

    def __init__(self, use_sentiment: bool = False,
                 segments_file: Optional[str] = None):
        self.absolute_path = os.path.dirname(os.path.abspath(__file__))
        self.mp4_folder = os.path.join(self.absolute_path, "data", "mp4")
        self.use_sentiment = use_sentiment
        self.text_key = "sentiment" if use_sentiment else "line"

        segments_file = segments_file or DEFAULT_SEGMENTS
        with open(os.path.join(self.absolute_path, "data", segments_file)) as f:
            self.items = json.load(f)  # list of dicts

        if use_sentiment and self.items and "sentiment" not in self.items[0]:
            raise KeyError(
                f"{segments_file} has no 'sentiment' field; use "
                "segments_with_descriptions.json for the sentiment arm."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx) -> Tuple[str, Tuple[torch.Tensor, int]]:
        item = self.items[idx]
        # keys: id, artist, title, youtube, line, sentiment, start_time, end_time
        mp4_file = os.path.join(self.mp4_folder, f"{item['id']}.mp4")
        audio = load_audio_segment(mp4_file, item["start_time"], item["end_time"])
        if audio is None:
            # Skip unreadable audio; wrap with modulo to avoid an IndexError at the tail.
            return self.__getitem__((idx + 1) % len(self.items))
        return item[self.text_key], audio
