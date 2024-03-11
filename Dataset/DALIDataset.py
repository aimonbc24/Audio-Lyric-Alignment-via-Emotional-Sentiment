# Implementation of DALI dataset for PyTorch
import torch
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram
import torchaudio

# import DALI
# from pytube import YouTube
import json
from typing import Tuple
import os

from .utils.audio_utils import load_audio_segment
from .utils.constants import AUDIO_LENGTH

class DALIDataset(Dataset):
    """DALI dataset for Audio-Lyric Song Alignment.
    
    Items are given as Tuples:
        (line, sentiment_description, spectrogram)
    
    line:
        A Tuple of length batch_size containing the lyrics of each song segment.
    sentiment_description:
        A Tuple of length batch_size containing the sentiment of each song segment.
    spectrogram:
        A Tensor of shape (batch_size, n_mels, time) containing the MelSpectrogram of each song segment."""

    def __init__(self, use_sentiment: bool = False):
        # get the absolute path of the current working directory
        self.absolute_path = os.path.dirname(os.path.abspath(__file__))
        self.mp4_folder = os.path.join(self.absolute_path, "data", "mp4")
        self.use_sentiment = use_sentiment
        self.spectrogram_transform = MelSpectrogram()

        # read in the segments.json file
        with open(os.path.join(self.absolute_path, "data", "segments_filtered.json"), "r") as f:
            self.items = json.load(f) # stored as a list of dictionaries

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Tuple[str, str, torch.Tensor]:
        item_dict = self.items[idx]
        # 'id', 'artist', 'title', 'youtube', 'line', 'start_time', 'end_time'

        mp4_file = os.path.join(self.mp4_folder, f"{item_dict['id']}.mp4")

        wav_file = load_audio_segment(mp4_file, item_dict['start_time'], item_dict['end_time'])
        # go to the next item if none
        if wav_file is None:
            return self.__getitem__(idx + 1)

        if self.use_sentiment:
            output = (
                item_dict['line'],
                item_dict['sentiment'],
                wav_file,
            )
        else:
            output = (
                item_dict['line'],
                wav_file,
            )

        return output

