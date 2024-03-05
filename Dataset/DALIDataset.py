# Implementation of DALI dataset for PyTorch
import torch
from torch.utils.data import Dataset

# import DALI
from pytube import YouTube
import json
from typing import Tuple
import os

from .utils.audio_utils import mp4_to_wav, load_audio_segment, generate_spectrogram


class DALIDataset(Dataset):
    """DALI dataset for Audio-Lyric Song Alignment.
    
    Items are stored as Tuples:
        (line, sentiment_description, spectrogram)"""

    def __init__(self):
        # get the absolute path of the current working directory
        self.absolute_path = os.path.dirname(os.path.abspath(__file__))

        # read in the segments.json file
        with open(os.path.join(self.absolute_path, "data", "segments.json"), "r") as f:
            self.items = json.load(f)
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Tuple[str, str, torch.Tensor]:
        item_dict = self.items[idx]
        # 'id', 'artist', 'title', 'youtube', 'line', 'start_time', 'end_time'

        yt = YouTube(f"https://www.youtube.com/watch?v={item_dict['youtube']}")

        mp4_folder = os.path.join(self.absolute_path, "data")

        # download mp4
        yt.streams.filter(only_audio=True).first().download(output_path=mp4_folder, filename=f"{idx}.mp4")

        mp4_file = os.path.join(mp4_folder, f"{idx}.mp4")
        wav_file = os.path.join(self.absolute_path, "data", f"{idx}.wav")

        # convert mp4 to wav
        mp4_to_wav(mp4_file, wav_file)

        # remove mp4
        os.remove(mp4_file)

        # load wav using librosa
        y, sr = load_audio_segment(wav_file, item_dict['start_time'], item_dict['end_time'])

        # generate spectrogram
        spectrogram = generate_spectrogram(y, sr)

        # remove wav
        os.remove(wav_file)

        # TODO: Get sentiment description
        sentiment_description = None

        output = (
            item_dict['line'],
            #sentiment_description,
            spectrogram,
        )

        return output
    
