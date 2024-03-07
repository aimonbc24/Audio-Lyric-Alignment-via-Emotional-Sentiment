import torchaudio
import soundfile
from torchaudio.transforms import Spectrogram
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt

def mp4_to_wav(input_file, output_file):
    with open(os.devnull, 'w') as FNULL:
        return subprocess.call(['ffmpeg', '-i', input_file, output_file], stdout=FNULL, stderr=subprocess.STDOUT)

def load_audio_segment(filepath, start_sec, end_sec):
    # Load the full audio file
    waveform, sample_rate = torchaudio.load(filepath)
    
    # Calculate start and end sample indices
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    
    # Slice the waveform to get the desired segment
    segment = waveform[:, start_sample:end_sample]
    
    return segment, sample_rate

def generate_spectrogram(waveform, sample_rate):
    # Initialize the Spectrogram transformer
    spectrogram_transformer = Spectrogram()
    
    # Generate spectrogram
    spectrogram = spectrogram_transformer(waveform)
    
    return spectrogram


def visualize_spectrogram(spectrogram):
    plt.figure(figsize=(10, 5))
    plt.imshow(spectrogram.log2()[0, :, :].numpy(), aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()