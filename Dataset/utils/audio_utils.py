import torchaudio
import subprocess
import os
import matplotlib.pyplot as plt

def mp4_to_wav(input_file, output_file):
    with open(os.devnull, 'w') as FNULL:
        return subprocess.call(['ffmpeg', '-i', input_file, output_file], stdout=FNULL, stderr=subprocess.STDOUT)

def load_audio_segment(filepath, start_sec, end_sec):
    wav_file = filepath.replace(".mp4", ".wav")

    # Convert mp4 to wav
    mp4_to_wav(filepath, wav_file)

    # Load the full wav file
    waveform, sample_rate = torchaudio.load(wav_file)

    # remove wav
    os.remove(wav_file)
    
    # Calculate start and end sample indices
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    
    # Slice the waveform to get the desired segment
    segment = waveform[:, start_sample:end_sample]
    
    return segment, sample_rate

def visualize_spectrogram(spectrogram):
    plt.figure(figsize=(10, 5))
    plt.imshow(spectrogram.log2()[0, :, :].numpy(), aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()