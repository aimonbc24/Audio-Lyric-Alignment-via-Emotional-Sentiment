# Audio-Lyric Alignment via Emotional Sentiment
This repository contains the code for the project "Audio-Lyric Alignment via Emotional Sentiment" by Aimon Benfield-Chand, Javon Hickmon, and Donovan Clay. The project is part of the course "CSE-447: Natural Language Processing" at the University of Washington.

## Abstract
In this research, we investigate the feasibility of using musical sentiment as a method for aligning audio and lyrics in songs, an area that has remained largely unexplored in Music Information Retrieval (MIR) and Music Emotion Recognition (MER). The study utilizes the DALI dataset, comprising over 5000 songs with synchronized audio, lyrics, and notes, to develop a multi-modal text-audio model. This model aims to bridge the gap between the sentiment in song lyrics and its corresponding audio signature. The approach involves segmenting songs into musical sections, generating MEL spectrograms for audio analysis, and applying sentiment analysis to the lyrics, followed by training a Vision Transformer (ViT) using contrastive learning to create a joint embedding space for audio spectrograms and text-based sentiment descriptions. This method diverges from traditional fixed-vocabulary sentiment analysis, allowing for more nuanced and diverse emotion recognition. The potential applications of the resulting model are broad, including sentiment-based song search and recommendation, and enhancement of music composition by aligning lyrical sentiment with audio.

## Root folder contains:
`scripts`: Contains scripts for fine-tuning  (imported or called by scripts outside this folder). Contains all the key implementations

`Dataset`: Folder for the project's data contents. This includes...
- `DALIDataset.py`: the project's audio-lyric dataset implementation
- `utils`: a folder containing utility functions for audio processing
- `mp4`: a folder of the raw MP4 audio files scraped from YouTube. The name of each audio files is the song's DALI ID. (In the future, we will create an easy-to-run data downloader script and remove the raw audio files to reduce storage overhead)
- `*.json`: JSON files containing metadata for each lyric-description text segment used for fine-tuning CLAP.

`notebooks`: Contains jupyter notebooks of prototyping code, included for reproducibility.

## Google Drive
- [Click](https://drive.google.com/drive/u/4/folders/1gaKDPwmVh8GvBVmySpjS11Jrf1zDOZIx) to view.

## Note:
- The code for the project is currently under development and will be released soon.

## Authors:
- Aimon Benfield-Chand
- Donovan Clay
- Javon Hickmon
