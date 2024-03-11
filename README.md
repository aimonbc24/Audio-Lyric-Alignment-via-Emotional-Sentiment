# Audio-to-Language-Alignment-via-Musical-Sentiment
This repository contains the code for the project "Audio-to-Language Alignment via Musical Sentiment" by Aimon Benfield-Chand, Javon Hickmon, and Donovan Clay. The project is part of the course "CSE-447: Natural Language Processing" at the University of Washington.

# Abstract
In this research, we investigate the feasibility of using musical sentiment as a method for aligning audio and lyrics in songs, an area that has remained largely unexplored in Music Information Retrieval (MIR) and Music Emotion Recognition (MER). The study utilizes the DALI dataset, comprising over 5000 songs with synchronized audio, lyrics, and notes, to develop a multi-modal text-audio model. This model aims to bridge the gap between the sentiment in song lyrics and its corresponding audio signature. The approach involves segmenting songs into musical sections, generating MEL spectrograms for audio analysis, and applying sentiment analysis to the lyrics, followed by training a Vision Transformer (ViT) using contrastive learning to create a joint embedding space for audio spectrograms and text-based sentiment descriptions. This method diverges from traditional fixed-vocabulary sentiment analysis, allowing for more nuanced and diverse emotion recognition. The potential applications of the resulting model are broad, including sentiment-based song search and recommendation, and enhancement of music composition by aligning lyrical sentiment with audio.

**Important**:
Please run all scripts (*.py) from the .\scripts\ directory of the project.

# Note:
The code for the project is currently under development and will be released soon.

# Authors:
- Aimon Benfield-Chand
- Donovan Clay
- Javon Hickmon