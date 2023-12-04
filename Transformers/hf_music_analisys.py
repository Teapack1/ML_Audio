import numpy as np
from transformers import pipeline
import torchaudio
import torch

song = "D:\\Code\\ProjectsPython\\ML_TrainingGround\\ML_Audio\\data\\Often(KygoRemix).mp3"
waveform, sample_rate = torchaudio.load(song)
waveform = torch.mean(waveform, dim=0)
num_samples_for_30_sec = sample_rate * 30
waveform = waveform[:num_samples_for_30_sec]

audio = waveform.numpy()


pipe = pipeline("audio-classification", trust_remote_code=True, model="mtg-upf/discogs-maest-30s-pw-73e-ts")
print(pipe(audio))