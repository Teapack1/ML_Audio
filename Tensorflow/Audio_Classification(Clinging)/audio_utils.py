# Helper code to read wav files and extract properties
import struct
from typing import Any
import librosa
import numpy as np


def read_file_properties(filename):

    wave_file = open(filename, "rb")
    
    riff = wave_file.read(12)
    fmt = wave_file.read(36)
    
    num_channels_string = fmt[10:12]
    num_channels = struct.unpack('<H', num_channels_string)[0]

    sample_rate_string = fmt[12:16]
    sample_rate = struct.unpack("<I", sample_rate_string)[0]
    
    bit_depth_string = fmt[22:24]
    bit_depth = struct.unpack("<H", bit_depth_string)[0]
    
    wave_file.close()

    # Load the audio file with librosa
    y, sr = librosa.load(filename, sr=None, mono=True)  # Load as mono

    # Compute RMS of the audio signal using librosa
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms)  # Average RMS over time if needed
    #avg_rms = None
    # Compute the length of the audio sample in seconds
    length_in_seconds = len(y) / sr  # Total samples / Sample rate
    
    # Length in samples
    length_in_frames = len(y)
    
    return (num_channels, sample_rate, bit_depth, avg_rms, length_in_seconds, length_in_frames)  # Added length_in_samples

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean
