import sys
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
import torch

#model_id = "distil-whisper/distil-medium.en"
model_id = "openai/whisper-tiny.en"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

pipe = pipeline("automatic-speech-recognition", model=model_id, device=device)
sampling_rate = pipe.feature_extractor.sampling_rate


chunk_length_s = 2  # how often returns the text
stream_chunk_s = 0.25  # how often the microphone is checked for new audio
mic = ffmpeg_microphone_live(
    sampling_rate=sampling_rate,
    chunk_length_s=chunk_length_s,
    stream_chunk_s=stream_chunk_s,
)
print("Start talking...")
for item in pipe(mic):
    sys.stdout.write("\033[K")
    print(item["text"], end="\r")
    if not item["partial"][0]:
        print("")
