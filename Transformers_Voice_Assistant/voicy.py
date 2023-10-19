import datetime
import sys
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device=0)
sampling_rate = pipe.feature_extractor.sampling_rate


start = datetime.datetime.now()

chunk_length_s = 4
stream_chunk_s = 0.25
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