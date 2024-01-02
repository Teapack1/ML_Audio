from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

import numpy as np
from transformers import pipeline
import torch
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

device = "cuda:0" if torch.cuda.is_available() else "cpu"

classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=device
)
intent_class_pipe = pipeline(
    "audio-classification", model="anton-l/xtreme_s_xlsr_minds14", device=device
)


async def launch_fn(
    wake_word="marvin",
    prob_threshold=0.5,
    chunk_length_s=2.0,
    stream_chunk_s=0.25,
    debug=False,
):
    if wake_word not in classifier.model.config.label2id.keys():
        raise ValueError(
            f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the set {classifier.model.config.label2id.keys()}."
        )

    sampling_rate = classifier.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Listening for wake word...")
    for prediction in classifier(mic):
        prediction = prediction[0]
        if debug:
            print(prediction)
        if prediction["label"] == wake_word:
            if prediction["score"] > prob_threshold:
                return True


async def listen(chunk_length_s=3.0, stream_chunk_s=3.0):
    sampling_rate = intent_class_pipe.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )
    audio_buffer = []

    print("Listening")
    for i in range(5):
        audio_chunk = next(mic)
        audio_buffer.append(audio_chunk["raw"])

        prediction = intent_class_pipe(audio_chunk["raw"])
        print(prediction)

        if await is_silence(audio_chunk["raw"], threshold=0.7):
            print("Silence detected, processing audio.")
            break

    combined_audio = np.concatenate(audio_buffer)
    prediction = intent_class_pipe(combined_audio)
    prediction = prediction[0]
    print(prediction)
    return prediction


async def is_silence(audio_chunk, threshold):
    silence = intent_class_pipe(audio_chunk)
    if silence[0]["label"] == "silence" and silence[0]["score"] > threshold:
        return True
    else:
        return False


# Initialize FastAPI app
app = FastAPI()

# Set up static file directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 Template for HTML rendering
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await websocket.send_text("Listening for wake word...")
        wake_word_detected = await launch_fn(debug=True)  # Use await here
        if wake_word_detected:
            await websocket.send_text("Wake word detected. Listening for your query...")
            intent_result = await listen()  # Use await here
            await websocket.send_text(f"Intent classified: {intent_result}")
            await websocket.send_text("Restarting system...")
    except WebSocketDisconnect:
        print("Client disconnected.")
