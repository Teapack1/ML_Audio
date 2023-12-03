import os
import argparse
from inference_class import SoundClassificationService

DATA_DIR = "data"
MODEL_PATH = os.path.join("model", "model.keras")
LABELER_PATH = os.path.join("model", "label_encoder.joblib")

AUDIO_CHUNK = 3 # seconds
NUM_CHANNELS = 1
SAMPLE_RATE = 44100

MEL_FRAMES = 35
N_MELS = 256
NFFT = 2048
FMAX = 44100 // 2
HOP_LENGTH = 512


def main():
    parser = argparse.ArgumentParser(description="Audio Classification Service")
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--labeler_path", type=str, default=LABELER_PATH, help="Path to labeler file"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=SAMPLE_RATE, help="Audio sample rate"
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=NUM_CHANNELS,
        help="Number of audio channels",
    )
    parser.add_argument(
        "--audio_chunk",
        type=float,
        default=AUDIO_CHUNK,
        help="Length of audio slice in seconds",
    )
    parser.add_argument(
        "--num_mels", type=int, default=N_MELS, help="Number of Mel bands to generate"
    )
    parser.add_argument(
        "--n_fft", type=int, default=NFFT, help="Number of samples in each FFT window"
    )
    parser.add_argument(
        "--fmax",
        type=int,
        default=FMAX,
        help="Maximum frequency when computing MEL spectrograms",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=HOP_LENGTH,
        help="Number of samples between successive FFT windows",
    )
    parser.add_argument(
        "--mel_frames",
        type=int,
        default=MEL_FRAMES,
        help="Number of samples between successive FFT windows",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for making predictions",
    )
    parser.add_argument(
        "--listening_hop_length",
        type=float,
        default=0.5,
        help="Hop length for listening in seconds",
    )

    args = parser.parse_args()

    config = {
        "model_path": args.model_path,
        "labels_path": args.labeler_path,
        "sample_rate": args.sample_rate,
        "num_channels": args.num_channels,
        "audio_chuk": args.audio_chunk,
        "mel_frames": args.mel_frames,
        "num_mels": args.num_mels,
        "n_fft": args.n_fft,
        "fmax": args.fmax,
        "hop_length": args.hop_length,
        "confidence_threshold": args.confidence_threshold,
        "listening_hop_length": args.listening_hop_length,
        "device": "cpu",
    }

    service = SoundClassificationService.get_instance(config)
    service.listen_and_predict(duration=args.audio_chunk, overlap=args.listening_hop_length)


if __name__ == "__main__":
    main()
