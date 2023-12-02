import numpy as np
import sounddevice as sd
from keras.models import load_model
from joblib import load
from classify_utilities import AudioProcessor


class SoundClassificationService:
    
    _instance = None

    def __init__(self, config):
        """Initialize the service with the given configuration."""
        self.config = config
        
        self.microphone_index = sd.query_devices(kind='input')["index"]
        
        self.audio_processor = AudioProcessor(
            sample_rate=config["sample_rate"],
            n_mels=config["num_mels"],
            fmax=config["fmax"],
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            slice_length=config["slice_length"],
        )
        
        try:
            self.model = load_model(config["model_path"])
            self.labels_encoder = load(config["labels_path"])
        except Exception as e:
            print(f"Error loading files: {e}")
            raise
        
        
    @classmethod
    def get_instance(cls, config):
        """Singleton method to get the instance of the class."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    

    def listen_and_predict(self, duration=3, overlap=0.5):
        """Listen to live audio and make predictions."""
        buffer_length = int(self.config["sample_rate"] * duration)
        buffer = np.zeros(buffer_length)
        try:
            with sd.InputStream(samplerate=self.config["sample_rate"], device=self.microphone_index, channels=self.config["num_channels"]) as stream:
                print("Listening... Press Ctrl+C to stop.")
                while True:
                    try:
                        audio_chunk, _ = stream.read(int(self.config["sample_rate"] * overlap))
                    except Exception as e:
                        print(f"Error while recording: {e}")
                        break
                    
                    buffer = np.roll(buffer, -len(audio_chunk))
                    buffer[-len(audio_chunk):] = audio_chunk.flatten()
                    prediction_feature = self.audio_processor(buffer)
                    print(prediction_feature.shape)
                    reshaped_feature = prediction_feature.reshape(1, 130, self.config["num_mels"], self.config["num_channels"])
                    prediction = self.model.predict(reshaped_feature)
                    keyword = self.idx2label(prediction, self.labels_encoder)
                    if keyword:
                        print(f"Predicted Keyword: {keyword}, with: {prediction * 100}")
        except KeyboardInterrupt:
            print("Stopped listening.")


    def idx2label(self, idx, encoder):
        idx_reshaped = np.array(idx).reshape(1, -1)
        return encoder.inverse_transform(idx_reshaped)[0][0]