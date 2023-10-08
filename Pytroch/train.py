# %%
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio

from dataset import UrbanSoundDataset
from cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "data/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "data/UrbanSound8K/audio"
SAMPLE_RATE = 22050 
NUM_SAMPLES = 22050

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    # cerate a loop that will iterate over the data loader
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # calcualte loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        
        # backpropagate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Loss: {loss.item()}")
               
def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("---------------------------")
    print("Finished training")

# %%
# Train script sequence:

if __name__ == "__main__":
    
    # Check if GPU is available, else use cpu
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")
    
    # Instantiate our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
    )
    
    # Create instance of our UrbanSoundDataset class
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram, 
                            SAMPLE_RATE, 
                            NUM_SAMPLES,
                            device)
    
    # Pass the datasetand batch size
    train_dataloader = create_data_loader(usd, BATCH_SIZE)
    
    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)
    
    #Instantuiate optimizer and loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    
    train(cnn, train_dataloader, loss_fn, optimizer, device, EPOCHS)

    torch.save(cnn.state_dict(), "cnn.pth")
    print("Model trained and saved")


