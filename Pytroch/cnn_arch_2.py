# %% [markdown]
# ### Implementing CNN for Sound Classification

# %% [markdown]
# The CNN layers parameters consist of a set of learnable filters (or kernels), 
# which have a small receptive field, but extend through the full depth of the 
# input volume. Their role is to slide (or convolve) across the input data 
# (such as an image) to produce a feature map or activation map.
# 
# Convolution: The filter slides over the input image. At every position, a matrix multiplication is performed between the filter and the part of the image it's currently on. This results in a single pixel in the output feature map. This process is repeated across the entire image.
# 
# Activation: After convolution, an activation function (like ReLU) is applied to introduce non-linearity to the model. This allows the model to learn more complex patterns.
# 
# Pooling: This step reduces the spatial dimensions of the feature map, retaining the most important information.
# 
# Fully Connected Layers: After several rounds of convolution and pooling, the data is flattened and passed through one or more fully connected layers to determine the final class or classes of the input image.

# %%
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

# %%
# our model calss will inherit from nn.Module
class CNNNetwork(nn.Module):
    # constructor
    def __init__(self, n_input=1, n_output=35, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv2d(n_input, n_channel, kernel_size=(3,3), stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(n_channel, n_channel, kernel_size=3, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(n_channel)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=3, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(2 * n_channel)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(2 * n_channel, 2 * n_channel, kernel_size=3, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(2 * n_channel)
        self.pool4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(2 * n_channel * 5 * 4 , 10)
        
            
    # forward method
    def forward(self, x): 
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = nn.Flatten(x)
        x = self.fc1(x)
        return F.log_softmax(x)


# %%
if __name__ == "__main__":
    # create a model
    cnn = CNNNetwork(n_input=1, n_output=10)
    # print summary of the model. use.cuda() to move the model to GPU
    
    summary(cnn.cuda(), input_size=(1, 64, 44))
    