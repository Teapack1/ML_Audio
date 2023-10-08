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

# %%
# our model calss will inherit from nn.Module
class CNNNetwork(nn.Module):
    # constructor
    def __init__(self):
        # The super() function returns a temporary object of the superclass (nn.Module in this case), allowing you to call its methods. 
        # The __init__() method of the superclass is explicitly called within the child class.
        super().__init__()
        # We will use 4 CNN blocks. Each cnn block will go through -> flattern / linear/ softmax (10 classes in our case)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,      # 1 channel for grayscale image
                out_channels=16,    # 16 filters (kernels), This means the layer will produce 16 feature maps as output.
                kernel_size=3,      # (3x3) A filter (or kernel) is a smaller-sized matrix in terms of width and height. It's used to slide over the input data (like an image) to produce a feature map. 
                stride=1,           # controls how the filter convolves around the input volume. A stride of 1 moves the filter one pixel at a time
                padding=2           # Padding is used to add layers of zeros to the outside of the input volume. 
            ),
            #Rectified linear unit
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, # Equal to the out_channels of the previous layer.     
                out_channels=32,    
                kernel_size=3,     
                stride=1,         
                padding=2
            ),
            #Rectified linear unit
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
            
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,      
                out_channels=64,    
                kernel_size=3,     
                stride=1,         
                padding=2
            ),
            #Rectified linear unit
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )   
            
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,      
                out_channels=128,    
                kernel_size=3,     
                stride=1,         
                padding=2
            ),
            #Rectified linear unit
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        # 128 -> num output channels from last conv block
        # 5*4 -> spatial dimansion (h x w) arriving from the last layer (Maxpool2d)
        # 5*4 can be found using torchsummary.summary(cnn.cuda(), input_size=(1, 64, 44))
        self.linear = nn.Linear(in_features=128*5*4, out_features=10)
        self.softmax = nn.Softmax(dim=1)     
            
    # forward method
    def forward(self, input_data): 
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions
    

# %%
if __name__ == "__main__":
    # create a model
    cnn = CNNNetwork()
    # print summary of the model. use.cuda() to move the model to GPU
    summary(cnn.cuda(), input_size=(1, 64, 44))
# %%



