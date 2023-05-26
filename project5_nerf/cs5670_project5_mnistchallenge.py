# -*- coding: utf-8 -*-
"""CS5670_Project5_MNISTChallenge.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QJUs15_E5iCna3CGk8nJb9rZUj1krwSG

# CS5670 Project 5 - MNIST Challenge Notebook

Welcome to the Project 5 4-credit MNIST Challenge.

In this notebook you will create and train your own neural network model in PyTorch for classifying handwritten digits on the MNIST dataset.
We will pretend that we are targeting low-power and low-memory devices, and so you will have to limit the number of model parameters that you use.

## Installation of Dependencies

First you want to **make sure that you're using the GPU backend**.

* Go to Runtime -> Change runtime type and choose "GPU" under Hardware Accelerator.
* Only then run the next cell. It should print "GPU Support: True"
"""

# http://pytorch.org/
from os import path
!pip install pillow==7.1.2
!python --version
import torch
print(torch. __version__)
import torchvision
print(torchvision. __version__)
print(f"GPU Support: {torch.cuda.is_available()}")

"""## Useful functions

Run the below cell to define functions that will be used later
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

import urllib
import cv2
import numpy as np
import os, sys,  math, random, subprocess
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from IPython.display import clear_output, Image, display, HTML
from google.protobuf import text_format
from io import StringIO
import PIL.Image

def get_n_params(module):
  nparam = 0
  for name, param in module.named_parameters():
    param_count = 1
    for size in list(param.size()):
      param_count *= size
    nparam += param_count
  return nparam

def get_model_params(model):
  nparam = 0
  for name, module in model.named_modules():
    nparam += get_n_params(module)
  return nparam

def to_numpy_image(tensor_or_variable):
  
  # If this is already a numpy image, just return it
  if type(tensor_or_variable) == np.ndarray:
    return tensor_or_variable
  
  # Make sure this is a tensor and not a variable
  if type(tensor_or_variable) == Variable:
    tensor = tensor_or_variable.data
  else:
    tensor = tensor_or_variable
  
  # Convert to numpy and move to CPU if necessary
  np_img = tensor.cpu().numpy()
  
  # If there is no batch dimension, add one
  if len(np_img.shape) == 3:
    np_img = np_img[np.newaxis, ...]
  
  # Convert from BxCxHxW (PyTorch convention) to BxHxWxC (OpenCV/numpy convention)
  np_img = np_img.transpose(0, 2, 3, 1)
  
  return np_img

def normalize_zero_one_range(tensor_like):
  x = tensor_like - tensor_like.min()
  x = x / (x.max() + 1e-9)
  return x

def prep_for_showing(image):
  np_img = to_numpy_image(image)
  if len(np_img.shape) > 3:
    np_img = np_img[0]
  np_img = normalize_zero_one_range(np_img)
  return np_img

def show_image(tensor_var_or_np, title=None, bordercolor=None):
  np_img = prep_for_showing(tensor_var_or_np)
  
  if bordercolor is not None:
    np_img = draw_border(np_img, bordercolor)
  
  # plot it
  np_img = np_img.squeeze()
  plt.figure(figsize=(4,4))
  plt.imshow(np_img)
  plt.axis('off')
  if title: plt.title(title)
  plt.show()

"""## Training Data

We will use the [MNIST handrwritten digit dataset](http://yann.lecun.com/exdb/mnist/) to train our neural network models. There is a simple wrapper for the MNIST dataset in the torchvision package that implements the Dataset class. We will use that in conjunction with the DataLoader to load training data. Run the below cell to download and initialize our training and test datasets. You should see an example batch of images and their labels shown.
"""

import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 32

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
show_image(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(BATCH_SIZE)))

"""## Model Definition (Example Model)

This is an example model provided to you. It is pretty good and it reaches >98% accuracy on the test set, but we can do better. In particular, we will try to break the 99% accuracy barrier with a tight limit on the number of model parameters.
"""

class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        # Convolution. Input channels: 1, output channels: 6, kernel size: 5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # Max-pooling layer that will halve the HxW resolution
        self.pool = nn.MaxPool2d(2, 2)
        # Another 5x5 convolution that brings channel count up to 16
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Three fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 60)
        self.fc2 = nn.Linear(60, 40)
        self.fc3 = nn.Linear(40, 10)

    def forward(self, x):
        # Apply convolution, activation and pooling
        # Output width after convolution = (input_width - (kernel_size - 1) / 2)
        # Output width after pooling = input_width / 2
        
        # x.size() = Bx1x28x28
        x = self.pool(F.relu(self.conv1(x)))
        # x.size() = Bx6x12x12
        x = self.pool(F.relu(self.conv2(x)))
        # x.size() = Bx16x4x4
        
        # Flatten the output
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = ExampleModel()

nparams = get_n_params(net)
print(f"Model number of parameters: {nparams}")

"""## Model Definition (Your Model)

Define your own model here. It should satisfy the following constraints:
<font color="red">
*** Number of total model parameters: < 50000**
<br/>
*** Reaches test-set accuracy greater than or equal to 99% after training for 10 epochs**
</font>
  
A couple of ideas you can try:
* Different model structure (e.g. more layers, smaller/bigger kernels)
* Residual connections
* Batch [2] / Layer Normalization [3]
* Densely connected architectures [1]

<font size="1em">[1] Huang, G., Liu, Z., Weinberger, K. Q., & van der Maaten, L. (2017, July). Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (Vol. 1, No. 2, p. 3).</font>
</br>
<font size="1em">[2] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167.</font>
</br>
<font size="1em">[3] Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. arXiv preprint arXiv:1607.06450.
</font>
"""

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        # Define your modules here

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(16)
        self.drop = nn.Dropout(0.1)
        self.fc1 = nn.Linear(16*5*5, 100)
        self.fc2 = nn.Linear(100, 10)
        

    def forward(self, x):
        # Define your dynamic computational graph here

        # Output width after convolution = (input_width - (kernel_size - 1) / 2)
        # Output width after pooling = input_width / 2
        
        # x = 1x28x28
        x = self.pool(self.norm1(F.relu(self.conv1(x))))
        # x = 32x13x13
        x = self.pool(self.norm2(F.relu(self.conv2(x))))
        # x = 16x5x5
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(self.drop(x)))
        x = self.fc2(self.drop(x))

        return x


net = StudentModel()

nparams = get_n_params(net)
print(f"Model number of parameters: {nparams}")

assert nparams < 50000, "Too many model parameters!"

"""## Training Loop

Run the following training loop to train your model. **Uncomment the line `net = StudentModel()` to use your model instead of the example model**.
"""

import torch.optim as optim
PRINT_EVERY = 900

#-----------------------------------------------------------------------------
# Comment this line:
# net = ExampleModel()

# Uncomment this line
net = StudentModel()
#-----------------------------------------------------------------------------

if type(net) == ExampleModel:
  print("WARNING! Running example model. Uncomment line to run your own model!")

#-----------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
  net = net.cuda()
  print("Using GPU")
else:
  print("Not using GPU")
  
net.train()
  
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        
        inputs = Variable(inputs)
        labels = Variable(labels)
        
        if USE_CUDA:
          inputs = inputs.cuda()
          labels = labels.cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data
        if i % PRINT_EVERY == PRINT_EVERY - 1:    # print every PRINT_EVERY mini-batches
            #show_image(torchvision.utils.make_grid(inputs.data))
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

"""## Testing

Use the below cell to test your model on the MNIST test set. By right you should only run your model on the test set once, before publishing your results. In this assignment, we will allow you to re-use the test set, which is technically an incorrect practice.
"""

correct = 0
total = 0

net.eval()

for data in testloader:
    images, labels = data
    images = Variable(images, volatile=True)
    if USE_CUDA:
      images, labels = images.cuda(), labels.cuda()
    
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

acc = 100 * correct / total
    
print(f'Accuracy of the network on the 10000 test images: {acc}%')
print(f'Correct: {correct}/{total}')
print("")

from IPython.display import Markdown, display

if acc >= 99:
  print("Congratulations! You beat the 99% barrier!")
else:
  print("Sorry, but you can do better. Try again!")

"""### Per-class accuracy

Run the below cell to see which digits your model is better at recognizing and which digits it gets confused by.
"""

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

for data in testloader:
    images, labels = data
    images = Variable(images, volatile=True)
    if USE_CUDA:
      images, labels = images.cuda(), labels.cuda()
    
    outputs = net(images)
    
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))