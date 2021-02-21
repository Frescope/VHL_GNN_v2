import torch
from PIL import Image
from torchvision import models, transforms
import urllib
import numpy as np

# model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
model.eval()
# Download an example image from the pytorch website

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
# sample execution (requires torchvision)

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

fmap_block = []
input_block = []

def forward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input)
model.fc.register_forward_hook(forward_hook)

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

features = input_block[0][0].squeeze()
features = np.array(features.cpu())
print(features)
print(features.shape,np.mean(features))

# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)

# import torch
# import torch.nn as nn
# from torch import nn, Tensor
# from torch.autograd import Variable
# from torchvision import models, transforms
# from typing import Callable, Any, Optional, Tuple, List
# from PIL import Image
# import numpy as np
# import os, glob
# import urllib
#
# class net(nn.Module):
#     def __init__(self):
#         super(net, self).__init__()
#         self.net = models.inception_v3(pretrained=True)
#
#     def forward(self, x):
#         # N x 3 x 299 x 299
#         x = self.net.Conv2d_1a_3x3(x)
#         # N x 32 x 149 x 149
#         x = self.net.Conv2d_2a_3x3(x)
#         # N x 32 x 147 x 147
#         x = self.net.Conv2d_2b_3x3(x)
#         # N x 64 x 147 x 147
#         x = self.net.maxpool1(x)
#         # N x 64 x 73 x 73
#         x = self.net.Conv2d_3b_1x1(x)
#         # N x 80 x 73 x 73
#         x = self.net.Conv2d_4a_3x3(x)
#         # N x 192 x 71 x 71
#         x = self.net.maxpool2(x)
#         # N x 192 x 35 x 35
#         x = self.net.Mixed_5b(x)
#         # N x 256 x 35 x 35
#         x = self.net.Mixed_5c(x)
#         # N x 288 x 35 x 35
#         x = self.net.Mixed_5d(x)
#         # N x 288 x 35 x 35
#         x = self.net.Mixed_6a(x)
#         # N x 768 x 17 x 17
#         x = self.net.Mixed_6b(x)
#         # N x 768 x 17 x 17
#         x = self.net.Mixed_6c(x)
#         # N x 768 x 17 x 17
#         x = self.net.Mixed_6d(x)
#         # N x 768 x 17 x 17
#         x = self.net.Mixed_6e(x)
#         # N x 768 x 17 x 17
#         aux_defined = self.net.training and self.net.aux_logits
#         if aux_defined:
#             aux = self.net.AuxLogits(x)
#         else:
#             aux = None
#         # N x 768 x 17 x 17
#         x = self.net.Mixed_7a(x)
#         # N x 1280 x 8 x 8
#         x = self.net.Mixed_7b(x)
#         # N x 2048 x 8 x 8
#         x = self.net.Mixed_7c(x)
#         # N x 2048 x 8 x 8
#         # Adaptive average pooling
#         x = self.net.avgpool(x)
#         # N x 2048 x 1 x 1
#         x = self.net.dropout(x)
#         # N x 2048 x 1 x 1
#         x = torch.flatten(x, 1)
#         # N x 2048
#         x = self.net.fc(x)
#         # N x 1000 (num_classes)
#         return x, aux
#
# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)
# # sample execution (requires torchvision)
#
# input_image = Image.open(filename)
# preprocess = transforms.Compose([
#     transforms.Resize(299),
#     transforms.CenterCrop(299),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
# input_batch = input_batch.repeat(4,1,1,1)
# # move the input and model to GPU for speed if available
# input_batch = input_batch.to('cuda')
#
# model2 = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
# model2.to('cuda')
# model2.eval()
# with torch.no_grad():
#     output2 = model2(input_batch)
#
# model = net()
# model.to('cuda')
# model.eval()
# with torch.no_grad():
#     output = model(input_batch)
#
# print(output2[0] - output[0])
def hello():
    print("hello!")