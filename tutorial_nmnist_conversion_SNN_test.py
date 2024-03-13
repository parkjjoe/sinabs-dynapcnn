import datetime
import os
import time
from collections import Counter

import samna
#######################################################################################################
import torch
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.from_torch import from_model
from tonic.datasets.nmnist import NMNIST
from tonic.transforms import ToFrame
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm.notebook import tqdm

#######################################################################################################
# Data Preparation
#######################################################################################################
# download dataset
root_dir = "/home/parkjoe/PycharmProjects/sinabs-dynapcnn/datasets"
_ = NMNIST(save_to=root_dir, train=True)
_ = NMNIST(save_to=root_dir, train=False)

# Load both training and testing datasets
nmnist_train = NMNIST(save_to=root_dir, train=True)
nmnist_test = NMNIST(save_to=root_dir, train=False)
#######################################################################################################
# Train & Test CNN
epochs = 5
lr = 1e-3
batch_size = 4
num_workers = 4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
shuffle = True
n_time_steps = 100
#######################################################################################################
snn_model_path = "/home/parkjoe/PycharmProjects/sinabs-dynapcnn/saved_models/tutorial_nmnist_conversion_deeper20240308_152441.pth"
snn_convert = torch.load(snn_model_path)
print(snn_convert) # change ReLU to IAFSqueeze

# Test Converted SNN
# define a transform that accumulate the events into a raster-like tensor
to_raster = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=n_time_steps)
snn_test_dataset = NMNIST(save_to=root_dir, train=False, transform=to_raster)
snn_test_dataloader = DataLoader(snn_test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)

snn_convert = snn_convert.to(device)

correct_predictions = []
with torch.no_grad():
    test_p_bar = tqdm(snn_test_dataloader)
    for data, label in test_p_bar:
        # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
        data = data.reshape(-1, 2, 34, 34).to(dtype=torch.float, device=device)
        label = label.to(dtype=torch.long, device=device)
        # forward;
        output = snn_convert(data)
        # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
        output = output.reshape(batch_size, n_time_steps, -1)
        # accumulate all time-steps output for final prediction
        output = output.sum(dim=1)
        # calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        # compute the total correct predictions
        correct_predictions.append(pred.eq(label.view_as(pred)))
        # set progressing bar
        test_p_bar.set_description(f"Testing SNN Model...")

    correct_predictions = torch.cat(correct_predictions)
    print(f"accuracy of converted SNN: {correct_predictions.sum().item() / (len(correct_predictions)) * 100}%")