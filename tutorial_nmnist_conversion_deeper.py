"""
Quick Start With N-MNIST

https://synsense.gitlab.io/sinabs-dynapcnn/getting_started/notebooks/nmnist_quick_start.html

The summarized explanation is organized in Notion.
"""
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

sample_data, label = NMNIST(save_to=root_dir, train=False)[0]

print(sample_data)
print(f"type of data is: {type(sample_data)}")
print(f"time length of sample data is: {sample_data['t'][-1] - sample_data['t'][0]} micro seconds")
print(f"there are {len(sample_data)} events in the sample data")
print(f"the label of the sample data is: {label}")

# Load both training and testing datasets
nmnist_train = NMNIST(save_to=root_dir, train=True)
nmnist_test = NMNIST(save_to=root_dir, train=False)

# Initialize counters
total_events_train = 0
total_events_test = 0

# Iterate through the training dataset and sum up the number of events
for data, label in nmnist_train:
    total_events_train += len(data)

# Repeat for the testing dataset
for data, label in nmnist_test:
    total_events_test += len(data)

# Now, 'total_events_train' and 'total_events_test' contain the total number of events
# in the training and testing datasets, respectively.
print(f"Total events in training dataset: {total_events_train}")
print(f"Total events in testing dataset: {total_events_test}")
#######################################################################################################
# CNN-To-SNN
#######################################################################################################
# Define CNN
# ann = nn.Sequential(
#             nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2),
#             nn.ReLU(),
#
#             nn.Flatten(),
#             nn.Linear(32 * 2 * 2, 10, bias=False),
# )

# tutorial document model
ann = nn.Sequential(
    nn.Conv2d(2, 8, 3, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),

    nn.Conv2d(8, 16, 3, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),

    nn.Conv2d(16, 32, 3, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),

    nn.Conv2d(32, 32, 3, 2, 1, bias=False),
    nn.ReLU(),

    nn.Flatten(),
    nn.Linear(32 * 2 * 2, 10, bias=False),
)

# ann = nn.Sequential(
#     nn.Conv2d(2, 16, 3, padding=1, bias=False),
#     nn.ReLU(),
#     nn.AvgPool2d(2, 2),
#
#     nn.Conv2d(16, 16, 3, padding=1, bias=False),
#     nn.ReLU(),
#
#     nn.Conv2d(16, 16, 3, padding=1, bias=False),
#     nn.ReLU(),
#     nn.AvgPool2d(2, 2),
#
#     nn.Conv2d(16, 32, 3, 2, 1, bias=False),
#     nn.ReLU(),
#
#     nn.Flatten(),
#     nn.Linear(32 * 4 * 4, 10, bias=False),
# )

# init the model weights
for layer in ann.modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(layer.weight.data)

# Define CNN Training & Testing Datasets
# define a transform that accumulate the events into a single frame image
to_frame = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=1)

cnn_train_dataset = NMNIST(save_to=root_dir, train=True, transform=to_frame)
cnn_test_dataset = NMNIST(save_to=root_dir, train=False, transform=to_frame)

# check the transformed data
sample_data, label = cnn_train_dataset[0]
print(f"The transformed array is in shape [Time-Step, Channel, Height, Width] --> {sample_data.shape}")

# Train & Test CNN
epochs = 5
lr = 1e-3
batch_size = 4
num_workers = 4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
shuffle = True
n_time_steps = 100

cnn = ann.to(device=device)

cnn_train_dataloader = DataLoader(cnn_train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=shuffle)
cnn_test_dataloader = DataLoader(cnn_test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=shuffle)

optimizer = SGD(params=cnn.parameters(), lr=lr)
criterion = CrossEntropyLoss()

for e in range(epochs):

    # train
    train_p_bar = tqdm(cnn_train_dataloader)
    for data, label in train_p_bar:
        # remove the time-step axis since we are training CNN
        # move the data to accelerator
        data = data.squeeze(dim=1).to(dtype=torch.float, device=device)
        label = label.to(dtype=torch.long, device=device)
        # forward
        optimizer.zero_grad()
        output = cnn(data)
        loss = criterion(output, label)
        # backward
        loss.backward()
        optimizer.step()
        # set progressing bar
        train_p_bar.set_description(f"Epoch {e} - Training Loss: {round(loss.item(), 4)}")

    # validate
    correct_predictions = []
    with torch.no_grad():
        test_p_bar = tqdm(cnn_test_dataloader)
        for data, label in test_p_bar:
            # remove the time-step axis since we are training CNN
            # move the data to accelerator
            data = data.squeeze(dim=1).to(dtype=torch.float, device=device)
            label = label.to(dtype=torch.long, device=device)
            # forward
            output = cnn(data)
            # calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            # compute the total correct predictions
            correct_predictions.append(pred.eq(label.view_as(pred)))
            # set progressing bar
            test_p_bar.set_description(f"Epoch {e} - Testing Model...")

        correct_predictions = torch.cat(correct_predictions)
        print(f"Epoch {e} - accuracy: {correct_predictions.sum().item() / (len(correct_predictions)) * 100}%")
#######################################################################################################
# Convert CNN-To-SNN
snn_convert = from_model(model=cnn, input_shape=(2, 34, 34), add_spiking_output=True, batch_size=batch_size).spiking_model
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

# Degraded Performance After Conversion

# Save trained models
base_save_path = "/home/parkjoe/PycharmProjects/sinabs-dynapcnn/saved_models"
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path_ann = os.path.join(base_save_path, f"tutorial_nmnist_conversion_ann_deeper{current_time}.pth")
model_save_path_snn = os.path.join(base_save_path, f"tutorial_nmnist_conversion_deeper{current_time}.pth")
torch.save(cnn, model_save_path_ann)
torch.save(snn_convert, model_save_path_snn)
print(f"Model saved to {model_save_path_ann}")
print(f"Model saved to {model_save_path_snn}")