import torch
import os
import datetime
import samna
import time
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm.notebook import tqdm
from torch.nn import CrossEntropyLoss
from tonic.datasets import CIFAR10DVS
from torch import nn
from tonic.transforms import ToFrame
from torch.utils.data import random_split
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork
from collections import Counter
from torch.utils.data import Subset
#######################################################################################################
# Data Preparation
#######################################################################################################
# download dataset
root_dir = "/home/parkjoe/PycharmProjects/sinabs-dynapcnn/datasets"
cifar10dvs = CIFAR10DVS(save_to=root_dir)

train_ratio = 0.8
num_train = int(len(cifar10dvs) * train_ratio)
num_test = len(cifar10dvs) - num_train

cifar10dvs_train_dataset, cifar10dvs_test_dataset = random_split(cifar10dvs, [num_train, num_test])

sample_data, label = cifar10dvs_test_dataset[0]

print(sample_data)
print(f"type of data is: {type(sample_data)}")
print(f"time length of sample data is: {sample_data['t'][-1] - sample_data['t'][0]} micro seconds")
print(f"there are {len(sample_data)} events in the sample data")
print(f"the label of the sample data is: {label}")
#######################################################################################################
# CNN-To-SNN
#######################################################################################################
# Define CNN
ann = nn.Sequential(
    nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1, stride=2, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Flatten(),
    nn.Linear(64 * 2 * 2, 100, bias=False),
    nn.ReLU(),
    nn.Linear(100, 10, bias=False),
)

# init the model weights
for layer in ann.modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(layer.weight.data)

# Define CNN Training & Testing Datasets
# define a transform that accumulate the events into a single frame image
def transform_dataset(dataset, transform):
    transformed_dataset = []
    for events, label in dataset:
        transformed_data = transform(events)
        transformed_dataset.append((transformed_data, label))
    return transformed_dataset

to_frame = ToFrame(sensor_size=cifar10dvs.sensor_size, n_time_bins=1)
cnn_train_dataset = transform_dataset(cifar10dvs_train_dataset, to_frame)
cnn_test_dataset = transform_dataset(cifar10dvs_test_dataset, to_frame)

# check the transformed data
# sample_data, label = cnn_train_dataset[0]
# print(f"The transformed array is in shape [Time-Step, Channel, Height, Width] --> {sample_data.shape}")

# Train & Test CNN
epochs = 5
lr = 1e-3
batch_size = 4
num_workers = 4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
shuffle = True

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
snn_convert = from_model(model=cnn, input_shape=(2, 128, 128), add_spiking_output=True, batch_size=batch_size).spiking_model
print(snn_convert) # change ReLU to IAFSqueeze

# Test Converted SNN
# define a transform that accumulate the events into a raster-like tensor
n_time_steps = 100
to_raster = ToFrame(sensor_size=cifar10dvs.sensor_size, n_time_bins=n_time_steps)
snn_test_dataset = transform_dataset(cifar10dvs_test_dataset, to_raster)
snn_test_dataloader = DataLoader(snn_test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)

snn_convert = snn_convert.to(device)

correct_predictions = []
with torch.no_grad():
    test_p_bar = tqdm(snn_test_dataloader)
    for data, label in test_p_bar:
        # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
        data = data.reshape(-1, 2, 128, 128).to(dtype=torch.float, device=device)
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
model_save_path_ann = os.path.join(base_save_path, f"cifar10dvs_conversion_ann_{current_time}.pth")
model_save_path_snn = os.path.join(base_save_path, f"cifar10dvs_conversion_{current_time}.pth")
torch.save(cnn, model_save_path_ann)
torch.save(snn_convert, model_save_path_snn)
print(f"Model saved to {model_save_path_ann}")
print(f"Model saved to {model_save_path_snn}")
#######################################################################################################
#######################################################################################################
# Depoly SNN To The Devkit
#######################################################################################################
# cpu_snn = snn_convert.to(device="cpu")
cpu_snn = snn_convert.to(device="cpu")
dynapcnn = DynapcnnNetwork(snn=cpu_snn, input_shape=(2, 128, 128), discretize=True, dvs_input=False)
devkit_name = "speck2fdevkit"

# use the `to` method of DynapcnnNetwork to deploy the SNN to the devkit
dynapcnn.to(device=devkit_name, chip_layers_ordering="auto")
print(f"The SNN is deployed on the core: {dynapcnn.chip_layers_ordering}")

# Inference On The Devkit
# for time-saving, we only select a subset for on-chip infernce， here we select 1/100 for an example run
subset_indices = list(range(0, len(cifar10dvs_test_dataset), 100))
snn_test_dataset = Subset(cifar10dvs_test_dataset, subset_indices)

inference_p_bar = tqdm(snn_test_dataset)

test_samples = 0
correct_samples = 0
total_output_spikes = 0

# Start to record inference time
start_time = time.time()

for events, label in inference_p_bar:

    # create samna Spike events stream
    samna_event_stream = []
    for ev in events:
        spk = samna.speck2f.event.Spike()
        spk.x = ev['x']
        spk.y = ev['y']
        spk.timestamp = ev['t'] - events['t'][0]
        spk.feature = ev['p']
        # Spikes will be sent to layer/core #0, since the SNN is deployed on core: [0, 1, 2, 3]
        spk.layer = 0
        samna_event_stream.append(spk)

    # inference on chip
    # output_events is also a list of Spike, but each Spike.layer is 3, since layer#3 is the output layer
    output_events = dynapcnn(samna_event_stream)
    total_output_spikes += len(output_events)

    # use the most frequent output neruon index as the final prediction
    neuron_index = [each.feature for each in output_events]
    if len(neuron_index) != 0:
        frequent_counter = Counter(neuron_index)
        prediction = frequent_counter.most_common(1)[0][0]
    else:
        prediction = -1
    inference_p_bar.set_description(f"label: {label}, prediction: {prediction}， output spikes num: {len(output_events)}")

    if prediction == label:
        correct_samples += 1

    test_samples += 1

print(f"Total output spikes: {total_output_spikes}")
print(f"On chip inference accuracy: {correct_samples / test_samples}")

# Stop to record inference time
end_time = time.time()
# Calculate total inference time
total_inference_time = end_time - start_time
print(f"Total inference time on hareware: {total_inference_time} seconds")
