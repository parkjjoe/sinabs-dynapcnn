"""
Quick Start With N-MNIST

https://synsense.gitlab.io/sinabs-dynapcnn/getting_started/notebooks/nmnist_quick_start.html

The summarized explanation is organized in Notion.
"""
#######################################################################################################
# Data Preparation
#######################################################################################################
try:
    from tonic.datasets.nmnist import NMNIST
except ImportError:
    #!pip install tonic
    from tonic.datasets.nmnist import NMNIST

# download dataset
root_dir = "datasets"
_ = NMNIST(save_to=root_dir, train=True)
_ = NMNIST(save_to=root_dir, train=False)

sample_data, label = NMNIST(save_to=root_dir, train=False)[0]

print(f"type of data is: {type(sample_data)}")
print(f"time length of sample data is: {sample_data['t'][-1] - sample_data['t'][0]} micro seconds")
print(f"there are {len(sample_data)} events in the sample data")
print(f"the label of the sample data is: {label}")
#######################################################################################################
# CNN-To-SNN
#######################################################################################################
# Define CNN
from torch import nn
# define a CNN model
cnn = nn.Sequential(
    # [2, 34, 34] -> [8, 17, 17]
    nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(3, 3), padding=(1, 1), bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    # [8, 17, 17] -> [16, 8, 8]
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    # [16 * 8 * 8] -> [16, 4, 4]
    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), bias=False),
    nn.ReLU(),
    # [16 * 4 * 4] -> [10]
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 10, bias=False),
    nn.ReLU(),
)

# init the model weights
for layer in cnn.modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(layer.weight.data)

# Define CNN Training & Testing Datasets
from tonic.transforms import ToFrame
from tonic.datasets import nmnist
# define a transform that accumulate the events into a single frame image
to_frame = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=1)

cnn_train_dataset = NMNIST(save_to=root_dir, train=True, transform=to_frame)
cnn_test_dataset = NMNIST(save_to=root_dir, train=False, transform=to_frame)

# check the transformed data
sample_data, label = cnn_train_dataset[0]
print(f"The transformed array is in shape [Time-Step, Channel, Height, Width] --> {sample_data.shape}")

# Train & Test CNN
#!pip install ipywidgets
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm.notebook import tqdm
from torch.nn import CrossEntropyLoss

epochs = 5
lr = 1e-3
batch_size = 4
num_workers = 4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
shuffle = True

cnn = cnn.to(device=device)

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

# Convert CNN-To-SNN
from sinabs.from_torch import from_model

snn_convert = from_model(model=cnn, input_shape=(2, 34, 34), batch_size=batch_size).spiking_model
print(snn_convert) # change ReLU to IAFSqueeze

# Test Converted SNN
# define a transform that accumulate the events into a raster-like tensor
n_time_steps = 100
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
        # forward
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

#######################################################################################################
# Train SNN with BPTT
#######################################################################################################
# Define SNN
import sinabs.layers as sl
from torch import nn
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential

# just replace the ReLU layer with the sl.IAFSqueeze
snn_bptt = nn.Sequential(
    # [2, 34, 34] -> [8, 17, 17]
    nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(3, 3), padding=(1, 1), bias=False),
    sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
    nn.AvgPool2d(2, 2),
    # [8, 17, 17] -> [16, 8, 8]
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), bias=False),
    sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
    nn.AvgPool2d(2, 2),
    # [16 * 8 * 8] -> [16, 4, 4]
    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), bias=False),
    sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
    # [16 * 4 * 4] -> [10]
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 10, bias=False),
    sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
)

# init the model weights
for layer in snn_bptt.modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(layer.weight.data)

# Why Disable All "Bias" Of The Convolutional Layer?

# Convert To Exodus Model If Exodus Available
try:
    from sinabs.exodus import conversion
    snn_bptt = conversion.sinabs_to_exodus(snn_bptt)
except ImportError:
    print("Sinabs-exodus is not installed.")

print(snn_bptt) # change IAFSqueeze to EXODUS IAFSqueeze

# Define SNN Training & Testing Datasets
n_time_steps = 100
to_raster = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=n_time_steps)

snn_train_dataset = NMNIST(save_to=root_dir, train=True, transform=to_raster)
snn_test_dataset = NMNIST(save_to=root_dir, train=False, transform=to_raster)

# Train & Test SNN With BPTT
epochs = 1
lr = 1e-3
batch_size = 4
num_workers = 4
device = "cuda:0"
shuffle = True

snn_train_dataloader = DataLoader(snn_train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True,
                                  shuffle=True)
snn_test_dataloader = DataLoader(snn_test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True,
                                 shuffle=False)

snn_bptt = snn_bptt.to(device=device)

optimizer = SGD(params=snn_bptt.parameters(), lr=lr)
criterion = CrossEntropyLoss()

for e in range(epochs):

    # train
    train_p_bar = tqdm(snn_train_dataloader)
    for data, label in train_p_bar:
        # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
        data = data.reshape(-1, 2, 34, 34).to(dtype=torch.float, device=device)
        label = label.to(dtype=torch.long, device=device)
        # forward
        optimizer.zero_grad()
        output = snn_bptt(data)
        # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
        output = output.reshape(batch_size, n_time_steps, -1)
        # accumulate all time-steps output for final prediction
        output = output.sum(dim=1)
        loss = criterion(output, label)
        # backward
        loss.backward()
        optimizer.step()

        # detach the neuron states and activations from current computation graph(necessary)
        for layer in snn_bptt.modules():
            if isinstance(layer, sl.StatefulLayer):
                for name, buffer in layer.named_buffers():
                    buffer.detach_()

        # set progressing bar
        train_p_bar.set_description(f"Epoch {e} - BPTT Training Loss: {round(loss.item(), 4)}")

    # validate
    correct_predictions = []
    with torch.no_grad():
        test_p_bar = tqdm(snn_test_dataloader)
        for data, label in test_p_bar:
            # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
            data = data.reshape(-1, 2, 34, 34).to(dtype=torch.float, device=device)
            label = label.to(dtype=torch.long, device=device)
            # forward
            output = snn_bptt(data)
            # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
            output = output.reshape(batch_size, n_time_steps, -1)
            # accumulate all time-steps output for final prediction
            output = output.sum(dim=1)
            # calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            # compute the total correct predictions
            correct_predictions.append(pred.eq(label.view_as(pred)))
            # set progressing bar
            test_p_bar.set_description(f"Epoch {e} - BPTT Testing Model...")

        correct_predictions = torch.cat(correct_predictions)
        print(f"Epoch {e} - BPTT accuracy: {correct_predictions.sum().item() / (len(correct_predictions)) * 100}%")

# Convert Back To Sinabs Model If Using Exodus Model For Training
try:
    from sinabs.exodus import conversion
    snn_bptt = conversion.exodus_to_sinabs(snn_bptt)
except ImportError:
    print("Sinabs-exodus is not installed.")

print(snn_bptt) # rechange EXODUS IAFSqueeze to IAFSqueeze
#######################################################################################################
#######################################################################################################
# Depoly SNN To The Devkit
#######################################################################################################
from sinabs.backend.dynapcnn import DynapcnnNetwork

# cpu_snn = snn_convert.to(device="cpu")
cpu_snn = snn_bptt.to(device="cpu")
dynapcnn = DynapcnnNetwork(snn=cpu_snn, input_shape=(2, 34, 34), discretize=True, dvs_input=False)
devkit_name = "speck2fdevkit"

# use the `to` method of DynapcnnNetwork to deploy the SNN to the devkit
dynapcnn.to(device=devkit_name, chip_layers_ordering=[0,0,7,4])
print(f"The SNN is deployed on the core: {dynapcnn.chip_layers_ordering}")

# Inference On The Devkit
import samna
from collections import Counter
from torch.utils.data import Subset

snn_test_dataset = NMNIST(save_to=root_dir, train=False)
# for time-saving, we only select a subset for on-chip inference， here we select 1/100 for an example run
subset_indices = list(range(0, len(snn_test_dataset), 100))
snn_test_dataset = Subset(snn_test_dataset, subset_indices)

inference_p_bar = tqdm(snn_test_dataset)

test_samples = 0
correct_samples = 0

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

    # use the most frequent output neuron index as the final prediction
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

print(f"On chip inference accuracy: {correct_samples / test_samples}")
#######################################################################################################
