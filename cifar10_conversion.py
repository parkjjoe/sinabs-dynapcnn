import os
import torch
import samna
import time
import datetime
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sinabs.from_torch import from_model
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from tqdm.notebook import tqdm
from sinabs.backend.dynapcnn import DynapcnnNetwork
from collections import Counter
from torch.utils.data import Subset
#######################################################################################################
# Data Preparation
#######################################################################################################
#transform = transforms.Compose([transforms.ToTensor()])
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

root_dir = "/home/parkjoe/PycharmProjects/sinabs-dynapcnn/datasets"
cifar10_train_dataset = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)
cifar10_test_dataset = datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
#######################################################################################################
# CNN-To-SNN
#######################################################################################################
# Define CNN

ann = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, bias=False),
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

    nn.Flatten(),
    nn.Linear(64 * 2 * 2, 128, bias=False),
    nn.ReLU(),
    nn.Linear(128, 10, bias=False),
)

# init the model weights
for layer in ann.modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(layer.weight.data)

# Define CNN Training & Testing Datasets
# Train & Test CNN
epochs = 5
lr = 1e-3
batch_size = 4
num_workers = 4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

cnn = ann.to(device=device)

cnn_train_dataloader = DataLoader(cifar10_train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)
cnn_test_dataloader = DataLoader(cifar10_test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)

optimizer = SGD(params=cnn.parameters(), lr=lr)
criterion = CrossEntropyLoss()

for e in range(epochs):

    # train
    train_p_bar = tqdm(cnn_train_dataloader)
    for data, label in train_p_bar:
        # remove the time-step axis since we are training CNN
        # move the data to accelerator
        data = data.to(dtype=torch.float, device=device)
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
            data = data.to(dtype=torch.float, device=device)
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
snn_convert = from_model(model=cnn, input_shape=(3, 32, 32), add_spiking_output=True, synops=True, batch_size=batch_size).spiking_model
print(snn_convert) # change ReLU to IAFSqueeze

# Test Converted SNN
# define a transform that accumulate the events into a raster-like tensor
n_time_steps = 100

snn_convert = snn_convert.to(device)

correct_predictions = []
with torch.no_grad():
    test_p_bar = tqdm(cnn_test_dataloader)
    for data, label in test_p_bar:
        # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
        data = data.to(dtype=torch.float, device=device)
        label = label.to(dtype=torch.long, device=device)
        # forward;
        output = snn_convert(data)
        # # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
        # output = output.reshape(batch_size, n_time_steps, -1)
        # # accumulate all time-steps output for final prediction
        # output = output.sum(dim=1)
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
model_save_path_ann = os.path.join(base_save_path, f"cifar10_conversion_ann_{current_time}.pth")
model_save_path_snn = os.path.join(base_save_path, f"cifar10_conversion_{current_time}.pth")
torch.save(cnn, model_save_path_ann)
torch.save(snn_convert, model_save_path_snn)
print(f"Model saved to {model_save_path_ann}")
print(f"Model saved to {model_save_path_snn}")
