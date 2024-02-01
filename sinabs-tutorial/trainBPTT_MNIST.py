"""
Traning by backpropagation through time (BPTT)

https://sinabs.readthedocs.io/en/v1.2.10/tutorials/bptt.html

Train a spiking network directly (without training an analog network first), on the Sequential MNIST task.

"""
#######################################################################################################
from torchvision import datasets
import torch

torch.manual_seed(0)

class MNIST(datasets.MNIST):
    def __init__(self, root, train=True, single_channel=False):
        datasets.MNIST.__init__(self, root, train=train, download=True)
        self.single_channel = single_channel

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = img.float() / 255.0

        # default is by row, output is [time, channels] = [28, 28]
        # OR if we want by single item, output is [784, 1]
        if self.single_channel:
            img = img.reshape(-1).unsqueeze(1)

        spikes = torch.rand(size=img.shape) < img
        spikes = spikes.float()

        return spikes, target

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64

dataset_test = MNIST(root="./data/", train=False)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=BATCH_SIZE, drop_last=True
)

dataset = MNIST(root="./data/", train=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True)
#######################################################################################################
# Training a baseline
#######################################################################################################
from torch import nn

ann = nn.Sequential(
    nn.Conv2d(1, 20, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    nn.Conv2d(20, 32, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    nn.Conv2d(32, 128, 3, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(128, 500, bias=False),
    nn.ReLU(),
    nn.Linear(500, 10, bias=False),
)

# Training
from tqdm.notebook import tqdm

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(ann.parameters())

for epoch in range(100):
    pbar = tqdm(dataloader)
    for img, target in pbar:
        optimizer.zero_grad()

        target = target.unsqueeze(1).repeat([1, 28])
        img = img.reshape([-1, 28])
        target = target.reshape([-1])

        out = ann(img)
        #           out = out.sum(1)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())

# Testing
accs = []

pbar = tqdm(dataloader_test)
for img, target in pbar:

    img = img.reshape([-1, 28])
    out = ann(img)
    out = out.reshape([64, 28, 10])
    out = out.sum(1)

    predicted = torch.max(out, axis=1)[1]
    acc = (predicted == target).sum().numpy() / BATCH_SIZE
    accs.append(acc)

print(sum(accs) / len(accs))
#######################################################################################################
# Defining a spiking network
#######################################################################################################
# define a 4-layer fully connected spiking neural network
from sinabs.from_torch import from_model

model = from_model(ann, batch_size=BATCH_SIZE).to(device)
model = model.train()
#######################################################################################################
# Training
#######################################################################################################
# PyTorch convolutional layers don't support inputs that aren't 4-dimensional (batch, channels, height, width).
# As a workaround, when using sinabs, you'll have to squeeze the time and batch dimensions.
# Starting from data in the form (batch, time, channels, ...), the data should be squeezed to (batch*time, channels, ...).
for img in dataloader:
    print(img[0].shape)
    break
# In the training, we included a reshape(-1, 28) (to squeeze the dimensions in input) and a reshape((BATCH_SIZE, 28, 10)) on the output to restore the original form.
from tqdm.notebook import tqdm

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):
    pbar = tqdm(dataloader)
    for img, target in pbar:
        optimizer.zero_grad()
        model.reset_states()

        img = img.reshape((-1, 28)) # merging time and batch dimensions
        out = model.spiking_model(img.to(device))
        out = out.reshape((BATCH_SIZE, 28, 10)) # restoring original dimensions

        # the output of the network is summed over the 28 time steps (rows)

    out = out.sum(1)
    loss = criterion(out, target.to(device))
    loss.backward()
    optimizer.step()

    pbar.set_postfix(loss=loss.item())
#######################################################################################################
# Testing
#######################################################################################################
accs = []

pbar = tqdm(dataloader_test)
for img, target in pbar:
    model.reset_states()

    img = img.reshape((-1, 28)) # merging time and batch dimensions
    out = model.spiking_model(img.to(device))
    out = out.reshape((BATCH_SIZE, 28, 10)) # restoring original dimensions

    out = out.sum(1)
    predicted = torch.max(out, axis=1)[1]
    acc= (predicted == target.to(device)).sum().cpu().numpy() / BATCH_SIZE
    accs.append(acc)

print(sum(accs) / len(accs))
