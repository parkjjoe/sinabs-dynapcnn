import sinabs
import sinabs.layers as sl
import torch
import torch.nn as nn
import numpy as np

iaf = sl.IAF(record_states=True, spike_threshold=5.)

n_steps = 400
input_ = (torch.rand((1, n_steps, 1)) < 0.05).float()
output = iaf(input_)

import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(15,5))

ax1.eventplot(torch.where(input_)[1])
ax1.set_ylabel("Input events")
ax2.plot(iaf.recordings['v_mem'].squeeze().numpy())
ax2.set_ylabel("IF Vmem")
ax3.eventplot(torch.where(output)[1])
ax3.set_ylabel("Output Events")
ax3.set_xlabel("Time")
plt.figure()
#######################################################################################################
from tonic import datasets, transforms

trainset = datasets.NMNIST('/home/parkjoe/PycharmProjects/sinabs-dynapcnn/datasets', train=True)
testset = datasets.NMNIST('/home/parkjoe/PycharmProjects/sinabs-dynapcnn/datasets', train=False)

transform = transforms.Compose([
    transforms.ToFrame(sensor_size=trainset.sensor_size, n_time_bins=30, include_incomplete=True), lambda x: x.astype(np.float32),
])

events, label = trainset[0]

frames = transform(events)
print(frames.shape)

plt.imshow(frames[:10, 0].sum(0))
#######################################################################################################
trainset = datasets.NMNIST('/home/parkjoe/PycharmProjects/sinabs-dynapcnn/datasets', train=True, transform=transform)
testset = datasets.NMNIST('/home/parkjoe/PycharmProjects/sinabs-dynapcnn/datasets', train=False, transform=transform)

batch_size = 16

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=4, drop_last=True)

frames = next(iter(trainloader))[0]
print(frames.shape)
#######################################################################################################
import sinabs.exodus.layers as sel

backend = sl # sinabs
backend = sel # sinabs exodus

model = nn.Sequential(
    sl.FlattenTime(),
    nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
    backend.IAFSqueeze(batch_size=batch_size, min_v_mem=-1),
    sl.SumPool2d(2),
    nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
    backend.IAFSqueeze(batch_size=batch_size, min_v_mem=-1),
    sl.SumPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
    backend.IAFSqueeze(batch_size=batch_size, min_v_mem=-1),
    sl.SumPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
    backend.IAFSqueeze(batch_size=batch_size, min_v_mem=-1),
    sl.SumPool2d(2),
    nn.Conv2d(64, 10, kernel_size=2, padding=0, bias=False),
    backend.IAFSqueeze(batch_size=batch_size, min_v_mem=-1),
    nn.Flatten(),
    sl.UnflattenTime(batch_size=batch_size),
).cuda()

print(model(frames.cuda()).shape)

from tqdm.notebook import tqdm
n_epochs = 1
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.functional.cross_entropy

for epoch in range(n_epochs):
    losses = []
    for data, targets in tqdm(trainloader):
        data, targets = data.cuda(), targets.cuda()
        sinabs.reset_states(model)
        optimizer.zero_grad()
        y_hat = model(data)
        pred = y_hat.sum(1)
        loss = crit(pred, targets, )
        loss.backward()
        losses.append(loss)
        optimizer.step()
    print(f"Loss: {torch.stack(losses).mean()}")

import torchmetrics

acc = torchmetrics.Accuracy('multiclass', num_classes=10).cuda()
model.eval()

for data, targets in tqdm(testloader):
    data, targets = data.cuda(), targets.cuda()
    sinabs.reset_states(model)
    with torch.no_grad():
        y_hat = model(data)
        pred = y_hat.sum(1)
        acc(pred, targets)

print(f"Test accuracy: {100*acc.compute():.2f}%")
#######################################################################################################
from sinabs.exodus.conversion import exodus_to_sinabs

sinabs_model = exodus_to_sinabs(model)

torch.save(sinabs_model, "nmnist_model.pth")
plt.show()