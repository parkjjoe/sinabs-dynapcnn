import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.seq = nn.Sequential(
            # 1st Conv + ReLU + Pooling
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 2nd Conv + ReLU + Pooling
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Dense layers
            nn.Flatten(),
            nn.Linear(4 * 4* 50, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        return self.seq(x)
