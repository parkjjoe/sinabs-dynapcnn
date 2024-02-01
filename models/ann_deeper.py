import torch.nn as nn
import sinabs.layers as sl
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.ann = nn.Sequential(
            # [2, 34, 34] -> [8, 17, 17]
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            # [8, 17, 17] -> [16, 9, 9]
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            # [16, 9, 9] -> [32, 5, 5]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            # [32, 5, 5] -> [64, 3, 3]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            # [64, 3, 3] -> [128, 2, 2]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            # [128, 2, 2] -> [256, 1, 1]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),

            # [256] -> [10]
            nn.Flatten(),
            nn.Linear(256, 10, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.ann(x)

class SNN_BPTT(nn.Module):
    def __init__(self):
        super(SNN_BPTT, self).__init__()
        self.snn = nn.Sequential(
            # [2, 34, 34] -> [8, 17, 17]
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(3, 3), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
            nn.AvgPool2d(2, 2),
            # [8, 17, 17] -> [16, 8, 8]
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
            nn.AvgPool2d(2, 2),
            # [16, 8, 8] -> [32, 4, 4]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), bias=False),
            sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
            nn.AvgPool2d(2, 2),
            # [32, 4, 4] -> [64, 2, 2]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
            # Flatten 후 [64 * 2 * 2] -> [10]
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 10, bias=False),
            sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
        )

    def forward(self, x):
        return self.snn(x)

ann1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.ReLU(),
            #nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(32 * 1 * 1, 10),
            nn.ReLU(),
)

ann2 = nn.Sequential(
    # Layer 1: [2, 34, 34] -> [8, 17, 17]
    nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(3, 3), padding=(1, 1), bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    # Layer 2: [8, 17, 17] -> [16, 8, 8]
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    # Layer 3: [16, 8, 8] -> [32, 4, 4]
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    # Layer 4: [32, 4, 4] -> [64, 2, 2]
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    # Layer 5: [64, 2, 2] -> [128, 1, 1]
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
    nn.ReLU(),
    #nn.AvgPool2d(2, 2),
    # Layer 6: [128, 1, 1] -> [256, 1, 1]
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1), bias=False),
    nn.ReLU(),
    # Layer 7: Add additional layers to increase depth
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), bias=False),
    nn.ReLU(),
    # Layer 8: Final convolution layer to maintain the maximum of 8 conv layers as per the requirement
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), bias=False),
    nn.ReLU(),
    # Flatten before the Linear layer
    nn.Flatten(),
    # Linear layer: [256] -> [10]
    nn.Linear(256, 10, bias=False),
    nn.ReLU(),
)

snn1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1),
            sl.IAFSqueeze(spike_threshold=1.0, batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2),
            sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2),
            sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
            #nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(32 * 1 * 1, 10),
            sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
)