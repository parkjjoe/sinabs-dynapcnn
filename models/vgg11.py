import torch.nn as nn

class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.vgg11 = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # 두 번째 블록
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # 세 번째 블록
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # 네 번째 블록
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # 다섯 번째 블록
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # 완전 연결 레이어
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 4096),  # 입력 크기는 컨볼루션과 풀링 레이어를 거치며 변경됨
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10)  # N-MNIST는 10개의 클래스를 가짐
        )

    def forward(self, x):
        return self.vgg11(x)
