import torch

model_path = "/home/parkjoe/PycharmProjects/sinabs-dynapcnn/saved_models/tutorial_cifar10_conversion_20240125_162657.pth"

model = torch.load(model_path)
print(model)

for name, param in model.state_dict().items():
    print(f"Layer: {name}")
    print(f"Parameters: {param.size()}")
    print(param)