import torch
import torch.nn as nn

### BACKBONE
class ConvNet(nn.Module):
    def __init__(self, input_channels, input_size):
        """
        p = 64
        backbone = ConvNet(3, 64).to(torch.double)
        """
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.avgpool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.final_size = 64 * 9 * 9
        self.fc1 = nn.Linear(self.final_size, 64)
        """
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | Number of parameters: {param.numel()}")
        """

    def forward(self, x):
        x = self.avgpool1(nn.functional.relu(self.conv1(x)))
        x = self.avgpool2(nn.functional.relu(self.conv2(x)))
        x = self.avgpool3(nn.functional.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(self.fc1(x))
        return x

def get_backbone(cfg):
    p = cfg.get("p", 64)
    data_channels = cfg.get("data_channels", 3)
    backbone = ConvNet(data_channels, p).to(torch.double)
    return backbone
