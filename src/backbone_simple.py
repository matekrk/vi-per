
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, in_channel, image_size, hidden_size = 64):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(in_channel * image_size**2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out

class SimpleCNN(nn.Module):
    def __init__(self, in_channel, hidden_channels = 32):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return out

class LeNet(nn.Module):
    def __init__(self, in_channel, option_id):
        super(LeNet, self).__init__()
        # 0: size 28, 1: size 32, 2: size 64
        options = [[[5, 1, 2], [5, 1, 0]],
                   [[3, 1, 0], [3, 1, 0]],
                   [[5, 1, 2], [5, 1, 0]]] # kernel, stride, padding
        self.option_id = option_id 
        option = options[option_id]
        self.conv1 = nn.Conv2d(in_channel, 6, kernel_size=option[0][0], stride=option[0][1], padding=option[0][2])
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=option[1][0], stride=option[1][1], padding=option[1][2])
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        final_size = self.compute_final_size(option_id, in_channel)
        self.fc1 = nn.Linear(16 * final_size * final_size, 120)
        self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, num_classes)

    def compute_final_size(self, option_id, in_channel):
        if in_channel == 3:
            match option_id:
                case 0:
                    return 5
                case 1:
                    return 6
                case 2:
                    return 14
        return

    def forward(self, x):
        out = self.conv1(x)
        out = self.avgpool1(out)
        out = self.conv2(out)
        out = self.avgpool2(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        # out = self.fc3(out)
        return out



# def SimpleMLP(**kwargs):
#     model = network_9layers(**kwargs)
#     return model

# def SimpleCNN(**kwargs):
#     model = network_9layers(**kwargs)
#     return model

# def Lenet(**kwargs):
#     model = LeNet(**kwargs)
#     return model

def SimpleMLP_templet(in_channel, input_size, hidden_size, pretrained=False):
    model = SimpleMLP(in_channel, input_size, hidden_size)
    if pretrained:
        pass
    return model

def SimpleConv_templet(in_channel, hidden_size, pretrained=False):
    model = SimpleCNN(in_channel, hidden_size)
    if pretrained:
        pass
    return model

def Lenet_templet(in_channel, option_id, pretrained=False):
    model = LeNet(in_channel, option_id)
    if pretrained:
        pass
    return model

