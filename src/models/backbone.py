import torch
import torch.nn as nn

"""# BACKBONE """

"""# UTILS """
def get_activation(activation_name):
    """
    Get the activation function by name.

    Args:
        activation_name (str): Name of the activation function (e.g., 'relu', 'sigmoid', 'tanh').

    Returns:
        torch.nn.Module: The corresponding activation function.

    Raises:
        ValueError: If the activation name is not recognized.
    """
    if activation_name is None:
        activation_name = "none"

    activations = {
        "relu": torch.nn.ReLU(),
        "sigmoid": torch.nn.Sigmoid(),
        "tanh": torch.nn.Tanh(),
        "softmax": torch.nn.Softmax(dim=-1),
        "leaky_relu": torch.nn.LeakyReLU(),
        "elu": torch.nn.ELU(),
        "selu": torch.nn.SELU(),
        "gelu": torch.nn.GELU(),
        "softplus": torch.nn.Softplus(),
        "none": torch.nn.Identity()  # No activation
    }

    if activation_name.lower() not in activations:
        raise ValueError(f"Unsupported activation function: {activation_name}. "
                        f"Supported activations are: {list(activations.keys())}")

    return activations[activation_name.lower()]

class ConvNet(nn.Module):
    """
    A simple convolutional neural network (ConvNet) for processing input data.
    This network consists of three convolutional layers, each followed by an average pooling layer.
    The final output is flattened and passed through a fully connected layer to produce the output of size `p`.
    """
    def __init__(self, input_channels, input_size, p=64, activation = "relu", activation_in_the_end = None):
        """
        Initialize an instance of the ConvNet class.

        Args:
            input_channels (int): Number of input channels for the convolutional layers (e.g., 3 for RGB images).
            input_size (int): Size of the input image (assumes square images with dimensions input_size x input_size).
            p (int, optional): Dimensionality of the output features after processing by the network. Defaults to 64.

        Notes:
            - The ConvNet architecture consists of three convolutional layers, each followed by an average pooling layer.
            - The final output is flattened and passed through a fully connected layer to produce the output of size `p`.
            - The `final_size` attribute is calculated based on the input size and the architecture of the network.
        """
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.avgpool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.final_size = p * 9 * 9
        self.fc1 = nn.Linear(self.final_size, p)
        self.activation = get_activation(activation)
        self.activation_in_the_end = get_activation(activation_in_the_end)
        """
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | Number of parameters: {param.numel()}")
        """

    def forward(self, x):
        x = self.avgpool1(self.activation(self.conv1(x)))
        x = self.avgpool2(self.activation(self.conv2(x)))
        x = self.avgpool3(self.activation(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.activation_in_the_end(x)
        return x

class MLP(nn.Module):
    """
    A simple two-layer fully connected neural network (MLP) for processing input data.
    """
    def __init__(self, input_channels, input_size, p=64, activation = "relu", activation_in_the_end = None):
        """
        Initialize the MLP class.
        Args:
            input_channels (int): Number of input channels in the data.
            input_size (int): Height and width of the input data (assumed to be square).
            p (int, optional): Number of units in the hidden layer. Defaults to 64.
        Notes:
            - The input data is flattened before being passed to the first fully connected layer.
            - This class defines a simple two-layer fully connected neural network.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_channels * input_size * input_size, p)
        self.fc2 = nn.Linear(p, p)
        self.activation = get_activation(activation)
        self.activation_in_the_end = get_activation(activation_in_the_end)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x = self.activation_in_the_end(x)
        return x

def get_backbone(cfg):
    """
    Constructs a backbone neural network based on the provided configuration.
    Args:
        cfg (object): Configuration object containing the following attributes:
            - backbone_type (str): Specifies the type of backbone to use. 
              Must be one of ["ConvNet", "MLP", "ResNet34", "ResNet50", "ResNet18", "MobileNetV2"].
            - p (int, optional): Dimensionality of the output features from the backbone. Defaults to 64.
            - data_channels (int, optional): Number of input data channels. Defaults to 3.
            - data_size (int, optional): Input data size, required for MLP backbone.
            - backbone_pretrained (bool, optional): Whether to use a pretrained model for ResNet or MobileNetV2 backbones.
    Returns:
        torch.nn.Module: The constructed backbone neural network.
    Raises:
        AttributeError: If `cfg` does not have the required attributes for the specified backbone type.
        ValueError: If `cfg.backbone_type` is not one of the supported backbone types.
    Notes:
        - For ResNet and MobileNetV2 backbones, the final layers are modified to exclude the classifier 
          and include a flattening operation for feature extraction.
        - The `cfg.backbone_pretrained` attribute is used to load pretrained weights for ResNet and MobileNetV2 models.
    """

    p = cfg.p if hasattr(cfg, 'p') else 64
    data_channels = cfg.data_channels if hasattr(cfg, 'data_channels') else 3
    activation = cfg.activation if hasattr(cfg, 'activation') else "relu"
    activation_in_the_end = cfg.activation_in_the_end if hasattr(cfg, 'activation_in_the_end') else None
    if cfg.backbone_type == 'ConvNet':
        backbone = ConvNet(data_channels, p, activation=activation, activation_in_the_end=activation_in_the_end)
    elif cfg.backbone_type == 'MLP':
        backbone = MLP(data_channels, cfg.data_size, p, activation=activation, activation_in_the_end=activation_in_the_end)
    elif cfg.backbone_type == 'ResNet34':
        import torchvision.models as torchmodels
        backbone = torchmodels.resnet34(pretrained=cfg.backbone_pretrained)
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten(start_dim=1))
    elif cfg.backbone_type == 'ResNet50':
        import torchvision.models as torchmodels
        backbone = torchmodels.resnet50(pretrained=cfg.backbone_pretrained)
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten(start_dim=1))
    elif cfg.backbone_type == 'ResNet18':
        import torchvision.models as torchmodels
        backbone = torchmodels.resnet18(pretrained=cfg.backbone_pretrained)
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten(start_dim=1))
    elif cfg.backbone_type == "MobileNetV2":
        import torchvision.models as torchmodels
        backbone = torchmodels.mobilenet_v2(pretrained=cfg.backbone_pretrained)
        backbone = nn.Sequential(*list(backbone.children())[:-1] + [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1)])
    elif cfg.backbone_type == "none":
        backbone = None
    else:
        raise ValueError(f"Backbone type '{cfg.backbone_type}' not recognized. Please specify a valid backbone type.")
    return backbone.to(torch.double) if backbone is not None else None
