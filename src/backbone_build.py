import torch.nn as nn

class MultiLabelModel(nn.Module):
    def __init__(self, basemodel, basemodel_output, num_classes, binary_squeezed):
        super(MultiLabelModel, self).__init__()
        self.basemodel = basemodel
        self.num_classes = num_classes
        self.binary_squeezed = binary_squeezed
        if binary_squeezed:
            setattr(self, "FullyConnectedLayerSqueezed", nn.Linear(basemodel_output, len(num_classes)))
        else:
            for index, num_class in enumerate(num_classes):
                setattr(self, "FullyConnectedLayer_" + str(index), nn.Linear(basemodel_output, num_class))
    
    def forward(self, x):
        x = self.basemodel.forward(x)
        if self.binary_squeezed:
            out = self.FullyConnectedLayerSqueezed(x)
            return out
        outs = list()
        dir(self)
        for index, num_class in enumerate(self.num_classes):
            fun = eval("self.FullyConnectedLayer_" + str(index))
            out = fun(x)
            outs.append(out)
        return outs

def LoadPretrainedModel(model, pretrained_state_dict):
    model_dict = model.state_dict()
    union_dict = {k : v for k,v in pretrained_state_dict.iteritems() if k in model_dict}
    model_dict.update(union_dict)
    return model_dict

def BuildMultiLabelModel(basemodel, basemodel_output, num_classes, binary_squeezed):
    return MultiLabelModel(basemodel, basemodel_output, num_classes, binary_squeezed)
