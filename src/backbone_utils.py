import os
import logging
import sys
import torch
from torch.autograd import Variable

from backbone_build import BuildMultiLabelModel
from backbone_build_bayesian import BuildMultiLabelModelBayesian
from backbone_cnn import LightCNN_9Layers_templet, LightCNN_29Layers_v2_templet, LightCNN_29Layers, LightCNN_29Layers_v2
from backbone_resnet import Resnet18Templet
from backbone_simple import SimpleMLP_templet, SimpleConv_templet, Lenet_templet

def save_model(model, opt, epoch):
    checkpoint_name = opt.model_dir + "/epoch_%s_snapshot.pth" %(epoch)
    torch.save(model.cpu().state_dict(), checkpoint_name)
    if opt.cuda and torch.cuda.is_available():
        model.cuda(opt.devices[0])

def modify_last_layer_lr(named_params, base_lr, lr_mult_w, lr_mult_b, bayesian = False):
    params = list()
    if bayesian:
        for name, param in named_params:
            params += [{'params': param, 'lr': base_lr * 1}]
            if "BayesianLastLayer_" in name:
                pass
            else:
                pass
    else:
        for name, param in named_params: 
            if 'bias' in name:
                if 'FullyConnectedLayer_' in name:
                    params += [{'params': param, 'lr': base_lr * lr_mult_b, 'weight_decay': 0}]
                else:
                    params += [{'params': param, 'lr': base_lr * 2, 'weight_decay': 0}]
            else:
                if 'FullyConnectedLayer_' in name:
                    params += [{'params': param, 'lr': base_lr * lr_mult_w}]
                else:
                    params += [{'params': param, 'lr': base_lr * 1}]
    return params

def load_model(opt, num_classes, binary_squeezed, bayesian = False, 
               len_dataset = None, mu = None, sig = None, Sig = None, s_init = None, intercept = None):
    # load templet
    # if opt.model == "Alexnet":
    #     templet = AlexnetTemplet(opt.input_channel, opt.pretrain)
    if opt.model == "LightenB":
        templet = LightCNN_29Layers_v2_templet(opt.input_channel, opt.pretrain)
    elif opt.model == "Lighten9":
        templet = LightCNN_9Layers_templet(opt.input_channel, opt.pretrain)
    elif opt.model == "Resnet18":
        templet = Resnet18Templet(opt.input_channel, opt.pretrain)
    # elif opt.model == "VGG16":
    #     templet = VGG16Templet(opt.input_channel, opt.pretrain)
    elif opt.model == "SimpleMLP":
        templet = SimpleMLP_templet(opt.input_channel, opt.pretrain)
    elif opt.model == "SimpleConv":
        templet = SimpleConv_templet(opt.input_channel, opt.pretrain)
    elif opt.model == "Lenet":
        option_id = 0 if opt.input_size == 28 else (1 if opt.input_size == 32 else 2)
        templet = Lenet_templet(opt.input_channel, option_id, opt.pretrain)
    else:
        logging.error("unknown model type")
        sys.exit(0)
    
    # build model
    tmp_input = Variable(torch.FloatTensor(1, opt.input_channel, opt.input_size, opt.input_size))
    tmp_output = templet(tmp_input)
    output_dim = int(tmp_output.size()[-1])
    if bayesian:
        model = BuildMultiLabelModelBayesian(templet, output_dim, num_classes, binary_squeezed,
                                             len_dataset, mu, sig, Sig, s_init, intercept)
        model.init_covar(opt.method)
    else:
        model = BuildMultiLabelModel(templet, output_dim, num_classes, binary_squeezed)
    logging.info(model)
    
    # imagenet pretrain model
    if opt.pretrain:
        logging.info("use imagenet pretrained model")
    
    # load exsiting model
    if opt.checkpoint_name != "":
        if os.path.exists(opt.checkpoint_name):
            logging.info("load pretrained model from "+opt.checkpoint_name)
            model.load_state_dict(torch.load(opt.checkpoint_name))
        elif os.path.exists(opt.model_dir):
            checkpoint_name = opt.model_dir + "/" + opt.checkpoint_name
            model.load_state_dict(torch.load(checkpoint_name))
            logging.info("load pretrained model from "+ checkpoint_name)
        else:
            opt.checkpoint_name = ""
            logging.warning("WARNING: unknown pretrained model, skip it.")

    return model
