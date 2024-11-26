import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from generate_data_matchformat import create_artificialshapes_dataset, load_artificial_shapes_dataset

def prepare_data_synthetic():
    N = 1000  # number of samples
    D = 10    # input features
    K = 4     # number of labels (multi-label classification)

    bias_on = 1.
    noise = 0.25

    torch.manual_seed(0)

    # Generate random data
    X = torch.randn(N, D)

    # Generate random weights for true labels
    true_W = torch.randn(D, K)
    # true_W = torch.ones(D, K)
    true_b = torch.randn(K)

    # Compute logits
    logits = X @ true_W + true_b*bias_on + torch.randn(N, K)*noise

    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)

    # Generate labels (multi-label)
    y = torch.bernoulli(probs)  # y is of shape (N, K), with 0s and 1s

    # Convert X,y to double precision (to match the expected dtype)
    X = X.double()
    y = y.double()

    # Generate test data
    X_test = torch.randn(10000, D)

    # Compute logits
    logits = X_test @ true_W + true_b*bias_on + torch.randn(10000, K)*noise

    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)

    # Generate labels (multi-label)
    y_test = torch.bernoulli(probs)  # y is of shape (N, K), with 0s and 1s

    # Convert X,y to double precision (to match the expected dtype)
    X_test  = X_test.double()
    y_test = y_test.double()

    return X, y, X_test, y_test


def get_appendix(coloured_background, coloured_figues, no_overlap):
    def get_bool_str(v):
        return "T" if v else "F"
    return f"cb{get_bool_str(coloured_background)}_cf{get_bool_str(coloured_figues)}_no{get_bool_str(no_overlap)}"

def create_binary_matrix(indices, num_cols):
    return np.array([[1 if j in row else 0 for j in range(num_cols)] for row in indices])

def prepare_data_shapes(cfg):

    size =  cfg.get("size", 64)
    N = cfg.get("N", 1024)
    N_test_ratio = cfg.get("N_test_ratio", 0.2)
    N_test = int(N * N_test_ratio)

    coloured_background = cfg.get("coloured_background", False)
    coloured_figues = cfg.get("coloured_figues", False)
    no_overlap = cfg.get("no_overlap", False)
    bias_classes = cfg.get("bias_classes", [0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    simplicity = cfg.get("simplicity", 3)

    main_dir = "/shared/sets/datasets/vision/artificial_shapes"
    path_to_save = os.path.join(main_dir, f"size{size}_" + f"simplicity{simplicity}_" + f"len{N}_" + get_appendix(coloured_background, coloured_figues, no_overlap))
    datasetdir = os.path.join(path_to_save, "images")
    datasettxt = os.path.join(path_to_save, "data.txt")
    labelstxt = os.path.join(path_to_save, "label.txt")
    print(f"Your data path will be: {path_to_save}")

    if os.path.isdir(path_to_save):
        dataset, labels = load_artificial_shapes_dataset(path_to_save)
    else:
        dataset, labels = create_artificialshapes_dataset(N, size, datasetdir, datasettxt, labelstxt, no_overlap, coloured_figues, coloured_background, bias_classes, simplicity)

    shapes = ['disk', 'square', 'triangle', 'star', 'hexagon', 'pentagon']

    label_ids = []
    K_calc = 0
    shape2ix = {shape: i for i, shape in enumerate(shapes)}
    for labels1 in labels:
        labels1 = [shape2ix[l[0]] for l in labels1]
        if labels1:
            K_calc = max(K_calc, max(labels1)+1)
        label_ids.append(labels1)

    K = cfg.get("K", K_calc)
    y = create_binary_matrix(label_ids, K)
    X = dataset
    X = X * (1/255)

    X_test = torch.tensor(X[-N_test:]).permute(0, 3, 1, 2).double()
    y_test = torch.tensor(y[-N_test:]).double()
    X = torch.tensor(X[:N_test]).permute(0, 3, 1, 2).double()
    y = torch.tensor(y[:N_test]).double()

    # print(N, K, X.shape, y.shape, X_test.shape, y_test.shape)
    return X, y, X_test, y_test


def prepare_dataloader(X, y, batch_size = 64):

    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
