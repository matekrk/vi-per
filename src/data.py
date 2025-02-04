import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pascal05 import PascalVOCDataset05, create_data
from pascal12 import PascalVOCDataset12, collate_wrapper, tr, augs
# FIXME: vbll path // !pip install vbll & import vbll
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generate_data_matchformat import create_artificialshapes_dataset, load_artificial_shapes_dataset

## SYNTHETIC

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

## SHAPES

# FIXME: hardcoded dependencies
def get_prependix_dependencies(ood = False):
    if ood:
        return "dependencies/dependenciesOOD"
    return "dependencies/dependenciesDEF"

def get_prependix(bias_classes):
    classes = ['disk', 'square', 'triangle', 'star', 'hexagon', 'pentagon']
    classes_symbols = ["o", "s", "^", "*", "H", "p"]
    if bias_classes is None:
        bias_classes = [1/(len(classes)) for _ in classes]
        n_classes = len(classes)
        relevant_classes = classes_symbols[:n_classes]
    else:
        relevant_classes = [classes_symbols[i] for i, b in enumerate(bias_classes) if b > 0]
    return "".join([str(b) for b in relevant_classes])

def get_appendix(coloured_background, coloured_figues, no_overlap):
    def get_bool_str(v):
        return "T" if v else "F"
    return f"cb{get_bool_str(coloured_background)}_cf{get_bool_str(coloured_figues)}_no{get_bool_str(no_overlap)}"

def create_binary_matrix(indices, num_cols):
    return np.array([[1 if j in row else 0 for j in range(num_cols)] for row in indices])

def prepare_data_shapes(cfg):
    if not isinstance(cfg, dict):
        cfg = vars(cfg)
    size =  cfg.get("size", 64)
    N = cfg.get("N", 1024)
    N_test_ratio = cfg.get("N_test_ratio", 0.75)
    N_test = int(N * N_test_ratio)

    coloured_background = cfg.get("coloured_background", False)
    coloured_figues = cfg.get("coloured_figues", False)
    no_overlap = cfg.get("no_overlap", False)
    bias_classes = cfg.get("bias_classes", [0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    simplicity = cfg.get("simplicity", 1)

    main_dir = cfg.get("data_path", "/shared/sets/datasets/vision/artificial_shapes")
    # path_to_save = os.path.join(main_dir, f"classes{get_prependix(bias_classes)}_" + f"size{size}_" + f"simplicity{simplicity}_" + f"len{N}_" + get_appendix(coloured_background, coloured_figues, no_overlap))
    path_to_save = os.path.join(main_dir, f"{get_prependix_dependencies(ood=False)}_" + f"size{size}_" + f"len{N}_" + get_appendix(coloured_background, coloured_figues, no_overlap))
    datasetdir = os.path.join(path_to_save, "images")
    datasettxt = os.path.join(path_to_save, "data.txt")
    labelstxt = os.path.join(path_to_save, "label.txt")
    targettxt = os.path.join(path_to_save, "target.txt")
    print(f"Your data path will be: {path_to_save}")

    if os.path.isdir(path_to_save):
        dataset, labels = load_artificial_shapes_dataset(datasetdir, datasettxt, labelstxt, targettxt)
    else:
        dataset, labels = create_artificialshapes_dataset(N, size, datasetdir, datasettxt, labelstxt, targettxt, no_overlap, coloured_figues, coloured_background, bias_classes, simplicity)

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

    X_test = torch.tensor(X[:N_test]).permute(0, 3, 1, 2).double()
    y_test = torch.tensor(y[:N_test]).double()
    X = torch.tensor(X[N_test:]).permute(0, 3, 1, 2).double()
    y = torch.tensor(y[N_test:]).double()

    print(f"Your data is of shape: [train] {X.shape}, [test] {X_test.shape}, [num_attributes] {y.shape[1]}")

    return X, y, X_test, y_test

def prepare_data_shapes_ood(cfg):
    if not isinstance(cfg, dict):
        cfg = vars(cfg)
    size =  cfg.get("size", 64)
    N_ood = cfg.get("N_ood", 1024)

    coloured_background = cfg.get("coloured_background", False)
    coloured_figues = cfg.get("coloured_figues", False)
    no_overlap = cfg.get("no_overlap", False)
    bias_classes_ood = cfg.get("bias_classes_ood", [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    simplicity = cfg.get("simplicity_ood", 6)

    main_dir = cfg.get("data_path", "/shared/sets/datasets/vision/artificial_shapes")
    # path_to_save = os.path.join(main_dir, f"classes{get_prependix(bias_classes_ood)}_" + f"size{size}_" + f"simplicity{simplicity}_" + f"len{N_ood}_" + get_appendix(coloured_background, coloured_figues, no_overlap))
    path_to_save = os.path.join(main_dir, f"{get_prependix_dependencies(ood=True)}_" + f"size{size}_" + f"len{N_ood}_" + get_appendix(coloured_background, coloured_figues, no_overlap))
    datasetdir = os.path.join(path_to_save, "images_ood")
    datasettxt = os.path.join(path_to_save, "data.txt")
    labelstxt = os.path.join(path_to_save, "label.txt")
    targettxt = os.path.join(path_to_save, "target.txt")
    print(f"Your data path will be: {path_to_save}")

    if os.path.isdir(path_to_save):
        dataset, labels = load_artificial_shapes_dataset(datasetdir, datasettxt, labelstxt, targettxt)
    else:
        dataset, labels = create_artificialshapes_dataset(N_ood, size, datasetdir, datasettxt, labelstxt, targettxt, no_overlap, coloured_figues, coloured_background, bias_classes_ood, simplicity)

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

    X = torch.tensor(X[:N_ood]).permute(0, 3, 1, 2).double()
    y = torch.tensor(y[:N_ood]).double()

    return X, y, None, None

## PASCAL-VOC 05

def prepare_data_pascal_voc_05(cfg):
    
    main_dir = getattr(cfg, "data_path", "/shared/sets/datasets/vision/pascal")
    train_dir = os.path.join(main_dir, "VOC2005_1")
    test_dir = os.path.join(main_dir, "VOC2005_2")

    train_json_file = os.path.join(train_dir, 'TRAIN' + '_images.json')
    test_json_file = os.path.join(test_dir, 'TEST' + '_images.json')
    print(train_json_file, test_json_file)
    if not (os.path.isfile(train_json_file) and os.path.isfile(test_json_file)):
        create_data(train_dir, test_dir, None)
    print(f"Your data path will be: {train_json_file}, {test_json_file}")

    dataset = PascalVOCDataset05(train_dir, 'TRAIN', getattr(cfg, "data_size", 300), getattr(cfg, "N", 1843))
    dataset_test = PascalVOCDataset05(test_dir, 'TEST', getattr(cfg, "data_size", 300), getattr(cfg, "N_test", 389))

    print(f"Your data is of shape: [train] {dataset.dataset_shape}, [test] {dataset_test.dataset_shape}, [num_attributes] {dataset.n_attributes}")

    return dataset, dataset_test

def prepare_loader_pascal_voc_05(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=dataset.collate_fn, num_workers=4,
                                               pin_memory=True)

## PASCAL-VOC 12

def prepare_data_pascal_voc_12(cfg):
    
    main_dir = getattr(cfg, "data_path", "/shared/sets/datasets/vision/pascal")
    data_dir = os.path.join(main_dir, "VOC2012")

    dataset = PascalVOCDataset12(data_dir, 'train', getattr(cfg, "data_size", 300), getattr(cfg, "N", 5717), transforms=augs, multi_instance=False)
    dataset_test = PascalVOCDataset12(data_dir, 'val', getattr(cfg, "data_size", 300), getattr(cfg, "N_test", 5823), transforms=tr)

    print(f"Your data is of shape: [train] {dataset.dataset_shape}, [test] {dataset_test.dataset_shape}, [num_attributes] {dataset.n_attributes}")

    return dataset, dataset_test

def prepare_loader_pascal_voc_12(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=True, num_workers=4, pin_memory=True)


## ALL

def prepare_dataset(X, y):
    return TensorDataset(X, y)

def prepare_dataloader(dataset, batch_size = 64):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
