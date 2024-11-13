import os
import json
from tqdm import tqdm
import numpy as np
import torch

def create_artificialshapes_dataset(N, dim, K, datapath, datatxt, labeltxt, seed):

    bias_on = 1.
    noise = 0.05

    data_file = open(datatxt, 'w')
    label_file = open(labeltxt, 'w')

    torch.manual_seed(seed)
    true_W = torch.randn(dim, K)
    true_b = torch.randn(K)
    data = torch.randn(N, dim)
    # labels = torch.randn(N, K)

    logits = data @ true_W + true_b*bias_on + torch.randn(N, K)*noise
    probs = torch.sigmoid(logits)
    y = torch.bernoulli(probs)
    np.save(datapath, data.double())

    for i in range(K):
        label_file.write(str(2) + f";attribute_{i};attribute_{i}" + '\n')
        label_file.write(f"no_{i}" + ';' + f"there_isno_{i}" + '\n')
        label_file.write(f"yes_{i}" + ';' + f"there_is_{i}" + '\n')

    for data_id in tqdm(range(N)):
        single_data = {
            'box': {'y': 0, 'x': 0, 'w': 1, 'h': 1},
            'box_id': f"e{data_id}x",
            'image_id': f"e{data_id}x",
            'data': data[data_id].numpy().tolist(),
            'id': [f"yes_{i}" if v == 1 else f"no_{i}" for i, v in enumerate(y[data_id])],
            'size': {'width': 1, 'height': 1}
        }
        data_file.write(json.dumps(single_data) + '\n')


def main():
    D =  10
    K = 4
    N = 10240
    seed = 1
    main_dir = f"/home/pyla/bayesian/vi-per/data/synthetic"

    path_to_save = os.path.join(main_dir, f"dim{D}_len{N}_seed{seed}")
    os.makedirs(path_to_save, exist_ok=True)

    datasetpath = os.path.join(path_to_save, "points.npy")
    datasettxt = os.path.join(path_to_save, "data.txt")
    labelstxt = os.path.join(path_to_save, "label.txt")
    create_artificialshapes_dataset(N, D, K, datasetpath, datasettxt, labelstxt, seed)

if __name__ == "__main__":
    main()