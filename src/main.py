import copy
import argparse
import json
import os
import matplotlib.pyplot as plt
import torch

from backbone import get_backbone
from data import prepare_dataloader, prepare_data_shapes
from model import create_model
from utils import create_optimizer_scheduler, evaluate

default_config = {
    "p": 64,
    "K": 6,
    "beta": 0.0,
    "method": 0,
    "l_max": 10.0,
    "n_samples": 1000,
    "intercept": False,
    "optimizer": "adam",
    "lr": 0.001,
    "n_epochs": 500,
    "wd": 0.0,
    "n_test_data_pred": 5,
    "f1_thres": 0.55,
    "path_to_results": "./results",
    "name_exp": "shapes_bla",
    "verbose": True,
    "data_channels": 3,
    "data_size": 64,
    "N": 1024,
    "N_test_ratio": 0.2,
    "coloured_background": False,
    "coloured_figues": False,
    "no_overlap": False,
    "bias_classes": [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
    "simplicity": 3,
    "main_dir": "/shared/sets/datasets/vision/artificial_shapes",
    # "shapes": ['disk', 'square', 'triangle', 'star', 'hexagon', 'pentagon'],
    "wandb": False,
}

def train(cfg):

    X, y, X_test, y_test = prepare_data_shapes(cfg)
    data_size = X.shape[0]
    data_loader = prepare_dataloader(X, y, batch_size=cfg.batch_size)

    backbone = get_backbone(cfg)

    model = create_model(cfg, backbone)

    model_init = copy.deepcopy(model)

    evaluate(model, X_test, y_test, cfg.K)
    model.backbone.train()
    model.m_list.train()

    evaluate(model, X, y, cfg.K, prefix="train")
    evaluate(model, X_test, y_test, cfg.K, prefix="test")

    optimizer, scheduler = create_optimizer_scheduler(cfg, model)

    model_best = None
    f1_thres = cfg.f1_thres

    metrics = {
        "train_loss": [],
        "train_f1": [],
        "test_f1": []
    }

    for epoch in range(cfg.n_epochs):
        verbose = cfg.verbose and (epoch % 10 == 0)

        epoch_loss = 0
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            loss = model.train_loss(X_batch, y_batch, data_size, verbose=verbose)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        metrics["train_loss"].append(epoch_loss / len(data_loader))

        if verbose:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
            train_f1 = evaluate(model, X, y, data_size=data_size, K=cfg.K, prefix="train")
            test_f1 = evaluate(model, X_test, y_test, prefix="test")
            train_f1 = sum(train_f1) / len(train_f1)
            test_f1 = sum(test_f1) / len(test_f1)
            metrics["train_f1"].append(train_f1)
            metrics["test_f1"].append(test_f1)
            if test_f1 > f1_thres:
                f1_thres = test_f1
                model_best = copy.deepcopy(model)
                torch.save(model_best.state_dict(), os.path.join(cfg.path_to_results, cfg.name_exp, "model_best.pth"))
            preds = model.predict(X_test[0:cfg.n_test_data_pred])
            print("true", y_test[0:cfg.n_test_data_pred], "pred", preds)

    # Plotting metrics
    epochs = range(cfg.n_epochs)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, metrics["train_f1"], label="Train F1")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("Training F1 Score")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, metrics["test_f1"], label="Test F1")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("Test F1 Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.path_to_results, cfg.name_exp, "metrics.png"))
    plt.show()

if __name__ == "__main__":
    def merge_configs(default_cfg, user_cfg):
        merged_cfg = default_cfg.copy()
        merged_cfg.update(user_cfg)
        return merged_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file')
    
    args = parser.parse_args()

    user_config = {}
    if args.config:
        with open(args.config, 'r') as f:
            user_config = json.load(f)

    cfg = merge_configs(default_config, user_config)
    os.makedirs(os.path.join(cfg.path_to_results, cfg.name_exp), exist_ok=True)
    with open(os.path.join(cfg.path_to_results, cfg.name_exp, "final_cfg.json"), 'w') as f:
        json.dump(cfg, f, indent=4)
    train(cfg)
