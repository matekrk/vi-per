import copy
import argparse
import json
import os
import matplotlib.pyplot as plt
import torch

from backbone import get_backbone
from data import prepare_dataloader, prepare_data_shapes
from model import create_model
from utils import create_optimizer_scheduler, evaluate, wandb_init, log

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
    "evaluate_freq": 10,
    "verbose_freq": 50,
    "save_best": True,
    "wd": 0.0,
    "n_test_data_pred": 5,
    "f1_thres": 0.55,
    "path_to_results": "./results",
    "name_exp": "shapes_bla",
    "batch_size": 32,
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
    "wandb_user_file": "wandb.txt",
    "wandb_run_name": "shapes_bla",
    "wandb_tags": ["shapes"],
    "wandb_offline": False,
    "seed": 8
}

def train(cfg):

    X, y, X_test, y_test = prepare_data_shapes(cfg)
    data_size = X.shape[0]
    data_loader = prepare_dataloader(X, y, batch_size=cfg.batch_size)

    backbone = get_backbone(cfg)

    model = create_model(cfg, backbone)

    model_init = copy.deepcopy(model)

    evaluate(model, X_test, y_test, data_size, cfg.K)
    model.backbone.train()
    # model.m_list.train() #FIXME: handle this

    evaluate(model, X, y, data_size, cfg.K, prefix="train")
    evaluate(model, X_test, y_test, data_size, cfg.K, prefix="test")

    optimizer, scheduler = create_optimizer_scheduler(cfg, model)

    model_best = None
    f1_thres = cfg.f1_thres
    log(cfg.wandb, time=0, particular_metric_key="test_best_mean_f1", particular_metric_value = -1.0)

    metrics = {
        "train_running_loss": [],
        "train_loss": [],
        "train_f1": [],
        "train_mean_f1": [],
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "test_loss": [],
        "test_f1": [],
        "test_mean_f1": [],
        "test_accuracy": [],
        "test_precision": [],
        "test_recall": []
    }

    for epoch in range(cfg.n_epochs):
        verbose = cfg.verbose_freq and (epoch % cfg.verbose_freq == 0)
        test_evaluate = (cfg.evaluate_freq and (epoch % cfg.evaluate_freq == 0)) or epoch == cfg.n_epochs - 1

        epoch_loss = 0
        for iter, (X_batch, y_batch) in enumerate(data_loader):
            optimizer.zero_grad()
            loss = model.train_loss(X_batch, y_batch, data_size, verbose=verbose)
            loss.backward()
            epoch_loss += loss.item()
            metrics["train_running_loss"].append(loss.item())
            log(cfg.wandb, metrics, specific_key = "train_running_loss", time=epoch * len(data_loader) + iter)
            optimizer.step()
            scheduler.step()
        if verbose:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
        metrics["train_loss"].append(epoch_loss / len(data_loader))
        log(cfg.wandb, metrics, epoch, specific_key = "train_loss")

        if test_evaluate:
            train_metrics_eval = evaluate(model, X, y, data_size=data_size, K=cfg.K, prefix="train", verbose=verbose)
            log(cfg.wandb, train_metrics_eval, epoch, evaluated=True, prefix="train")
            test_metrics_eval = evaluate(model, X_test, y_test, data_size=data_size, K=cfg.K, prefix="test", verbose=verbose)
            log(cfg.wandb, test_metrics_eval, epoch, evaluated=True, prefix="test")
            metrics["train_mean_f1"].append(sum(train_metrics_eval["f1"]) / len(train_metrics_eval["f1"]))
            log(cfg.wandb, metrics, epoch, specific_key = "train_mean_f1")
            metrics["test_mean_f1"].append(sum(test_metrics_eval["f1"]) / len(test_metrics_eval["f1"]))
            log(cfg.wandb, metrics, epoch, specific_key = "test_mean_f1")
            for key in test_metrics_eval.keys():
                metrics[f"train_{key}"].append(train_metrics_eval[key])
                metrics[f"test_{key}"].append(test_metrics_eval[key])
            if metrics["test_mean_f1"][-1] > f1_thres:
                f1_thres = metrics["test_mean_f1"][-1]
                log(cfg.wandb, time=epoch, particular_metric_key="test_best_mean_f1", particular_metric_value = f1_thres)
                if cfg.save_best:
                    model_best = copy.deepcopy(model)
                    torch.save(model_best.state_dict(), os.path.join(cfg.path_to_results, cfg.name_exp, "model_best.pth"))
            if verbose and cfg.n_test_data_pred:
                preds = model.predict(X_test[0:cfg.n_test_data_pred])
                print("true", y_test[0:cfg.n_test_data_pred].detach().cpu().numpy(), "pred", preds.detach().cpu().numpy())

    # Plotting metrics
    epochs = range(cfg.n_epochs)
    fig = plt.figure(figsize=(12, 4))

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
    log(cfg.wandb, metrics, epoch, figure=fig)
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
    os.makedirs(os.path.join(cfg.get("path_to_results"), cfg.get("name_exp")), exist_ok=True)
    with open(os.path.join(cfg.get("path_to_results"), cfg.get("name_exp"), "final_cfg.json"), 'w') as f:
        json.dump(cfg, f, indent=4)
    cfg = argparse.Namespace(**cfg)
    torch.manual_seed(cfg.seed) # np.random.seed(cfg.seed)
    if cfg.wandb:
        wandb_init(cfg)
    train(cfg)
