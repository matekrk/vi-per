import copy
import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import torch

from backbone import get_backbone
from data import prepare_dataloader, prepare_data_shapes
from model import create_model
from utils import create_optimizer_scheduler, evaluate, wandb_init, log

default_config = {
    "p": 64,
    "K": 6,
    "intercept": False,
    "model_type": None, # cannot be None, fill it
    "beta": 0.0,
    "f1_thres": 0.55,
    "method": None,
    "l_max": None,
    "n_samples": None,
    "scale": None,
    "VBLL_PATH": None,
    "VBLL_TYPE": None,
    "VBLL_SOFTMAX_BOUND": None,
    "VBLL_RETURN_EMPIRICAL": None,
    "VBLL_RETURN_OOD": None,
    "VBLL_PRIOR_SCALE": None,
    "VBLL_WIDTH_SCALE": None,
    "VBLL_PARAMETRIZATION": None,
    "VBLL_WISHART_SCALE": None,
    "optimizer": "adam",
    "lr": 0.001,
    "lr_mult_w": 1.0,
    "lr_mult_b": 1.0,
    "gamma": 1.0,
    "lr_decay_in_epoch": 100,
    "wd": 0.0,
    "no_wd_last": False,
    "n_epochs": 500,
    "evaluate_freq": 10,
    "verbose_freq": 50,
    "save_best": True,
    "n_test_data_pred": 5,
    "path_to_results": "./results",
    "name_exp": "shapes",
    "batch_size": 32,
    "data_channels": 3,
    "data_size": 64,
    "N": 1024,
    "N_test_ratio": 0.2,
    "coloured_background": False,
    "coloured_figues": False,
    "no_overlap": False,
    "bias_classes": [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
    "bias_classes_ood": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "simplicity": 3,
    "main_dir": "/shared/sets/datasets/vision/artificial_shapes",
    # "shapes": ['disk', 'square', 'triangle', 'star', 'hexagon', 'pentagon'],
    "wandb": False,
    "wandb_user_file": "wandb.txt",
    "wandb_run_name": "shapes",
    "wandb_tags": ["shapes"],
    "wandb_offline": False,
    "seed": 8
}

def train(cfg):

    X, y, X_test, y_test = prepare_data_shapes(cfg)
    data_size = X.shape[0]
    data_loader = prepare_dataloader(X, y, batch_size=cfg.batch_size)

    _, _, X_ood, y_ood = prepare_data_shapes(cfg)
    ood = prepare_dataloader(X_ood, y_ood, batch_size=cfg.batch_size)

    backbone = get_backbone(cfg)

    model = create_model(cfg, backbone)

    model_init = copy.deepcopy(model)

    # evaluate(model, X_test, y_test, data_size, cfg.K)
    model.backbone.train()
    # model.m_list.train() #FIXME: handle this

    evaluate(model, X, y, data_size, cfg.K, prefix="train")
    evaluate(model, X_test, y_test, data_size, cfg.K, prefix="test")
    evaluate(model, X_ood, y_ood, data_size, cfg.K, prefix="ood")

    optimizer, scheduler = create_optimizer_scheduler(cfg, model)

    model_best = None
    f1_thres = cfg.f1_thres
    log(cfg.wandb, time=0, particular_metric_key="test/best_mean_f1", particular_metric_value = -1.0)

    metrics = {
        "train/running_loss": [],
        "train/running_loss_mean": [],
        "train_loss": [],
        "train_f1": [],
        "train/mean_f1": [],
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "train_ece": [],
        "test_loss": [],
        "test_f1": [],
        "test/mean_f1": [],
        "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_ece": [],
        "test/mean_ece": [],
        "ood_loss": [],
        "ood_f1": [],
        "ood/mean_f1": [],
        "ood_accuracy": [],
        "ood_precision": [],
        "ood_recall": [],
        "ood_ece": [],
        "ood/mean_ece": []
    }
    epochs_eval = []

    for epoch in range(cfg.n_epochs):
        verbose = cfg.verbose_freq and (epoch % cfg.verbose_freq == 0)
        test_evaluate = (cfg.evaluate_freq and (epoch % cfg.evaluate_freq == 0)) or epoch == cfg.n_epochs - 1

        epoch_loss = 0
        for iter, (X_batch, y_batch) in enumerate(data_loader):
            optimizer.zero_grad()
            loss = model.train_loss(X_batch, y_batch, data_size, verbose=verbose)
            loss.backward()
            epoch_loss += loss.item()
            metrics["train/running_loss"].append(loss.item())
            log(cfg.wandb, metrics, specific_key = "train/running_loss", time=epoch * len(data_loader) + iter)
            optimizer.step()
            scheduler.step()
        if verbose:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
        metrics["train/running_loss_mean"].append(epoch_loss / len(data_loader))
        log(cfg.wandb, metrics, epoch, specific_key = "train/running_loss_mean")

        if test_evaluate:
            epochs_eval.append(epoch)
            train_metrics_eval = evaluate(model, X, y, data_size=data_size, K=cfg.K, prefix="train", verbose=verbose)
            log(cfg.wandb, train_metrics_eval, epoch, evaluated=True, prefix="train")
            test_metrics_eval = evaluate(model, X_test, y_test, data_size=data_size, K=cfg.K, prefix="test", verbose=verbose)
            log(cfg.wandb, test_metrics_eval, epoch, evaluated=True, prefix="test")
            
            metrics["train/mean_f1"].append(sum(train_metrics_eval["f1"]) / len(train_metrics_eval["f1"]))
            log(cfg.wandb, metrics, epoch, specific_key = "train/mean_f1")
            metrics["test/mean_f1"].append(sum(test_metrics_eval["f1"]) / len(test_metrics_eval["f1"]))
            log(cfg.wandb, metrics, epoch, specific_key = "test/mean_f1")
            for key in test_metrics_eval.keys():
                metrics[f"train_{key}"].append(train_metrics_eval[key])
                metrics[f"test_{key}"].append(test_metrics_eval[key])
            if metrics["test/mean_f1"][-1] > f1_thres:
                f1_thres = metrics["test/mean_f1"][-1]
                log(cfg.wandb, time=epoch, particular_metric_key="test/best_mean_f1", particular_metric_value = f1_thres)
                if cfg.save_best:
                    model_best = copy.deepcopy(model)
                    torch.save(model_best.state_dict(), os.path.join(cfg.path_to_results, cfg.name_exp, f"seed_{cfg.seed}", "model_best.pth"))
            
            metrics["test/mean_ece"].append(sum([sum(ece_list) for ece_list in test_metrics_eval["ece"]]) / (len(test_metrics_eval["ece"][0]) * len(test_metrics_eval["ece"])))
            log(cfg.wandb, metrics, epoch, specific_key = "test/mean_ece")
            if verbose and cfg.n_test_data_pred:
                preds = model.predict(X_test[0:cfg.n_test_data_pred])
                print("true:", y_test[0:cfg.n_test_data_pred].detach().cpu().numpy(), "\npred:", preds.detach().cpu().numpy())

            ood_metrics_eval = evaluate(model, X_ood, y_ood, data_size=data_size, K=cfg.K, prefix="ood", verbose=verbose)
            log(cfg.wandb, ood_metrics_eval, epoch, evaluated=True, prefix="ood")
            metrics["ood/mean_f1"].append(sum(ood_metrics_eval["f1"]) / len(ood_metrics_eval["f1"]))
            log(cfg.wandb, metrics, epoch, specific_key = "ood/mean_f1")
            metrics["ood/mean_ece"].append(sum([sum(ece_list) for ece_list in ood_metrics_eval["ece"]]) / (len(ood_metrics_eval["ece"][0]) * len(ood_metrics_eval["ece"])))
            log(cfg.wandb, metrics, epoch, specific_key = "ood/mean_ece")
            for key in ood_metrics_eval.keys():
                metrics[f"ood_{key}"].append(ood_metrics_eval[key])

        #TODO: add satisfactory_accuracy early stopping

    epochs = range(cfg.n_epochs)
    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.plot(epochs, metrics["train/running_loss_mean"], "o--", label="Train Running")
    plt.plot(epochs_eval, metrics["train_loss"], "o-", label="Train")
    plt.plot(epochs_eval, metrics["test_loss"], "^-", label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(epochs_eval, metrics["train/mean_f1"], "o-", label="Train (mean)")
    plt.plot(epochs_eval, metrics["test/mean_f1"], "^-", label="Test (mean)")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("F1 Score")
    plt.legend()

    plt.subplot(1, 4, 3)
    for k in range(cfg.K):
        plt.plot(epochs_eval, [metrics["train_accuracy"][s][k] for s in range(len(epochs_eval))], "o-", label=f"Train (Label {k})")
        plt.plot(epochs_eval, [metrics["test_accuracy"][s][k] for s in range(len(epochs_eval))], "^-", label=f"Test (Label {k})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(epochs_eval, metrics["test/mean_ece"], "^-", label="Test")
    plt.plot(epochs_eval, metrics["ood/mean_ece"], "s-", label="OOD")
    plt.xlabel("Epochs")
    plt.ylabel("ECE")
    plt.title("OOD vs Test ECE")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.path_to_results, cfg.name_exp, f"seed_{cfg.seed}", "metrics.png"))
    log(cfg.wandb, metrics, epoch, figure=fig)
    plt.show()
    return metrics

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
    os.makedirs(os.path.join(cfg.get("path_to_results"), cfg.get("name_exp"), f'seed_{cfg.get("seed")}'), exist_ok=True)
    with open(os.path.join(cfg.get("path_to_results"), cfg.get("name_exp", f'seed_{cfg.get("seed")}'), "cfg.json"), 'w') as f:
        json.dump(cfg, f, indent=4)
    cfg = argparse.Namespace(**cfg)
    cfg.vbll_cfg = argparse.Namespace(**{k[5:]: v for k, v in vars(cfg).items() if k.lower().startswith("vbll")})
    torch.manual_seed(cfg.seed) # np.random.seed(cfg.seed)
    if cfg.wandb:
        wandb_init(cfg)
    metrics_summary = train(cfg)
    with open(os.path.join(cfg.path_to_results, cfg.name_exp, f"seed_{cfg.seed}", "metrics_summary.json"), 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print("Training done")
