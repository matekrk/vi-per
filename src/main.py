import copy
import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import torch

from backbone import get_backbone
from data import prepare_dataset, prepare_dataloader, \
    prepare_data_shapes, prepare_data_shapes_ood, \
    prepare_data_pascal_voc_05, prepare_data_pascal_voc_12, \
    prepare_loader_pascal_voc_05, prepare_loader_pascal_voc_12
from model import create_model
from utils import compute_confusion_matrix, create_optimizer_scheduler, empty_metrics, default_config, evaluate, wandb_init, log

default_config = default_config()

def train(cfg):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): 
        device = torch.device("cuda") 
        print("Training on GPU") 
    else: 
        device = torch.device("cpu") 
        print("Training on CPU")

    if cfg.data_type == "shapes":
        X, y, X_test, y_test = prepare_data_shapes(cfg)
        data_size = X.shape[0]
        dataset = prepare_dataset(X, y)
        test_dataset = prepare_dataset(X_test, y_test)
        data_loader = prepare_dataloader(dataset, batch_size=cfg.batch_size)
        test_data_loder = prepare_dataloader(test_dataset, batch_size=cfg.batch_size)
    
    elif cfg.data_type == "pascal05":
        dataset, dataset_test = prepare_data_pascal_voc_05(cfg)
        data_size = len(dataset)
        data_loader = prepare_loader_pascal_voc_05(dataset, cfg.batch_size)
        X, y, X_test, y_test = None, None, None, None
        test_data_loder = prepare_loader_pascal_voc_05(dataset_test, cfg.batch_size)

    elif cfg.data_type == "pascal12":
        dataset, dataset_test = prepare_data_pascal_voc_12(cfg)
        data_size = len(dataset)
        data_loader = prepare_loader_pascal_voc_12(dataset, cfg.batch_size)
        X, y, X_test, y_test = None, None, None, None
        test_data_loder = prepare_loader_pascal_voc_12(dataset_test, cfg.batch_size)

    if cfg.data_type_ood == "shapes_ood":
        X_ood, y_ood, _, _ = prepare_data_shapes_ood(cfg)
        dataset_ood = prepare_dataset(X_ood, y_ood)
        ood_data_loader = prepare_dataloader(dataset_ood, batch_size=cfg.batch_size)

    elif cfg.data_type_ood == "pascal05_ood":
        # FIXME: prepare_data_pascal_voc_05_ood
        _, dataset_ood = prepare_data_pascal_voc_05(cfg)
        X_ood, y_ood = None, None
        ood_data_loader = prepare_loader_pascal_voc_05(dataset_test, cfg.batch_size)

    elif cfg.data_type_ood == "pascal12_ood":
        dataset, dataset_test = prepare_data_pascal_voc_12(cfg)
        data_size = len(dataset)
        data_loader = prepare_loader_pascal_voc_12(dataset, cfg.batch_size)
        X, y, X_test, y_test = None, None, None, None
        test_data_loder = prepare_loader_pascal_voc_12(dataset_test, cfg.batch_size)

    backbone = get_backbone(cfg)

    model = create_model(cfg, backbone).to(device) # can later use next(model.parameters()).device ?

    model_init = copy.deepcopy(model)

    model.backbone.train()

    # init evaluation turned off
    # evaluate(model, data_loader, X, y, data_size, cfg.K, device, prefix="train", threshold=cfg.pred_threshold)
    # evaluate(model, test_data_loder, X_test, y_test, data_size, cfg.K, device, prefix="test", threshold=cfg.pred_threshold)
    # evaluate(model, ood_data_loader, X_ood, y_ood, data_size, cfg.K, device, prefix="ood", threshold=cfg.pred_threshold)

    optimizer, scheduler = create_optimizer_scheduler(cfg, model)
    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of learnable parameters: {num_learnable_params}")

    model_best = None
    f1_thres = cfg.f1_thres
    log(cfg.wandb, time=0, particular_metric_key="test/best_mean_f1", particular_metric_value = -1.0)

    metrics = empty_metrics()
    epochs_eval = []

    for epoch in range(cfg.n_epochs):
        verbose = cfg.verbose_freq and (epoch % cfg.verbose_freq == 0)
        verbose_iter = False
        test_evaluate = (cfg.evaluate_freq and (epoch % cfg.evaluate_freq == 0)) or epoch == cfg.n_epochs - 1

        grad_norm_accum = 0.0
        param_norm_accum = 0.0

        epoch_loss = 0
        model.train()
        for iter, (X_batch, y_batch) in enumerate(data_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = model.train_loss(X_batch, y_batch, data_size, verbose=verbose_iter)
            loss.backward()
            epoch_loss += loss.item()
            metrics["running_loss"].append(loss.item())
            log(cfg.wandb, metrics, specific_key = "running_loss", time=epoch * len(data_loader) + iter, time_metric="iter")

            if test_evaluate:
                grad_norm = 0.0
                param_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                        param_norm += param.data.norm().item() ** 2
                grad_norm_accum += grad_norm ** 0.5
                param_norm_accum += param_norm ** 0.5

            optimizer.step()
            scheduler.step()

        if verbose:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
        metrics["train/running_loss_mean"].append(epoch_loss / len(data_loader))
        log(cfg.wandb, metrics, epoch, specific_key = "train/running_loss_mean")            

        if cfg.model_type == "logisticvi" and (cfg.prior_mean_learnable or cfg.prior_scale_learnable):
            metrics["vi/prior_mu"].append(model.prior_mu.item())
            metrics["vi/u_sig"].append(model.u_sig.item())
            metrics["vi/sig"].append(model.prior_scale.item())
            log(cfg.wandb, metrics, epoch, specific_key = "vi/prior_mu")
            log(cfg.wandb, metrics, epoch, specific_key = "vi/u_sig")
            log(cfg.wandb, metrics, epoch, specific_key = "vi/sig")

        if test_evaluate:

            grad_norm_avg = grad_norm_accum / len(data_loader)
            param_norm_avg = param_norm_accum / len(data_loader)

            metrics["grad_norm"].append(grad_norm_avg)
            log(cfg.wandb, metrics, epoch, specific_key="grad_norm")

            metrics["param_norm"].append(param_norm_avg)
            log(cfg.wandb, metrics, epoch, specific_key="param_norm")

            epochs_eval.append(epoch)
            print("Train evaluation")
            train_metrics_eval = evaluate(model, data_loader, X, y, data_size, cfg.K, device, prefix="train", threshold=cfg.pred_threshold, verbose=verbose)
            log(cfg.wandb, train_metrics_eval, epoch, evaluated=True, prefix="train")
            print("Test evaluation")
            test_metrics_eval = evaluate(model, test_data_loder, X_test, y_test, data_size, cfg.K, device, prefix="test", threshold=cfg.pred_threshold, verbose=verbose)
            log(cfg.wandb, test_metrics_eval, epoch, evaluated=True, prefix="test")
            
            metrics["train/mean_likelihood"].append(sum(train_metrics_eval["likelihood"]) / len(train_metrics_eval["likelihood"]))
            log(cfg.wandb, metrics, epoch, specific_key = "train/mean_likelihood")
            metrics["test/mean_likelihood"].append(sum(test_metrics_eval["likelihood"]) / len(test_metrics_eval["likelihood"]))
            log(cfg.wandb, metrics, epoch, specific_key = "test/mean_likelihood")
            metrics["train/mean_likelihood_mc"].append(sum(train_metrics_eval["likelihood_mc"]) / len(train_metrics_eval["likelihood_mc"]))
            log(cfg.wandb, metrics, epoch, specific_key = "train/mean_likelihood_mc")
            metrics["test/mean_likelihood_mc"].append(sum(test_metrics_eval["likelihood_mc"]) / len(test_metrics_eval["likelihood_mc"]))
            log(cfg.wandb, metrics, epoch, specific_key = "test/mean_likelihood_mc")
            metrics["train/min_likelihood"].append(min(train_metrics_eval["likelihood"]))
            log(cfg.wandb, metrics, epoch, specific_key = "train/min_likelihood")
            metrics["test/min_likelihood"].append(min(test_metrics_eval["likelihood"]))
            log(cfg.wandb, metrics, epoch, specific_key = "test/min_likelihood")
            metrics["train/min_likelihood_mc"].append(min(train_metrics_eval["likelihood_mc"]))
            log(cfg.wandb, metrics, epoch, specific_key = "train/min_likelihood_mc")
            metrics["test/min_likelihood_mc"].append(min(test_metrics_eval["likelihood_mc"]))
            log(cfg.wandb, metrics, epoch, specific_key = "test/min_likelihood_mc")

            metrics["train/mean_ece"].append(sum(train_metrics_eval["ece"]) / len(train_metrics_eval["ece"]))
            log(cfg.wandb, metrics, epoch, specific_key = "train/mean_ece")

            metrics["train/mean_f1"].append(sum(train_metrics_eval["f1"]) / len(train_metrics_eval["f1"]))
            log(cfg.wandb, metrics, epoch, specific_key = "train/mean_f1")
            metrics["test/mean_f1"].append(sum(test_metrics_eval["f1"]) / len(test_metrics_eval["f1"]))
            log(cfg.wandb, metrics, epoch, specific_key = "test/mean_f1")
            for key in test_metrics_eval.keys():
                if not "plot" in key:
                    metrics[f"train_{key}"].append(train_metrics_eval[key])
                    metrics[f"test_{key}"].append(test_metrics_eval[key])

            if metrics["test/mean_f1"][-1] > f1_thres:
                f1_thres = metrics["test/mean_f1"][-1]
                log(cfg.wandb, time=epoch, particular_metric_key="test/best_mean_f1", particular_metric_value = f1_thres)
                if cfg.save_best:
                    model_best = copy.deepcopy(model)
                    torch.save(model_best.state_dict(), os.path.join(cfg.path_to_results, cfg.name_exp, f"seed_{cfg.seed}", "model_best.pth"))
            
            #metrics["test/mean_ece"].append(sum([sum(ece_list) for ece_list in test_metrics_eval["ece"]]) / (len(test_metrics_eval["ece"][0]) * len(test_metrics_eval["ece"])))
            metrics["test/mean_ece"].append(sum(test_metrics_eval["ece"]) / len(test_metrics_eval["ece"]))
            log(cfg.wandb, metrics, epoch, specific_key = "test/mean_ece")

            if verbose and cfg.n_test_data_pred:
                if X_test is not None:
                    X_to_pred = X_test[0:cfg.n_test_data_pred]
                    y_to_pred = y_test[0:cfg.n_test_data_pred].detach().cpu().numpy()
                else:
                    data_to_pred = [test_data_loder.dataset[i] for i in range(cfg.n_test_data_pred)]
                    X_to_pred = torch.stack([x[0] for x in data_to_pred])
                    y_to_pred = [x[1] for x in data_to_pred]
                    
                    # p = [x[1] - 1 for x in data_to_pred]
                    # y = torch.zeros(cfg.n_test_data_pred, cfg.K)
                    # row_indices = torch.arange(cfg.n_test_data_pred)
                    # single_mask = [isinstance(idx, torch.Tensor) and idx.ndim == 0 for idx in p]
                    # single_indices = [idx.item() for idx, is_single in zip(p, single_mask) if is_single]
                    # single_rows = [i for i, is_single in enumerate(single_mask) if is_single]
                    # if single_rows:
                    #     y[single_rows, single_indices] = 1
                    # multi_mask = [isinstance(idx, torch.Tensor) and idx.ndim > 0 for idx in p]
                    # for row, (idx, is_multi) in enumerate(zip(p, multi_mask)):
                    #     if is_multi:
                    #         y[row, idx] = 1
                
                preds = model.predict(X_to_pred.to(device))
                print("true:", y_to_pred, "\npred:", preds.detach().cpu().numpy())

            print("OOD evaluation")
            ood_metrics_eval = evaluate(model, ood_data_loader, X_ood, y_ood, data_size, cfg.K, device, prefix="ood", threshold=cfg.pred_threshold, verbose=verbose)
            log(cfg.wandb, ood_metrics_eval, epoch, evaluated=True, prefix="ood")
            metrics["ood/mean_likelihood"].append(sum(ood_metrics_eval["likelihood"]) / len(ood_metrics_eval["likelihood"]))
            log(cfg.wandb, metrics, epoch, specific_key = "ood/mean_likelihood")
            metrics["ood/mean_likelihood_mc"].append(sum(ood_metrics_eval["likelihood_mc"]) / len(ood_metrics_eval["likelihood_mc"]))
            log(cfg.wandb, metrics, epoch, specific_key = "ood/mean_likelihood_mc")
            metrics["ood/min_likelihood"].append(min(ood_metrics_eval["likelihood"]))
            log(cfg.wandb, metrics, epoch, specific_key = "ood/min_likelihood")
            metrics["ood/min_likelihood_mc"].append(min(ood_metrics_eval["likelihood_mc"]))
            log(cfg.wandb, metrics, epoch, specific_key = "ood/min_likelihood_mc")
            metrics["ood/mean_f1"].append(sum(ood_metrics_eval["f1"]) / len(ood_metrics_eval["f1"]))
            log(cfg.wandb, metrics, epoch, specific_key = "ood/mean_f1")
            metrics["ood/mean_ece"].append(sum(ood_metrics_eval["ece"]) / len(ood_metrics_eval["ece"]))
            #metrics["ood/mean_ece"].append(sum([sum(ece_list) for ece_list in ood_metrics_eval["ece"]]) / (len(ood_metrics_eval["ece"][0]) * len(ood_metrics_eval["ece"])))
            log(cfg.wandb, metrics, epoch, specific_key = "ood/mean_ece")
            for key in ood_metrics_eval.keys():
                if not "plot" in key:
                    metrics[f"ood_{key}"].append(ood_metrics_eval[key])

        # Early stopping based on a moving window of test loss
        if cfg.early_stopping and cfg.evaluate_freq:
            patience = 10
            min_delta = 0.001
            if epoch > patience * cfg.evaluate_freq:
                recent_losses = metrics["test_loss"][-patience:]
                if all(recent_losses[i] - recent_losses[i + 1] < min_delta for i in range(patience - 1)):
                    print(f"Early stopping at epoch {epoch} due to no improvement in test loss over the last {patience} epochs.")
                    break

    epochs = range(epoch+1) if cfg.early_stopping and cfg.evaluate_freq else range(cfg.n_epochs)
    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 5, 1)
    plt.plot(epochs, metrics["train/running_loss_mean"], "o--", label="Train Running")
    plt.plot(epochs_eval, metrics["train_loss"], "o-", label="Train")
    plt.plot(epochs_eval, metrics["test_loss"], "^-", label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 5, 2)
    plt.plot(epochs_eval, metrics["train/mean_f1"], "o-", label="Train (mean)")
    plt.plot(epochs_eval, metrics["test/mean_f1"], "^-", label="Test (mean)")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("F1 Score")
    plt.legend()

    plt.subplot(1, 5, 3)
    for k in range(cfg.K):
        plt.plot(epochs_eval, [metrics["train_accuracy"][s][k] for s in range(len(epochs_eval))], "o-", label=f"Train (Label {k})")
        plt.plot(epochs_eval, [metrics["test_accuracy"][s][k] for s in range(len(epochs_eval))], "^-", label=f"Test (Label {k})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 5, 4)
    plt.plot(epochs_eval, metrics["test/mean_ece"], "^-", label="Test")
    plt.plot(epochs_eval, metrics["ood/mean_ece"], "s-", label="OOD")
    plt.xlabel("Epochs")
    plt.ylabel("ECE")
    plt.title("OOD vs Test ECE")
    plt.legend()

    confusion_matrix = compute_confusion_matrix(model, X_test, y_test, cfg.K, device, threshold=cfg.f1_thres)
    ax = plt.subplot(1, 5, 5)
    cmap = plt.get_cmap('RdYlGn')
    for i in range(cfg.K):
        for j in range(cfg.K):
            ax.text(j, i, confusion_matrix[i, j].item(), ha='center', va='center', fontsize=6, color='black')
    cax = ax.matshow(confusion_matrix, cmap=cmap)
    plt.xticks(range(cfg.K), [f"A {k}" for k in range(cfg.K)])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Final Confusion Matrix")
    plt.colorbar(cax, ax=ax)

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
    # print("[End of training] Metrics summary:")
    # print(metrics_summary)
    with open(os.path.join(cfg.path_to_results, cfg.name_exp, f"seed_{cfg.seed}", "metrics_summary.json"), 'w') as f:
        try:
            json.dump(metrics_summary, f, indent=4)
        except:
            print("Error in saving metrics_summary")
            for k, v in metrics_summary.items():
                if isinstance(v, list):
                    if isinstance(v[0], list):
                        print(k, type(v[0][0]))
                    else:
                        print(k, type(v[0]))
                else:
                    print(k, type(v))
    print("Training done")
