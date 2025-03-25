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
        ood_data_loader = prepare_loader_pascal_voc_05(dataset_ood, cfg.batch_size)

    elif cfg.data_type_ood == "pascal12_ood":
        # FIXME: prepare_data_pascal_voc_12_ood
        _, dataset_ood = prepare_data_pascal_voc_12(cfg)
        X_ood, y_ood = None, None
        ood_data_loader = prepare_loader_pascal_voc_12(dataset_ood, cfg.batch_size)

    elif cfg.data_type_ood == None:
        X_ood, y_ood = None, None
        ood_data_loader = None

    backbone = get_backbone(cfg)

    model = create_model(cfg, backbone).to(device) # can later use next(model.parameters()).device ?

    model_init = copy.deepcopy(model)

    model.backbone.train()

    # init evaluation turned off
    # evaluate(model, data_loader, cfg.K, device, prefix="train", threshold=cfg.pred_threshold)
    # evaluate(model, test_data_loder, cfg.K, device, prefix="test", threshold=cfg.pred_threshold)
    # evaluate(model, ood_data_loader, cfg.K, device, prefix="ood", threshold=cfg.pred_threshold)

    optimizer, scheduler = create_optimizer_scheduler(cfg, model)
    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of learnable parameters: {num_learnable_params}")

    model_best = None
    f1_thres = cfg.f1_thres
    best_train_f1_macro, best_train_subset_acc, best_train_acc_macro, best_train_likelihood_mean, best_train_hamming_loss, best_train_mean_ece = 0.0, 0.0, 0.0, 1.0, 1.0, 1.0
    best_test_f1_macro, best_test_subset_acc, best_test_acc_macro, best_test_likelihood_mean, best_test_hamming_loss, best_test_mean_ece = 0.0, 0.0, 0.0, 1.0, 1.0, 1.0
    log(cfg.wandb, time=0, particular_metric_key="best/train_f1_macro", particular_metric_value = best_train_f1_macro)
    log(cfg.wandb, time=0, particular_metric_key="best/train_subset_acc", particular_metric_value = best_train_subset_acc)
    log(cfg.wandb, time=0, particular_metric_key="best/train_acc_macro", particular_metric_value = best_train_acc_macro)
    log(cfg.wandb, time=0, particular_metric_key="best/train_likelihood_mean", particular_metric_value = best_train_likelihood_mean)
    log(cfg.wandb, time=0, particular_metric_key="best/train_hamming_loss", particular_metric_value = best_train_hamming_loss)
    log(cfg.wandb, time=0, particular_metric_key="best/train_mean_ece", particular_metric_value = best_train_mean_ece)
    log(cfg.wandb, time=0, particular_metric_key="best/test_f1_macro", particular_metric_value = best_test_f1_macro)
    log(cfg.wandb, time=0, particular_metric_key="best/test_subset_acc", particular_metric_value = best_test_subset_acc)
    log(cfg.wandb, time=0, particular_metric_key="best/test_acc_macro", particular_metric_value = best_test_acc_macro)
    log(cfg.wandb, time=0, particular_metric_key="best/test_likelihood_mean", particular_metric_value = best_test_likelihood_mean)
    log(cfg.wandb, time=0, particular_metric_key="best/test_hamming_loss", particular_metric_value = best_test_hamming_loss)
    log(cfg.wandb, time=0, particular_metric_key="best/test_mean_ece", particular_metric_value = best_test_mean_ece)

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
            train_metrics_eval = evaluate(model, data_loader, X, y, data_size, cfg.K, device, prefix="train", threshold=cfg.pred_threshold, verbose=verbose, plot_confusion=cfg.data_type != "pascal12")
            log(cfg.wandb, train_metrics_eval, epoch, evaluated=True, prefix="train")
            print("Test evaluation")
            test_metrics_eval = evaluate(model, test_data_loder, X_test, y_test, data_size, cfg.K, device, prefix="test", threshold=cfg.pred_threshold, verbose=verbose, plot_confusion=cfg.data_type != "pascal12")
            log(cfg.wandb, test_metrics_eval, epoch, evaluated=True, prefix="test")
            
            metrics["train/likelihood_mean"].append(sum(train_metrics_eval["likelihood"]) / len(train_metrics_eval["likelihood"]))
            log(cfg.wandb, metrics, epoch, specific_key = "train/likelihood_mean")
            metrics["test/likelihood_mean"].append(sum(test_metrics_eval["likelihood"]) / len(test_metrics_eval["likelihood"]))
            log(cfg.wandb, metrics, epoch, specific_key = "test/likelihood_mean")
            metrics["train/likelihood_mean_mc"].append(sum(train_metrics_eval["likelihood_mc"]) / len(train_metrics_eval["likelihood_mc"]))
            log(cfg.wandb, metrics, epoch, specific_key = "train/likelihood_mean_mc")
            metrics["test/likelihood_mean_mc"].append(sum(test_metrics_eval["likelihood_mc"]) / len(test_metrics_eval["likelihood_mc"]))
            log(cfg.wandb, metrics, epoch, specific_key = "test/likelihood_mean_mc")
            metrics["train/likelihood_min"].append(min(train_metrics_eval["likelihood"]))
            log(cfg.wandb, metrics, epoch, specific_key = "train/likelihood_min")
            metrics["test/likelihood_min"].append(min(test_metrics_eval["likelihood"]))
            log(cfg.wandb, metrics, epoch, specific_key = "test/likelihood_min")
            metrics["train/likelihood_min_mc"].append(min(train_metrics_eval["likelihood_mc"]))
            log(cfg.wandb, metrics, epoch, specific_key = "train/likelihood_min_mc")
            metrics["test/likelihood_min_mc"].append(min(test_metrics_eval["likelihood_mc"]))
            log(cfg.wandb, metrics, epoch, specific_key = "test/likelihood_min_mc")

            metrics["train/ece_mean"].append(sum(train_metrics_eval["ece"]) / len(train_metrics_eval["ece"]))
            log(cfg.wandb, metrics, epoch, specific_key = "train/ece_mean")

            # metrics["train/f1_mean"].append(sum(train_metrics_eval["f1"]) / len(train_metrics_eval["f1"]))
            # log(cfg.wandb, metrics, epoch, specific_key = "train/f1_mean")
            # metrics["test/f1_mean"].append(sum(test_metrics_eval["f1"]) / len(test_metrics_eval["f1"]))
            # log(cfg.wandb, metrics, epoch, specific_key = "test/f1_mean")
            for key in test_metrics_eval.keys():
                if not "plot" in key:
                    metrics[f"train_{key}"].append(train_metrics_eval[key])
                    metrics[f"test_{key}"].append(test_metrics_eval[key])

            if test_metrics_eval["f1_macro"] > f1_thres:
                f1_thres = test_metrics_eval["f1_macro"]
                log(cfg.wandb, time=epoch, particular_metric_key="test/best_f1_macro", particular_metric_value = f1_thres)
                if cfg.save_best:
                    model_best = copy.deepcopy(model)
                    torch.save(model_best.state_dict(), os.path.join(cfg.path_to_results, cfg.name_exp, f"seed_{cfg.seed}", "model_best.pth"))
            
            #metrics["test/mean_ece"].append(sum([sum(ece_list) for ece_list in test_metrics_eval["ece"]]) / (len(test_metrics_eval["ece"][0]) * len(test_metrics_eval["ece"])))
            metrics["test/ece_mean"].append(sum(test_metrics_eval["ece"]) / len(test_metrics_eval["ece"]))
            log(cfg.wandb, metrics, epoch, specific_key = "test/ece_mean")

            if train_metrics_eval["f1_macro"] > best_train_f1_macro:
                best_train_f1_macro = train_metrics_eval["f1_macro"]
                log(cfg.wandb, time=epoch, particular_metric_key="best/train_f1_macro", particular_metric_value = best_train_f1_macro)
            if train_metrics_eval["subset_accuracy"] > best_train_subset_acc:
                best_train_subset_acc = train_metrics_eval["subset_accuracy"]
                log(cfg.wandb, time=epoch, particular_metric_key="best/train_subset_acc", particular_metric_value = best_train_subset_acc)
            if train_metrics_eval["accuracy_macro"] > best_train_acc_macro:
                best_train_acc_macro = train_metrics_eval["accuracy_macro"]
                log(cfg.wandb, time=epoch, particular_metric_key="best/train_acc_macro", particular_metric_value = best_train_acc_macro)
            if metrics["train/likelihood_mean"][-1] < best_train_likelihood_mean:
                best_train_likelihood_mean = metrics["train/likelihood_mean"][-1]
                log(cfg.wandb, time=epoch, particular_metric_key="best/train_likelihood_mean", particular_metric_value = best_train_likelihood_mean)
            if train_metrics_eval["hamming_loss"]< best_train_hamming_loss:
                best_train_hamming_loss = train_metrics_eval["hamming_loss"]
                log(cfg.wandb, time=epoch, particular_metric_key="best/train_hamming_loss", particular_metric_value = best_train_hamming_loss)
            if metrics["train/ece_mean"][-1] < best_train_mean_ece:
                best_train_mean_ece = metrics["train/ece_mean"][-1]
                log(cfg.wandb, time=epoch, particular_metric_key="best/train_mean_ece", particular_metric_value = best_train_mean_ece)
            if test_metrics_eval["f1_macro"] > best_test_f1_macro:
                best_test_f1_macro = test_metrics_eval["f1_macro"]
                log(cfg.wandb, time=epoch, particular_metric_key="best/test_f1_macro", particular_metric_value = best_test_f1_macro)
            if test_metrics_eval["subset_accuracy"] > best_test_subset_acc:
                best_test_subset_acc = test_metrics_eval["subset_accuracy"]
                log(cfg.wandb, time=epoch, particular_metric_key="best/test_subset_acc", particular_metric_value = best_test_subset_acc)
            if test_metrics_eval["accuracy_macro"] > best_test_acc_macro:
                best_test_acc_macro = test_metrics_eval["accuracy_macro"]
                log(cfg.wandb, time=epoch, particular_metric_key="best/test_acc_macro", particular_metric_value = best_test_acc_macro)
            if metrics["test/likelihood_mean"][-1] < best_test_likelihood_mean:
                best_test_likelihood_mean = metrics["test/likelihood_mean"][-1]
                log(cfg.wandb, time=epoch, particular_metric_key="best/test_likelihood_mean", particular_metric_value = best_test_likelihood_mean)
            if test_metrics_eval["hamming_loss"] < best_test_hamming_loss:
                best_test_hamming_loss = test_metrics_eval["hamming_loss"]
                log(cfg.wandb, time=epoch, particular_metric_key="best/test_hamming_loss", particular_metric_value = best_test_hamming_loss)
            if metrics["test/ece_mean"][-1] < best_test_mean_ece:
                best_test_mean_ece = metrics["test/ece_mean"][-1]
                log(cfg.wandb, time=epoch, particular_metric_key="best/test_mean_ece", particular_metric_value = best_test_mean_ece)

            if verbose and cfg.n_test_data_pred:
                if X_test is not None:
                    X_to_pred = X_test[0:cfg.n_test_data_pred]
                    y_to_pred = y_test[0:cfg.n_test_data_pred].detach().cpu().numpy()
                else:
                    data_to_pred = [test_data_loder.dataset[i] for i in range(cfg.n_test_data_pred)]
                    X_to_pred = torch.stack([x[0] for x in data_to_pred])
                    y_to_pred = [x[1] for x in data_to_pred]
                
                preds = model(X_to_pred.to(device))
                print("true:", y_to_pred, "\npred:", preds.detach().cpu().numpy())
                for ii in range(cfg.n_test_data_pred):
                    fig = plt.figure()
                    plt.imshow(X_to_pred[ii].permute(1, 2, 0).detach().cpu().numpy())
                    plt.title(f"True: {y_to_pred[ii]}, Pred: {preds[ii].detach().cpu().numpy()}")
                    plt.axis("off")
                    log(cfg.wandb, particular_metric_key=f"img_{ii}", time=epoch, figure_img=fig)
                    plt.close(fig)

            if cfg.data_type_ood != None:
                print("OOD evaluation")
                ood_metrics_eval = evaluate(model, ood_data_loader, X_ood, y_ood, data_size, cfg.K, device, prefix="ood", threshold=cfg.pred_threshold, verbose=verbose, plot_confusion=cfg.data_type != "pascal12")
                log(cfg.wandb, ood_metrics_eval, epoch, evaluated=True, prefix="ood")
                metrics["ood/likelihood_mean"].append(sum(ood_metrics_eval["likelihood"]) / len(ood_metrics_eval["likelihood"]))
                log(cfg.wandb, metrics, epoch, specific_key = "ood/likelihood_mean")
                metrics["ood/likelihood_mean_mc"].append(sum(ood_metrics_eval["likelihood_mc"]) / len(ood_metrics_eval["likelihood_mc"]))
                log(cfg.wandb, metrics, epoch, specific_key = "ood/likelihood_mean_mc")
                metrics["ood/likelihood_min"].append(min(ood_metrics_eval["likelihood"]))
                log(cfg.wandb, metrics, epoch, specific_key = "ood/likelihood_min")
                metrics["ood/likelihood_min_mc"].append(min(ood_metrics_eval["likelihood_mc"]))
                log(cfg.wandb, metrics, epoch, specific_key = "ood/likelihood_min_mc")
                metrics["ood/ece_mean"].append(sum(ood_metrics_eval["ece"]) / len(ood_metrics_eval["ece"]))
                log(cfg.wandb, metrics, epoch, specific_key = "ood/ece_mean")
                for key in ood_metrics_eval.keys():
                    if not "plot" in key:
                        metrics[f"ood_{key}"].append(ood_metrics_eval[key])

        # Early stopping based on a moving window of test loss
        if cfg.early_stopping and cfg.evaluate_freq:
            patience = int(0.1 * cfg.n_epochs)
            min_delta = 0.001
            if epoch > patience * cfg.evaluate_freq:
                recent_losses = metrics["test_loss"][-patience:]
                if all(recent_losses[i] - recent_losses[i + 1] < min_delta for i in range(patience - 1)):
                    print(f"Early stopping at epoch {epoch} due to no improvement in test loss over the last {patience} epochs.")
                    break

    epochs = range(epoch+1) if cfg.early_stopping and cfg.evaluate_freq else range(cfg.n_epochs)
    fig = plt.figure(figsize=(12, 24))
    plt.subplot(5, 1, 1)
    plt.plot(epochs, metrics["train/running_loss_mean"], "o--", label="Train Running")
    plt.plot(epochs_eval, metrics["train_loss"], "o-", label="Train")
    plt.plot(epochs_eval, metrics["test_loss"], "^-", label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(5, 1, 2)
    plt.plot(epochs_eval, metrics["train_f1_macro"], "o-", label="Train (mean)")
    plt.plot(epochs_eval, metrics["test_f1_macro"], "^-", label="Test (mean)")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("F1 Score")
    plt.legend()

    plt.subplot(5, 1, 3)
    for k in range(cfg.K):
        plt.plot(epochs_eval, [metrics["train_accuracy"][s][k] for s in range(len(epochs_eval))], "o-", label=f"Train (Label {k})")
        plt.plot(epochs_eval, [metrics["test_accuracy"][s][k] for s in range(len(epochs_eval))], "^-", label=f"Test (Label {k})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(epochs_eval, metrics["test/ece_mean"], "^-", label="Test")
    if cfg.data_type_ood != None:
        plt.plot(epochs_eval, metrics["ood/ece_mean"], "s-", label="OOD")
    plt.xlabel("Epochs")
    plt.ylabel("ECE")
    plt.title("Test (vs OOD) ECE")
    plt.legend()

    ax = plt.subplot(5, 1, 5)
    if X_test is not None and y_test is not None:
        confusion_matrix = compute_confusion_matrix(model, X_test, y_test, cfg.K, device, threshold=cfg.f1_thres)
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
    log(cfg.wandb, metrics, epoch, figure_summary=fig)
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
