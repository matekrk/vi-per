import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.optim as optim
import wandb

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def evaluate(model, test_dataloader, X_test, y_test, data_size, K, device, prefix = "", threshold = 0.5, verbose = False):
    model.eval()
    preds = []
    y_tests = []
    total_loss = 0.0
    total_likelihoods = []
    total_sum_likelihood = 0.0
    total_likelihoods_mc = []
    total_sum_likelihood_mc = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            batch_preds = model.predict(X_batch)
            preds.append(batch_preds)
            y_tests.append(y_batch)
            total_loss += model.test_loss(X_batch, y_batch, data_size, verbose=verbose).item()
            likelihoods = model.compute_likelihood(X_batch, y_batch)
            sum_likelihood = sum(likelihoods).item()
            likelihoods_mc = model.compute_likelihood(X_batch, y_batch, mc=True)
            sum_likelihood_mc = sum(likelihoods_mc).item()
            total_likelihoods.append(likelihoods)
            total_sum_likelihood += sum_likelihood
            total_likelihoods_mc.append(likelihoods_mc)
            total_sum_likelihood_mc += sum_likelihood_mc

    preds = torch.cat(preds, dim=0)
    y_test = torch.cat(y_tests, dim=0)
    loss = total_loss / len(test_dataloader)
    total_likelihoods = torch.cat(total_likelihoods).mean(dim=0)
    min_likelihood = total_likelihoods.min().item()
    total_likelihoods = total_likelihoods.tolist()
    total_sum_likelihood = total_sum_likelihood / len(test_dataloader)
    total_likelihoods_mc = torch.cat(total_likelihoods_mc).mean(dim=0)
    min_likelihood_mc = total_likelihoods_mc.min().item()
    total_likelihoods_mc = total_likelihoods_mc.tolist()
    total_sum_likelihood_mc = total_sum_likelihood_mc / len(test_dataloader)

    y_pred = (preds >= threshold).double()
    assert y_pred.shape == y_test.shape, f"y_pred.shape={y_pred.shape} != y_test.shape={y_test.shape}"

    # for ECE
    confidences, predictions = preds.max(dim=1)

    metrics = {
        "f1": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "likelihood": [],
        "likelihood_mc": [],
        "loss": loss,
        "ece": []
    }


    print("------------------------------------")
    # Compute evaluation metrics (accuracy and F1-score) per label
    for k in range(K):
        metrics["likelihood"].append(likelihoods[k].item())
        metrics["likelihood_mc"].append(likelihoods_mc[k].item())
        acc = accuracy_score(y_test[:, k].flatten().int().cpu().numpy(), y_pred[:, k].int().flatten().cpu().numpy())
        precision = precision_score(y_test[:, k].cpu().numpy(), y_pred[:, k].cpu().numpy())
        recall = recall_score(y_test[:, k].cpu().numpy(), y_pred[:, k].cpu().numpy())
        f1 = f1_score(y_test[:, k].cpu().numpy(), y_pred[:, k].cpu().numpy())
        metrics["f1"].append(f1)
        metrics["accuracy"].append(acc)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        per_class_ece = []
        for cls in range(preds.shape[1]):
            mask = y_test[:, k] == cls
            if mask.sum() == 0:
                continue
            mask = mask.flatten()
            per_class_ece.append(torch.abs(confidences[mask] - (predictions[mask] == cls).double()).mean().item())
        metrics["ece"].append(per_class_ece)
        ece_avg = torch.tensor(per_class_ece).mean().item()
        if verbose:
            print(f"{prefix} : Label {k}: Loss = {loss:.2f}, SumLikelihood = {total_sum_likelihood:.2f}, MinLikelihood = {min_likelihood:.2f}, SumLikelihoodMC = {total_sum_likelihood_mc:.2f}, MinLikelihoodMC = {min_likelihood_mc:.2f}, ECE = {ece_avg:.2f}, Accuracy = {acc:.2f}, Precision = {precision:.2f}, Recall = {recall:.2f}, F1-score = {f1:.2f}")
    print("------------------------------------")

    return metrics

def compute_confusion_matrix(model, X_test, y_test, K, device, threshold = 0.5):
    # works for binary classification
    with torch.no_grad():
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        preds = model.predict(X_test)
    y_pred = (preds >= threshold).double()
    confusion_matrix = torch.zeros(K, K)
    for i in range(K):
        for j in range(K):
            confusion_matrix[i, j] = torch.sum((y_test[:, i] == 1) & (y_pred[:, j] == 1))
    return confusion_matrix

def modify_last_layer_lr(named_params, base_lr, lr_mult_w, lr_mult_b, base_wd, last_layer_wd = None, no_wd_last = False):
    if last_layer_wd is None:
        last_layer_wd = base_wd
    params = list()
    for name, param in named_params:
        if 'backbone' in name:
            if 'bias' in name:
                params += [{'params': param, 'lr': base_lr, 'weight_decay': 0}]
            else:
                params += [{'params': param, 'lr': base_lr, 'weight_decay': base_wd}]
        else:
            #FIXME: for now it does not work for neither of model type
            if 'bias' in name:
                params += [{'params': param, 'lr': base_lr * lr_mult_b, 'weight_decay': 0}]
            else:
                params += [{'params': param, 'lr': base_lr * lr_mult_w, 'weight_decay': 0 if no_wd_last else last_layer_wd}]
    return params

def create_optimizer_scheduler(args, model):
    # modify learning rate of last layer
    finetune_params = modify_last_layer_lr(model.named_parameters(), args.lr, args.lr_mult_w, args.lr_mult_b, args.wd, args.last_layer_wd, args.no_wd_last)
    # define optimizer
    common_params = {
        "lr": args.lr,
        "weight_decay": args.wd
    }
    optimizer_specific_params = {
        "adam": {
            "betas": (getattr(args, 'beta1', 0.9), 
                     getattr(args, 'beta2', 0.999))
        },
        "adamw": {
            "betas": (getattr(args, 'beta1', 0.9), 
                     getattr(args, 'beta2', 0.999))
        },
        "sgd": {
            "momentum": getattr(args, 'momentum', 0.9)
        }
    }
    optimizer_map = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD
    }
    optimizer_class = optimizer_map[args.optimizer.lower()]
    optimizer = optimizer_class(finetune_params, **common_params, **optimizer_specific_params[args.optimizer.lower()])
    
    # define learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                          step_size=args.lr_decay_in_epoch,
                                          gamma=args.gamma)
    return optimizer, scheduler

def wandb_unpack(file_path):
    # first line wandb API key
    # second line is wandb entity
    # third line wandb project
    return open(file_path).read().splitlines()

def wandb_init(cfg):
    key, entity, project = wandb_unpack(cfg.wandb_user_file)
    wandb.login(key=key)
    wandb.init(project=project, dir=os.path.join(cfg.path_to_results, cfg.name_exp, f"seed_{cfg.seed}"), entity=entity, config=cfg, name=cfg.wandb_run_name, tags=cfg.wandb_tags, mode="offline" if cfg.wandb_offline else "online")
    wandb.define_metric("iter")
    wandb.define_metric("running_loss", step_metric="iter")
    wandb.define_metric("train/epoch")
    wandb.define_metric("train/*", step_metric="train/epoch")
    wandb.define_metric("test/epoch")
    wandb.define_metric("test/*", step_metric="test/epoch")

def log(wandb_log, metrics = None, time = None, time_metric=None, specific_key = None, evaluated = False, prefix = "train", particular_metric_key = None, particular_metric_value = None, figure = None):
    if not wandb_log:
        return
    elif specific_key is not None:
        assert metrics is not None, "Metrics must be provided"
        if time_metric is not None:
            wandb.log({time_metric: time})
        wandb.log({specific_key: metrics[specific_key][-1]})
    elif particular_metric_key is not None:
        assert particular_metric_value is not None, "Particular metric value must be provided"
        wandb.log({particular_metric_key: particular_metric_value})
    elif evaluated:
        assert metrics is not None, "Metrics must be provided"
        assert time is not None, "Epoch must be provided"
        wandb.log({f"{prefix}/loss": metrics["loss"], f"{prefix}/epoch": time})
        for k in range(len(metrics["f1"])): # K labels
            wandb.log({f"{prefix}/{k}/likelihood": metrics["likelihood"][k]})
            wandb.log({f"{prefix}/{k}/likelihood_mc": metrics["likelihood_mc"][k]})
            wandb.log({f"{prefix}/{k}/f1": metrics["f1"][k]})
            wandb.log({f"{prefix}/{k}/accuracy": metrics["accuracy"][k]})
            wandb.log({f"{prefix}/{k}/precision": metrics["precision"][k]})
            wandb.log({f"{prefix}/{k}/recall": metrics["recall"][k]})
            for cls in range(len(metrics["ece"][k])):
                wandb.log({f"{prefix}/{k}/ece_{cls}": metrics["ece"][k][cls]})
    elif figure is not None:
        wandb.log({"plots/summary": wandb.Image(figure)})
    