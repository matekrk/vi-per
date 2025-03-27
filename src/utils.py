import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import jaccard_score, label_ranking_loss, label_ranking_average_precision_score
import torch
import torch.optim as optim
import wandb

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def default_dependency(M):
    if len(M) == 1:
        return [[1.0]]
    if len(M) == 4:
        p_o = [0.5]
        p_sq = [0.75, 0.25]
        p_t = [0.67, 0.67, 0.33, 0.33]
        p_st = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        return [p_o, p_sq, p_t, p_st]

def evaluate(model, test_dataloader, K, device, prefix = "", threshold = 0.5, verbose = False, plot_confusion=True):
    print("------------------------------------")
    model.eval()
    preds = []
    y_preds = []
    y_tests = []
    total_loss = 0.0
    total_likelihoods = []
    total_sum_likelihood = 0.0
    total_likelihoods_mc = []
    total_sum_likelihood_mc = 0.0
    data_size = len(test_dataloader.dataset)

    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred, batch_preds = model.predict(X_batch, threshold)
            preds.append(batch_preds)
            y_preds.append(y_pred)
            y_tests.append(y_batch)
            total_loss += model.test_loss(X_batch, y_batch, data_size, verbose=verbose).item()
            likelihoods = model.compute_negative_log_likelihood(X_batch, y_batch)
            sum_likelihood = sum(likelihoods).item()
            likelihoods_mc = model.compute_negative_log_likelihood(X_batch, y_batch, mc=True)
            sum_likelihood_mc = sum(likelihoods_mc).item()
            total_likelihoods.append(likelihoods)
            total_sum_likelihood += sum_likelihood
            total_likelihoods_mc.append(likelihoods_mc)
            total_sum_likelihood_mc += sum_likelihood_mc

    preds = torch.cat(preds, dim=0)
    y_pred = torch.cat(y_preds, dim=0)
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

    assert y_pred.shape == y_test.shape, f"y_pred.shape={y_pred.shape} != y_test.shape={y_test.shape}"

    metrics = {
        "f1": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "likelihood": [],
        "likelihood_mc": [],
        "loss": loss,
        "ece": [],
        "ece_plot": [],
    }

    metrics['subset_accuracy'] = torch.mean(torch.all(y_pred == y_test, axis=1).float()).item()
    hamming_per_attribute = torch.mean((y_pred != y_test).float(), dim=0)
    metrics['hamming_loss'] = torch.mean(hamming_per_attribute).item()
    metrics['hamming_loss_per_attribute'] = hamming_per_attribute.cpu().tolist()
    metrics['ranking_loss'] = label_ranking_loss(y_test.cpu().numpy(), y_pred.cpu().numpy())
    metrics['ranking_avg_precision'] = label_ranking_average_precision_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
    preds_for_ranking_loss = preds if preds.dim() == 2 else preds[:,:,1] # FIXME: hardcoded
    metrics['ranking_loss_probs'] = label_ranking_loss(y_test.cpu().numpy(), preds_for_ranking_loss.cpu().numpy())
    metrics['ranking_avg_precision_probs'] = label_ranking_average_precision_score(y_test.cpu().numpy(), preds_for_ranking_loss.cpu().numpy())
    intersection = torch.sum(y_pred * y_test, dim=0)
    union = torch.sum((y_pred + y_test) > 0, dim=0).float()
    jaccard_per_attribute = intersection / (union + 1e-8)
    metrics['jaccard_score_macro'] = torch.mean(jaccard_per_attribute).item()
    metrics['jaccard_score_per_attribute'] = jaccard_per_attribute.cpu().tolist()

    confidences = model.get_confidences(preds)

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

        y_test_k = y_test[:, k]
        y_pred_k = y_pred[:, k]
        preds_one_k = preds[:, k] if len(preds.shape) == 2 else preds[:, k, 1]
        confidences_k = confidences[:, k]
        ece, ece_plot = do_ece_single_attribute(y_test_k, y_pred_k, preds_one_k, confidences_k, num_bins=20)
        metrics["ece"].append(ece)
        metrics["ece_plot"].append(ece_plot)
        plt.close(ece_plot)

        if verbose:
            print(f"{prefix} : Label {k}: Loss = {loss:.2f}, SumLikelihood = {total_sum_likelihood:.2f}, MinLikelihood = {min_likelihood:.2f}, SumLikelihoodMC = {total_sum_likelihood_mc:.2f}, MinLikelihoodMC = {min_likelihood_mc:.2f}, ECE = {ece:.2f}, Accuracy = {acc:.2f}, Precision = {precision:.2f}, Recall = {recall:.2f}, F1-score = {f1:.2f}")
    
    metrics["accuracy_macro"] = torch.mean(torch.tensor(metrics["accuracy"])).item()
    metrics['precision_macro'] = torch.mean(torch.tensor(metrics["precision"])).item()
    metrics['recall_macro'] = torch.mean(torch.tensor(metrics["recall"])).item()
    metrics['f1_macro'] = torch.mean(torch.tensor(metrics["f1"])).item()

    tp = torch.sum((y_pred == 1) & (y_test == 1), dim=0)
    fp = torch.sum((y_pred == 1) & (y_test == 0), dim=0)
    fn = torch.sum((y_pred == 0) & (y_test == 1), dim=0)
    tn = torch.sum((y_pred == 0) & (y_test == 0), dim=0)
    metrics["accuracy_micro"] = (torch.sum(tp) + torch.sum(tn)) / (torch.sum(tp) + torch.sum(fp) + torch.sum(fn) + torch.sum(tn)).item()
    metrics['precision_micro'] = (torch.sum(tp) / (torch.sum(tp) + torch.sum(fp) + 1e-8)).item()
    metrics['recall_micro'] = (torch.sum(tp) / (torch.sum(tp) + torch.sum(fn) + 1e-8)).item()
    metrics['f1_micro'] = 2 * metrics['precision_micro'] * metrics['recall_micro'] / (metrics['precision_micro'] + metrics['recall_micro'] + 1e-8)

    tree_plot = do_tree_predictions(default_dependency(["o", "s", "^", "*"]), y_pred)
    metrics["tree_predictions_plot"] = tree_plot
    plt.close(tree_plot)
    if plot_confusion:
        conf_plot = do_confusion_row_matrix(y_test, y_pred)
        metrics["confusion_matrix_plot"] = conf_plot
        plt.close(conf_plot)
    corr_plot = do_correlation_matrix(y_test, y_pred)
    metrics["correlation_plot"] = corr_plot
    plt.close(corr_plot)
    deppred_plot = do_dependent_predictions(y_pred)
    metrics["dependent_predictions_plot"] = deppred_plot
    plt.close(deppred_plot)

    print("------------------------------------")

    return metrics

def do_correlation_matrix(y_test, y_pred):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    with torch.no_grad():
        correlation_matrix = np.zeros((y_test.shape[1], 2, 2))
        for i in range(y_test.shape[1]):
            heatmap_data, _, _ = np.histogram2d(y_test[:, i].cpu().numpy(), y_pred[:, i].cpu().numpy(), bins=2)
            correlation_matrix[i] = heatmap_data
        fig, axes = plt.subplots(nrows=1, ncols=y_test.shape[1], figsize=(20, 5))
        for i in range(y_test.shape[1]):
            sns.heatmap(correlation_matrix[i], annot=True, fmt=".2f", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1], ax=axes[i])
            axes[i].set_xlabel("y_test")
            axes[i].set_ylabel("y_pred")
            axes[i].set_title(f"Attribute {i+1}")
    plt.tight_layout()
    plt.show()
    return fig

def do_confusion_row_matrix(y_test, y_pred):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    n_attr = y_test.shape[1]
    with torch.no_grad():
        conf_mat = torch.zeros((2**n_attr, 2**n_attr)) # cannot allocate FIXME
        for i in range(2**n_attr):
            for j in range(2**n_attr):
                true_row = torch.tensor([int(x) for x in f"{i:0{n_attr}b}"], device=y_test.device)
                pred_row = torch.tensor([int(x) for x in f"{j:0{n_attr}b}"], device=y_test.device)
                denom = torch.sum(torch.all(y_pred == pred_row, dim=1)).item()
                if denom:
                    conf_mat[i, j] = torch.sum(torch.all(y_test == true_row, dim=1) & torch.all(y_pred == pred_row, dim=1)).item() / denom
    img = ax.matshow(conf_mat, cmap='hot', interpolation='nearest')
    ax.set_xticks(range(2**n_attr))
    ax.set_yticks(range(2**n_attr))
    ax.set_yticklabels([str([int(x) for x in f"{i:0{n_attr}b}"]) for i in range(2**n_attr)])
    ax.set_title("Accordance matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.colorbar(img, ax=ax)
    return fig

def do_tree_predictions(true_dependencies, y_pred):
    fig, ax = plt.subplots(2, 2)
    w, h = y_pred.shape[1], 2**(y_pred.shape[1])
    with torch.no_grad():
        mat_true = torch.zeros((len(true_dependencies), 2**len(true_dependencies)))
        for i, dependencies in enumerate(true_dependencies):
            k = len(dependencies)
            repeat_count = 2**(len(true_dependencies) - i - 1)
            for j, value in enumerate(dependencies):
                for r in range(repeat_count):
                    mat_true[i, j * repeat_count * 2 + r] = 1 - value
                    mat_true[i, j * repeat_count * 2 + repeat_count + r] = value
        img = ax[0, 0].imshow(mat_true.numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar(img, ax=ax[1, 0])
        ax[0, 0].set_title("True dependencies")
                
        unique_rows = [torch.tensor([int(x) for x in f"{i:0{w}b}"], device=y_pred.device) for i in range(2**w)]
        counts = [torch.sum(torch.all(y_pred == row, dim=1)).item() for row in unique_rows]
        row_counts = {tuple(row.tolist()): count for row, count in zip(unique_rows, counts)}

        mat_pred = torch.zeros((len(true_dependencies), 2**len(true_dependencies)))
        probs_1 = []
        for i in range(len(true_dependencies)):
            probs_1_cur = []
            for j in range(2**i):
                if i > 0:
                    count_1 = sum(count for row, count in row_counts.items() if row[:i] == tuple([int(x) for x in f"{j:0{i}b}"]) and row[i] == 1)
                    count_0 = sum(count for row, count in row_counts.items() if row[:i] == tuple([int(x) for x in f"{j:0{i}b}"]) and row[i] == 0)
                else:
                    count_1 = sum(count for row, count in row_counts.items() if row[i] == 1)
                    count_0 = sum(count for row, count in row_counts.items() if row[i] == 0)
                prob_1 = count_1 / (count_0 + count_1 + 1e-8)
                probs_1_cur.append(prob_1)
                repeat_count = 2**(len(true_dependencies) - i - 1)
                for r in range(repeat_count):
                    mat_pred[i, j * repeat_count * 2 + r] = 1 - prob_1
                    mat_pred[i, j * repeat_count * 2 + repeat_count + r] = prob_1
            probs_1.append(probs_1_cur)
    img = ax[0, 1].imshow(mat_pred.numpy(), cmap='hot', interpolation='nearest')
    plt.colorbar(img, ax=ax[1, 1])
    ax[0, 1].set_title("Predicted dependencies")

    text = "\n".join([f"{i}: {dependencies}" for i, dependencies in enumerate(true_dependencies)])
    ax[1, 0].text(0.5, 0.5, text, ha='center', va='center', fontsize=12, wrap=True)
    ax[1, 0].set_title("True Dependencies")
    ax[1, 0].axis('off')

    text = "\n".join([f"{i}: {[round(dep, 4) for dep in dependencies]}" for i, dependencies in enumerate(probs_1)])
    ax[1, 1].text(0.5, 0.5, text, ha='center', va='center', fontsize=12, wrap=True)
    ax[1, 1].set_title("Predicted Dependencies")
    ax[1, 1].axis('off')
    # plt.tight_layout()
    return fig

def do_dependent_predictions(y_pred):
    fig, ax = plt.subplots(1, 2)
    with torch.no_grad():
        mat_positive = torch.zeros((y_pred.shape[1], y_pred.shape[1]))
        for i in range(y_pred.shape[1]):
            for j in range(y_pred.shape[1]):
                mat_positive[i, j] = torch.sum(y_pred[:, i] * y_pred[:, j])
        mat_negative = torch.zeros((y_pred.shape[1], y_pred.shape[1]))
        for i in range(y_pred.shape[1]):
            for j in range(y_pred.shape[1]):
                mat_negative[i, j] = torch.sum((1 - y_pred[:, i]) * y_pred[:, j])
        img = ax[0].imshow(mat_positive.numpy(), cmap='hot', interpolation='nearest')
        for i in range(y_pred.shape[1]):
            for j in range(y_pred.shape[1]):
                ax[0].text(j, i, mat_positive[i, j].item(), ha="center", va="center", color="blue", fontsize=6)
        ax[0].set_title("Positive dependencies")
        ax[0].set_ylabel("Predicted positive")
        ax[0].set_xlabel("Occurence (positive) given positive")
        plt.colorbar(img, ax=ax[0])
        img = ax[1].imshow(mat_negative.numpy(), cmap='cool', interpolation='nearest')
        for i in range(y_pred.shape[1]):
            for j in range(y_pred.shape[1]):
                ax[1].text(j, i, mat_negative[i, j].item(), ha="center", va="center", color="w", fontsize=6)
        ax[1].set_title("Negative dependencies")
        ax[1].set_xlabel("Occurence (positive) given negative")
        plt.colorbar(img, ax=ax[1])
    return fig

def do_ece_single_attribute(y_test, y_pred, preds_one, confidences, num_bins=20):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    with torch.no_grad():
        bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=y_test.device)
        bin_indices = torch.bucketize(confidences.contiguous(), bin_boundaries) - 1
        ece = 0.0
        for bin_index in range(num_bins):
            in_bin = (bin_indices == bin_index)
            if in_bin.any():
                bin_size = in_bin.sum().item()
                bin_accuracy = (y_test[in_bin] == y_pred[in_bin]).float().mean().item()
                bin_confidence = confidences[in_bin].mean().item()
                ax[0].bar(bin_index/num_bins, bin_accuracy, width=1/num_bins, color='blue', linewidth=4, alpha=0.5)
                this_ece = abs(bin_accuracy - bin_confidence) * (bin_size / len(y_test))
                ax[0].text(bin_index / num_bins, max(0.1, min(0.8, bin_accuracy - 0.05)), f"ECE={this_ece:.4f}\nN={bin_size}", ha='center', va='center', color='black', rotation=90)
                ece += this_ece
                ax[0].set_xlabel("Confidence")
                ax[0].set_ylabel("Accuracy")
        ax[0].set_title(f"ECE plot: {ece:.4f}")

        bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=y_test.device)
        bin_indices = torch.bucketize(preds_one.contiguous(), bin_boundaries) - 1
        for bin_index in range(num_bins):
            in_bin = (bin_indices == bin_index)
            if in_bin.any():
                bin_size = in_bin.sum().item()
                bin_accuracy = (y_test[in_bin] == y_pred[in_bin]).float().mean().item()
                ax[1].bar(bin_index/num_bins, bin_accuracy, width=1/num_bins, color='blue', linewidth=4, alpha=0.5)
                ax[0].text(bin_index / num_bins, max(0.1, min(0.8, bin_accuracy - 0.05)), f"MeanAcc={bin_accuracy:.4f}\nN={bin_size}", ha='center', va='center', color='black', rotation=90)
                ax[1].set_xlabel("Predictions")
                ax[1].set_ylabel("Accuracy")
        ax[1].set_title(f"ECE plot: Mean Acc vs p for 1")
    return ece, fig
    

def compute_confusion_matrix(model, X_test, y_test, K, device, threshold = 0.5):
    with torch.no_grad():
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        y_pred, preds = model.predict(X_test, threshold)
        confusion_matrix = torch.zeros(K, K)
        for i in range(K):
            for j in range(K):
                confusion_matrix[i, j] = torch.sum((y_test[:, i] == 1) & (y_pred[:, j] == 1))
    return confusion_matrix

def modify_last_layer_lr(named_params, backbone_freeze, base_lr, lr_mult_w, lr_mult_b, base_wd, last_layer_wd = None, no_wd_last = False):
    if last_layer_wd is None:
        last_layer_wd = base_wd
    params = list()
    print("Model architecture...")
    for name, param in named_params:
        print(name, f"gradient: {param.requires_grad}", f"shape: {param.shape}")
        if 'backbone' in name:
            if backbone_freeze:
                param.requires_grad = False
            else:
                if 'bias' in name:
                    params += [{'params': param, 'lr': base_lr, 'weight_decay': 0}]
                else:
                    params += [{'params': param, 'lr': base_lr, 'weight_decay': base_wd}]
        else:
            if 'bias' in name:
                params += [{'params': param, 'lr': base_lr * lr_mult_b, 'weight_decay': 0}]
            else:
                params += [{'params': param, 'lr': base_lr * lr_mult_w, 'weight_decay': 0 if no_wd_last else last_layer_wd}]
    return params

def create_optimizer_scheduler(args, model):
    finetune_params = modify_last_layer_lr(model.named_parameters(), args.backbone_freeze, args.lr, args.lr_mult_w, args.lr_mult_b, args.wd, args.last_layer_wd, args.no_wd_last)
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

def log(wandb_log, metrics = None, time = None, time_metric=None, specific_key = None, evaluated = False, prefix = "train", particular_metric_key = None, particular_metric_value = None, figure_img = None, figure_summary = None):
    if not wandb_log:
        return
    elif figure_img is not None:
        assert particular_metric_key is not None, "Particular metric key must be provided"
        wandb.log({f"img/{particular_metric_key}": wandb.Image(figure_img)})
    elif figure_summary is not None:
        wandb.log({"plots/summary": wandb.Image(figure_summary)})
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
        wandb.log({f"{prefix}/subset_accuracy": metrics["subset_accuracy"], f"{prefix}/epoch": time})
        wandb.log({f"{prefix}/hamming_loss": metrics["hamming_loss"], f"{prefix}/epoch": time})
        wandb.log({f"{prefix}/ranking_loss": metrics["ranking_loss"], f"{prefix}/epoch": time})
        wandb.log({f"{prefix}/ranking_avg_precision": metrics["ranking_avg_precision"], f"{prefix}/epoch": time})
        wandb.log({f"{prefix}/ranking_loss_probs": metrics["ranking_loss_probs"], f"{prefix}/epoch": time})
        wandb.log({f"{prefix}/ranking_avg_precision_probs": metrics["ranking_avg_precision_probs"], f"{prefix}/epoch": time})
        # wandb.log({f"{prefix}/jaccard_score_micro": metrics["jaccard_score_micro"], f"{prefix}/epoch": time})
        wandb.log({f"{prefix}/jaccard_score_macro": metrics["jaccard_score_macro"], f"{prefix}/epoch": time})
        wandb.log({f"{prefix}/precision_micro": metrics["precision_micro"], f"{prefix}/epoch": time})
        wandb.log({f"{prefix}/recall_micro": metrics["recall_micro"], f"{prefix}/epoch": time})
        wandb.log({f"{prefix}/f1_micro": metrics["f1_micro"], f"{prefix}/epoch": time})
        wandb.log({f"{prefix}/precision_macro": metrics["precision_macro"], f"{prefix}/epoch": time})
        wandb.log({f"{prefix}/recall_macro": metrics["recall_macro"], f"{prefix}/epoch": time})
        wandb.log({f"{prefix}/f1_macro": metrics["f1_macro"], f"{prefix}/epoch": time})
        if prefix == "test":
            if "tree_predictions_plot" in metrics:
                wandb.log({f"{prefix}/tree_predictions": wandb.Image(metrics["tree_predictions_plot"]), f"{prefix}/epoch": time})
            if "confusion_matrix_plot" in metrics:
                wandb.log({f"{prefix}/confusion_matrix": wandb.Image(metrics["confusion_matrix_plot"]), f"{prefix}/epoch": time})
            if "correlation_plot" in metrics:
                wandb.log({f"{prefix}/correlation": wandb.Image(metrics["correlation_plot"]), f"{prefix}/epoch": time})
            if "dependent_predictions_plot" in metrics:
                wandb.log({f"{prefix}/dependent_predictions": wandb.Image(metrics["dependent_predictions_plot"]), f"{prefix}/epoch": time})
        for k in range(len(metrics["f1"])): # K attributes
            wandb.log({f"{prefix}/{k}/likelihood": metrics["likelihood"][k]})
            wandb.log({f"{prefix}/{k}/likelihood_mc": metrics["likelihood_mc"][k]})
            wandb.log({f"{prefix}/{k}/f1": metrics["f1"][k]})
            wandb.log({f"{prefix}/{k}/accuracy": metrics["accuracy"][k]})
            wandb.log({f"{prefix}/{k}/precision": metrics["precision"][k]})
            wandb.log({f"{prefix}/{k}/recall": metrics["recall"][k]})
            wandb.log({f"{prefix}/{k}/ece": metrics["ece"][k]})
            wandb.log({f"{prefix}/{k}/ece_plot": wandb.Image(metrics["ece_plot"][k])})
            wandb.log({f"{prefix}/{k}/hamming_loss": metrics["hamming_loss_per_attribute"][k], f"{prefix}/epoch": time})
            wandb.log({f"{prefix}/{k}/jaccard_score": metrics["jaccard_score_per_attribute"][k], f"{prefix}/epoch": time})
    
def empty_metrics():
    metrics = {
        "running_loss": [],
        "grad_norm": [],
        "param_norm": [],
        "train/running_loss_mean": [],
        "train_loss": [],
        "train_likelihood": [],
        "train_likelihood_mc": [],
        "train/likelihood_mean": [],
        "train/likelihood_min": [],
        "train/likelihood_mean_mc": [],
        "train/likelihood_min_mc": [],
        "train_f1": [],
        "train/f1_macro": [],
        "train_accuracy": [],
        "train_accuracy_macro": [],
        "train_accuracy_micro": [],
        "train_precision": [],
        "train_recall": [],
        "train_ece": [],
        "train/ece_mean": [],
        "train_subset_accuracy": [],
        "train_hamming_loss": [],
        "train_hamming_loss_per_attribute": [],
        "train_ranking_loss": [],
        "train_ranking_loss_probs": [],
        "train_ranking_avg_precision": [],
        "train_ranking_avg_precision_probs": [],
        "train_jaccard_score_per_attribute": [],
        "train_jaccard_score_macro": [],
        "train_precision_micro": [],
        "train_precision_macro": [],
        "train_recall_micro": [],
        "train_recall_macro": [],
        "train_f1_micro": [],
        "train_f1_macro": [],
        "test_loss": [],
        "test_likelihood": [],
        "test_likelihood_mc": [],
        "test/likelihood_mean": [],
        "test/likelihood_min": [],
        "test/likelihood_mean_mc": [],
        "test/likelihood_min_mc": [],
        "test_f1": [],
        "test/f1_macro": [],
        "test_accuracy": [],
        "test_accuracy_macro": [],
        "test_accuracy_micro": [],
        "test_precision": [],
        "test_recall": [],
        "test_ece": [],
        "test/ece_mean": [],
        "test_subset_accuracy": [],
        "test_hamming_loss": [],
        "test_hamming_loss_per_attribute": [],
        "test_ranking_loss": [],
        "test_ranking_loss_probs": [],
        "test_ranking_avg_precision": [],
        "test_ranking_avg_precision_probs": [],
        "test_jaccard_score_per_attribute": [],
        "test_jaccard_score_macro": [],
        "test_precision_micro": [],
        "test_precision_macro": [],
        "test_recall_micro": [],
        "test_recall_macro": [],
        "test_f1_micro": [],
        "test_f1_macro": [],
        "ood_loss": [],
        "ood_likelihood": [],
        "ood_likelihood_mc": [],
        "ood/likelihood_mean": [],
        "ood/likelihood_min": [],
        "ood/likelihood_mean_mc": [],
        "ood/likelihood_min_mc": [],
        "ood_f1": [],
        "ood/f1_macro": [],
        "ood_accuracy": [],
        "ood_accuracy_macro": [],
        "ood_accuracy_micro": [],
        "ood_precision": [],
        "ood_recall": [],
        "ood_ece": [],
        "ood/ece_mean": [],
        "ood_subset_accuracy": [],
        "ood_hamming_loss": [],
        "ood_hamming_loss_per_attribute": [],
        "ood_ranking_loss": [],
        "ood_ranking_loss_probs": [],
        "ood_ranking_avg_precision": [],
        "ood_ranking_avg_precision_probs": [],
        "ood_jaccard_score_per_attribute": [],
        "ood_jaccard_score_macro": [],
        "ood_precision_micro": [],
        "ood_precision_macro": [],
        "ood_recall_micro": [],
        "ood_recall_macro": [],
        "ood_f1_micro": [],
        "ood_f1_macro": [],
        "vi/prior_mu": [],
        "vi/u_sig": [],
        "vi/sig": []
    }
    return metrics

def default_config():
    cfg = \
        {
            "backbone_type": "ConvNet",
            "backbone_pretrained": False,
            "backbone_freeze": False,
            "p": 64,
            "K": 6,
            "intercept": False,
            "model_type": None, # cannot be None, fill it
            "beta": 0.0,
            "pred_threshold": 0.5,
            "f1_thres": 0.55,
            "method": None,
            "l_max": None,
            "n_samples": None,
            "prior_scale": None,
            "posterior_mean_init_scale": 1.0,
            "posterior_var_init_add": 0.0,
            "VBLL_PATH": None,
            "VBLL_TYPE": None,
            "VBLL_SOFTMAX_BOUND": None,
            "VBLL_RETURN_EMPIRICAL": None,
            "VBLL_RETURN_OOD": None,
            "VBLL_PRIOR_SCALE": None,
            "VBLL_WIDTH_SCALE": None,
            "VBLL_WUMEAN_SCALE": None,
            "VBLL_WULOGDIAG_SCALE": None,
            "VBLL_WUOFFDIAG_SCALE": None,
            "VBLL_NOISE_LABEL": None,
            "VBLL_PARAMETRIZATION": None,
            "VBLL_WISHART_SCALE": None,
            "optimizer": "adam",
            "lr": 0.001,
            "lr_mult_w": 1.0,
            "lr_mult_b": 1.0,
            "gamma": 1.0,
            "lr_decay_in_epoch": 100,
            "wd": 0.0,
            "last_layer_wd": None,
            "no_wd_last": False,
            "early_stopping": True,
            "n_epochs": 500,
            "evaluate_freq": 10,
            "verbose_freq": 50,
            "save_best": True,
            "n_test_data_pred": 5,
            "path_to_results": "./results",
            "name_exp": "shapes",
            "batch_size": 32,
            "data_type": "shapes",
            "data_type_ood": "shapes_ood",
            "data_channels": 3,
            "data_size": 64,
            "N": 4096,
            "N_ood": 3072,
            "N_test_ratio": 0.2,
            "coloured_background": False,
            "coloured_figues": False,
            "no_overlap": False,
            "bias_classes": [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            "bias_classes_ood": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "simplicity": 1,
            "simplicity_ood": 6,
            "main_dir": "/shared/sets/datasets/vision/artificial_shapes",
            # "shapes": ['disk', 'square', 'triangle', 'star', 'hexagon', 'pentagon'],
            "wandb": False,
            "wandb_user_file": "wandb.txt",
            "wandb_run_name": "shapes",
            "wandb_tags": ["shapes"],
            "wandb_offline": False,
            "seed": 8
        }
    return cfg
