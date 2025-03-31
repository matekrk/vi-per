import os
from matplotlib import pyplot as plt

def plot_summary(cfg, metrics, epoch, epochs_eval, confusion_matrix):
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

    if confusion_matrix is not None:
        ax = plt.subplot(5, 1, 5)
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
    return fig
