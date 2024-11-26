from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.optim as optim

def evaluate(model, X_test, y_test, data_size, K, prefix = ""):
    # Get predictions
    with torch.no_grad():
        preds = model.predict(X_test)
        loss = model.train_loss(X_test, y_test, data_size, verbose=False)

    # Binarize predictions using a threshold (e.g., 0.5)
    threshold = 0.5
    y_pred = (preds >= threshold).double()
    assert y_pred.shape == y_test.shape, f"y_pred.shape={y_pred.shape} != y_test.shape={y_test.shape}"

    f1s = []

    # Compute evaluation metrics (accuracy and F1-score) per label
    for k in range(K):
        acc = accuracy_score(y_test[:, k].flatten().int().numpy(), y_pred[:, k].int().flatten().numpy())
        precision = precision_score(y_test[:, k].numpy(), y_pred[:, k].numpy())
        recall = recall_score(y_test[:, k].numpy(), y_pred[:, k].numpy())
        f1 = f1_score(y_test[:, k].numpy(), y_pred[:, k].numpy())
        f1s.append(f1)
        print(f"{prefix} : Label {k}: Loss = {loss:.2f}, Accuracy = {acc:.2f}, Precision = {precision:.2f}, Recall = {recall:.2f}, F1-score = {f1:.2f}")

    return f1s

def create_optimizer_scheduler(args, model):
    # modify learning rate of last layer
    finetune_params = model.parameters()
    # define optimizer
    common_params = {
        "lr": args.lr,
        "weight_decay": args.weight_decay
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
    optimizer_class = optimizer_map[args.opti.lower()]
    optimizer = optimizer_class(finetune_params, **common_params, **optimizer_specific_params[args.opti.lower()])
    
    # define learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                          step_size=args.lr_decay_in_epoch,
                                          gamma=args.gamma)
    return optimizer, scheduler