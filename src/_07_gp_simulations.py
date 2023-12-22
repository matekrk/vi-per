import math

import torch
import numpy as np
import matplotlib.pyplot as plt
import dsdl
import gpytorch
import tqdm

from torcheval.metrics import BinaryAUROC
from joblib import Parallel, delayed


from _97_gpytorch import LogisticGPVI, LogitLikelihoodMC, PGLikelihood, LogitLikelihood


def generate_data(n, seed=1):
    torch.manual_seed(seed)
    func = lambda x: - 4.5 * torch.sin(math.pi / 2 * x)
    train_x = torch.cat((torch.linspace(0, 2.5, int(n/2)), torch.linspace(3.5, 5, int(n/2))))

    train_f = func(train_x) + torch.randn(train_x.shape[0]) * 1.0 # * math.sqrt(0.4)
    train_x = train_x.reshape(-1, 1)
    train_p = torch.sigmoid(train_f)
    train_y = torch.bernoulli(train_p)

    test_x = torch.linspace(0, 5, n) 
    test_f = func(test_x) + torch.randn(test_x.shape[0]) * 1.0  # * math.sqrt(0.4)
    test_p = torch.sigmoid((test_f))
    test_y = torch.bernoulli(test_p)

    xs = torch.linspace(0, 5, 100).reshape(-1, 1)
    true_f = func(xs)

    # x = torch.linspace(0, 5, n)
    # func = lambda x: torch.sin(x * (0.5 * math.pi)) * 4
    # f = func(x) + math.sqrt(0.4) * torch.randn(n)
    # x = x.reshape(-1, 1)
    # p = torch.sigmoid(f)
    # y = torch.bernoulli(p)

    # # split train and test
    # idx = torch.randperm(n)
    # n_train = int(n * 0.8)

    # train_x = x[idx[:n_train]]
    # train_y = y[idx[:n_train]]

    # test_x = x[idx[n_train:]]
    # test_y = y[idx[n_train:]]
    # test_p = p[idx[n_train:]]
    # test_f = f[idx[n_train:]]
    # 
    # test_x = test_x.reshape(-1, 1)
    # true_f = func(test_x)

    return train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f


def analyze_simulation(seed, train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f,
        n_iter=200, n_inducing=50, thresh=1e-7, lr=0.05, verbose=False, use_loader=False, batches=20):

    torch.manual_seed(seed)
    print(f"Run: {seed}")
        
    f0 = LogisticGPVI(train_y, train_x, n_inducing=n_inducing, n_iter=n_iter, thresh=thresh, verbose=verbose, 
                            use_loader=use_loader, batches=batches, seed=seed, lr=lr)
    f0.fit()

    f1 = LogisticGPVI(train_y, train_x, likelihood=LogitLikelihoodMC(), n_inducing=n_inducing, n_iter=n_iter, thresh=thresh,
                            verbose=verbose, use_loader=use_loader, batches=batches, seed=seed, lr=lr)
    f1.fit()

    f2 = LogisticGPVI(train_y, train_x, likelihood=PGLikelihood(), n_inducing=n_inducing, n_iter=n_iter, thresh=thresh, 
                            verbose=verbose, use_loader=use_loader, batches=batches, seed=seed, lr=lr)
    f2.fit()

    return torch.tensor([
        evaluate_method_simulation(f0, test_x, test_y, test_p, test_f, xs, true_f),
        evaluate_method_simulation(f1, test_x, test_y, test_p, test_f, xs, true_f),
        evaluate_method_simulation(f2, test_x, test_y, test_p, test_f, xs, true_f),
    ])



def evaluate_method_simulation(func, test_x, test_y, test_p, test_f, xs, true_f):
    pred_y = func.predict(test_x)

    auc = BinaryAUROC()
    auc.update(test_y, pred_y)
    auc = auc.compute().item()

    lower, upper = func.credible_intervals(test_x)
    ci_width = (upper - lower).mean().item()

    n = test_y.size()[0]
    coverage_f = torch.sum( (test_f > lower) & (test_f < upper) ) / n

    samp = func.model(test_x).sample(torch.Size([1000]))
    p0 = torch.sigmoid(samp)

    lower_p = torch.quantile(p0, 0.025, dim=0)
    upper_p = torch.quantile(p0, 0.975, dim=0)
    
    coverage_p = torch.sum( (test_p > lower_p) & (test_p < upper_p) ) / n

    f_pred = func.model(xs).mean
    mse = ((true_f.reshape(-1) - f_pred) ** 2).mean().item()
    
    return func.runtime, auc, mse, ci_width, \
            func.neg_log_likelihood().item(), func.neg_log_likelihood(test_x, test_y).item(), \
            func.log_marginal().item(),             func.log_marginal(test_x, test_y).item(), \
            func.ELB0_MC().item(),                       func.ELB0_MC(test_x, test_y).item(), \
            coverage_f.item(), coverage_p.item()

CPUS = -2
RUNS = 100
n = 50

def run_exp(seed):
    train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f = generate_data(50, seed=seed)
    return analyze_simulation(seed, train_x, train_y, test_x, test_y, test_p, test_f, xs, true_f,
     n_iter=1200, n_inducing=50)

res = Parallel(n_jobs=CPUS)(delayed(run_exp)(i) for i in range(1, RUNS+1))
res = torch.stack(res)
res = torch.transpose(res, 0, 1)           
res
torch.save(res, "../results/gp.pt")

run_exp(1)