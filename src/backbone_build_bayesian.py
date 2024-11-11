from typing import Union
from collections.abc import Callable
from dataclasses import dataclass
from numpy import pi
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal

from _02_method import ELBO_MC, ELBO_MC_mvn, ELBO_TB, ELBO_TB_mvn, ELBO_MC_squeezed, ELBO_TB_squeezed, ELBO_MC_mvn_squeezed, ELBO_TB_mvn_squeezed
# m, s, mu, sig. mean and std dev of variational distribution and mu sig of prior

@dataclass
class BayesianLastLayerReturn():
    predictive: Union[Normal]
    train_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    val_loss_fn: Callable[[torch.Tensor], torch.Tensor]

class BayesianLastLayerSqueezed(nn.Module):
    def __init__(self, basemodel_output, num_class, len_dataset, mu, sig, Sig, s_init, intercept):
        super(BayesianLastLayerSqueezed, self).__init__()
        self.len_dataset = len_dataset
        assert len_dataset is not None
        self.intercept = intercept

        final_basemodel_output = basemodel_output + int(intercept)

        self.p = final_basemodel_output * num_class
        self.num_class = num_class

        # Prior
        self.mu = torch.zeros(final_basemodel_output, num_class, dtype=torch.double) if mu is None else mu
        self.sig = torch.ones(final_basemodel_output, num_class, dtype=torch.double) if sig is None else sig
        self.Sig = torch.eye(final_basemodel_output * num_class, dtype=torch.double) if Sig is None else Sig

        # Posterior - optimize W_dist (m) and u 
        self.W_dist = Normal # MultivariateNormal
        self.W_mean = nn.Parameter(torch.randn(final_basemodel_output, num_class, dtype=torch.double))

        if s_init is None:
            self.u_init = torch.ones(final_basemodel_output * num_class, dtype=torch.double) * -1 # , dtype=torch.double # torch.tensor([-1.] * self.W_mean.size(-1), dtype=torch.double)
            self.s_init = torch.exp(self.u_init)
        else:
            self.s_init = s_init
            self.u_init = torch.log(s_init)
        self.u = self.u_init.clone()
        self.u.requires_grad = True

    @torch.no_grad
    def init_covar(self, method):
        if method in ["tb_mvn", "mc_mvn"]:
            self.u = torch.ones(int(self.p * (1 + self.p) / 2.0), device=self.W_mean.device) # , dtype=torch.double
            self.u = self.u * 1/self.p
        self.u_init = self.u.clone()
        self.u.requires_grad = True
        self.update_covar(method)

    @torch.no_grad
    def update_covar(self, method):
        if method == "tb" or method == "mc":
            self.s = torch.exp(self.u)
            self.S = torch.diag(self.s)
        elif method == "tb_mvn" or method == "mc_mvn":
            L = torch.zeros_like(self.Sig)
            L[torch.tril_indices(self.p, self.p, 0).tolist()] = self.u
            self.S = L.t() @ L
            self.s = torch.sqrt(torch.diag(self.S))
        else:
            Exception("bound!")

    def W(self):
        return self.W_dist(self.W_mean.view(-1), self.S)
    
    def multi_distr_tensor(self, d, inp):
        if isinstance(d, Normal):
            y = (d.scale ** 2).unsqueeze(-1)
        elif isinstance(d, MultivariateNormal):
            y = d.covariance_matrix
        new_cov = (d.scale ** 2).unsqueeze(-1) * (inp.unsqueeze(-3) ** 2).sum(-2)
        return Normal(d.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))
    
    # FIXME
    def logit_predictive(self, x):
        if self.intercept:
            x = torch.cat((torch.ones(x.size()[0], 1), x), 1)
        return (self.multi_distr_tensor(self.W(), x[..., None])).squeeze(-1)
    
    #FIXME
    def predictive(self, x, n_samples = 10):
        softmax_samples = F.sigmoid(self.logit_predictive(x).rsample(sample_shape=torch.Size([n_samples])), dim=-1)
        return torch.clip(torch.mean(softmax_samples, dim=0),min=0.,max=1.)

    def forward(self, x, method):
        # empirical
        #n_samples = 10
        #M = self.sample(method, n_samples)
        #probs = torch.mean(torch.sigmoid(x @ M), 0)
        # analytical just take mean. Remember E[SX] != S[EX] which is torch.sigmoid(x @ self.W_mean)
        probs = torch.sigmoid(x.to(self.W_mean.dtype) @ (self.W_mean/(torch.sqrt(1 + (pi/8.0) * (self.s**2).reshape_as(self.W_mean)))))
        probs = torch.clip(probs, min=1e-6,max=1.-(1e-6))
        returns = torch.distributions.Categorical(probs = probs)
        return BayesianLastLayerReturn(
            returns, # torch.distributions.Categorical(probs = self.predictive(x)), # or through sample
            self._get_train_loss_fn(x, method),
            self._get_train_loss_fn(x, method) # self._get_val_loss_fn(x) FIXME: val loss
        )
    
    def _get_train_loss_fn(self, x, method):

        def loss_fn(y, n_samples = None, l_max = None):
            return self.bound(x, y, method, n_samples, l_max)

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            return -torch.mean(torch.log(self.predictive(x)[torch.arange(x.shape[0]), y]))
        return loss_fn
    
    def bound(self, x, y, method, n_samples = None, l_max = None):
        data_size = self.len_dataset
        x = x.to(torch.float64)
        y = y.to(x.dtype)
        assert (method in ["tb", "tb_mvn"] and l_max is not None) or (method in ["mc", "mc_mvn"] and n_samples is not None)
        if method == "tb":
            xx = x ** 2
            value = sum(ELBO_TB_squeezed(w, u, y_single, x, mu, sig, l_max, xx, data_size) for w, u, y_single, mu, sig in zip(self.W_mean.t(), self.u.view(-1, self.num_class).t(), y.t(), self.mu.t(), self.sig.t()))
        elif method == "tb_mvn":
            value = sum(ELBO_TB_mvn_squeezed(w, u, y_single, x, mu, Sig, l_max, data_size) for w, u, y_single, mu, Sig in zip(self.W_mean.t(), self.u.view(-1, self.num_class).t(), y.t(), self.mu.t(), self.Sig.t()))
        elif method == "mc":
            value = sum(ELBO_MC_squeezed(w, u, y_single, x, mu, sig, n_samples, data_size) for w, u, y_single, mu, sig in zip(self.W_mean.t(), self.u.view(-1, self.num_class).t(), y.t(), self.mu.t(), self.sig.t()))
        elif method == "mc_mvn":
            value = sum(ELBO_MC_mvn_squeezed(w, u, y_single, x, mu, Sig, n_samples, data_size) for w, u, y_single, mu, Sig in zip(self.W_mean.t(), self.u.view(-1, self.num_class).t(), y.t(), self.mu.t(), self.Sig.t()))
        return value
    
    def sample(self, method, n_samples=10000):
        if method in ["tb", "mc"]:
            mvn = MultivariateNormal(self.W_mean.view(-1), torch.diag(self.s**2))
        elif method in ["tb_mvn", "mc_mvn"]:
            mvn = MultivariateNormal(self.W_mean.view(-1), self.S)
        return mvn.sample((n_samples, )).reshape(n_samples, -1, self.num_class)
    
    def neg_log_likelihood(self, x, n_samples=1000):
        if self.intercept:
            x = torch.cat((torch.ones(x.size()[0], 1), x), 1)

        M = self.sample(n_samples=n_samples)
        p = torch.mean(torch.sigmoid(x @ M), 0)

        p[p == 0] = 1e-7
        p[p == 1] = 1 - 1e-7

        return -torch.sum(self.y * torch.log(p) + (1 - self.y) * torch.log(1 - p))
    
    def credible_intervals(self, width=torch.tensor(0.95)):
        d = Normal(self.W_mean, self.S)
        a = (1 - width) / 2

        lower = d.icdf(a)
        upper = d.icdf(1 - a)

        return torch.stack((lower, upper)).t()


class BayesianLastLayer(nn.Module):
    def __init__(self, basemodel_output, num_class, len_dataset = None, mu = None, sig = None, Sig = None, s_init = None, intercept = False) -> None:
        super(BayesianLastLayer, self).__init__()
        self.len_dataset = len_dataset
        assert len_dataset is not None
        self.intercept = intercept

        final_basemodel_output = basemodel_output + int(intercept)
        self.p = final_basemodel_output * num_class
        self.num_class = num_class

        # Prior
        self.mu = torch.zeros(final_basemodel_output, num_class, dtype=torch.double) if mu is None else mu
        self.sig = torch.ones(final_basemodel_output, num_class, dtype=torch.double) if sig is None else sig
        self.Sig = torch.eye(final_basemodel_output * num_class, dtype=torch.double) if Sig is None else Sig

        # Posterior - optimize W_dist (m) and u 
        self.W_dist = Normal # MultivariateNormal
        self.W_mean = nn.Parameter(torch.randn(final_basemodel_output, num_class))
        # self.W_logdiag = nn.Parameter(torch.randn(basemodel_output + int(intercept), num_class) - 0.5 * torch.log(torch.tensor(num_class)))

        # Noise
        self.noise_mean = nn.Parameter(torch.zeros(final_basemodel_output), requires_grad = False)
        self.noise_logdiag = nn.Parameter(torch.randn(final_basemodel_output) - 1)

        if s_init is None:
            self.u_init = torch.ones(final_basemodel_output * num_class) * -1 # , dtype=torch.double # torch.tensor([-1.] * self.W_mean.size(-1), dtype=torch.double)
            self.s_init = torch.exp(self.u_init)
        else:
            self.s_init = s_init
            self.u_init = torch.log(s_init)
        self.u = self.u_init.clone()
        self.u.requires_grad = True

    @torch.no_grad
    def init_covar(self, method):
        if method in ["tb_mvn", "mc_mvn"]:
            self.u = torch.ones(int(self.p * (1 + self.p) / 2.0), device=self.W_mean.device) # , dtype=torch.double
            self.u = self.u * 1/self.p
        self.u_init = self.u.clone()
        self.u.requires_grad = True
        self.update_covar(method)

    @torch.no_grad
    def update_covar(self, method):
        if method == "tb" or method == "mc":
            self.s = torch.exp(self.u)
            self.S = torch.diag(self.s)
        elif method == "tb_mvn" or method == "mc_mvn":
            L = torch.zeros_like(self.Sig)
            L[torch.tril_indices(self.p, self.p, 0).tolist()] = self.u
            self.S = L.t() @ L
            self.s = torch.sqrt(torch.diag(self.S))
        else:
            Exception("bound!")

    def W(self):
        return self.W_dist(self.W_mean.view(-1), self.S)
    
    def multi_distr_tensor(self, d, inp):
        # not self
        # d 512, 512x512. inp .x256
        # hence need reshapes
        if isinstance(d, Normal):
            y = (d.scale ** 2).unsqueeze(-1)
        elif isinstance(d, MultivariateNormal):
            y = d.covariance_matrix
        new_cov = (d.scale ** 2).unsqueeze(-1) * (inp.unsqueeze(-3) ** 2).sum(-2)
        return Normal(d.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))
    
    def logit_predictive(self, x):
        if self.intercept:
            x = torch.cat((torch.ones(x.size()[0], 1), x), 1)
        return (self.multi_distr_tensor(self.W(), x[..., None])).squeeze(-1)
    
    def predictive(self, x, n_samples = 10):
        softmax_samples = F.softmax(self.logit_predictive(x).rsample(sample_shape=torch.Size([n_samples])), dim=-1)
        return torch.clip(torch.mean(softmax_samples, dim=0),min=0.,max=1.)

    def forward(self, x, method):
        n_samples = 10 #FIXME
        M = self.sample(method, n_samples)
        probs = torch.mean(torch.sigmoid(x @ M), 0)
        returns = torch.distributions.Categorical(probs = probs) # or through sample
        return BayesianLastLayerReturn(
            returns,
            self._get_train_loss_fn(x, method),
            # self._get_val_loss_fn(x) FIXME: val loss
            self._get_train_loss_fn(x, method)
        )
    
    def _get_train_loss_fn(self, x, method):

        def loss_fn(y, n_samples = None, l_max = None):
            return self.bound(x, y, method, n_samples, l_max)

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            return -torch.mean(torch.log(self.predictive(x)[torch.arange(x.shape[0]), y]))
        return loss_fn
    
    def bound(self, x, y, method, n_samples = None, l_max = None):
        assert (method in ["tb", "tb_mvn"] and l_max is not None) or (method in ["mc", "mc_mvn"] and n_samples is not None)
        if method == "tb":
            xx = x ** 2
            value = ELBO_TB(self.W_mean, self.u.view(-1, self.num_class), y, x, self.mu, self.sig, self.noise_mean, self.noise_logdiag)
        elif method == "tb_mvn":
            # assume updated
            value = ELBO_TB_mvn(self.W_mean, self.u.view(-1, self.num_class), y, x, self.mu, self.Sig, l_max)
        elif method == "mc":
            value = ELBO_MC(self.W_mean, self.u.view(-1, self.num_class), y, x, self.mu, self.sig, n_samples)
        elif method == "mc_mvn":
            # assume updated
            value = ELBO_MC_mvn(self.W_mean, self.u.view(-1, self.num_class), y, x, self.mu, self.Sig, n_samples)
        return value
    
    def sample(self, method, n_samples=10000):
        if method in ["tb", "mc"]:
            mvn = MultivariateNormal(self.W_mean.view(-1), torch.diag(self.s**2))
        elif method in ["tb_mvn", "mc_mvn"]:
            mvn = MultivariateNormal(self.W_mean.view(-1), self.S)
        return mvn.sample((n_samples, )).reshape(n_samples, -1, self.num_class)
    
    def neg_log_likelihood(self, x, n_samples=1000):
        if self.intercept:
            x = torch.cat((torch.ones(x.size()[0], 1), x), 1)

        M = self.sample(n_samples=n_samples)
        p = torch.mean(torch.sigmoid(x @ M), 0)

        p[p == 0] = 1e-7
        p[p == 1] = 1 - 1e-7

        return -torch.sum(self.y * torch.log(p) + (1 - self.y) * torch.log(1 - p))
    
    def credible_intervals(self, width=torch.tensor(0.95)):
        d = Normal(self.W_mean, self.S)
        a = (1 - width) / 2

        lower = d.icdf(a)
        upper = d.icdf(1 - a)

        return torch.stack((lower, upper)).t()

class MultiLabelModelBayesian(nn.Module):
    def __init__(self, basemodel, basemodel_output, num_classes, binary_squeezed, len_dataset, mu, sig, Sig, s_init, intercept):
        super(MultiLabelModelBayesian, self).__init__()
        self.basemodel = basemodel
        self.num_classes = num_classes
        self.binary_squeezed = binary_squeezed
        if binary_squeezed:
            setattr(self, "BayesianLastLayerSqueezed", self.make_BayesianLastLayerSqueezed(basemodel_output, len(num_classes), len_dataset, mu, sig, Sig, s_init, intercept))
        else:
            for index, num_class in enumerate(num_classes):
                setattr(self, "BayesianLastLayer_" + str(index), self.make_BayesianLastLayer(basemodel_output, num_class, 
                                                                                         len_dataset, mu, sig, Sig, s_init, intercept))

    def make_BayesianLastLayer(self, basemodel_output, num_class, len_dataset, mu, sig, Sig, s_init, intercept):
        return BayesianLastLayer(basemodel_output, num_class, len_dataset, mu, sig, Sig, s_init, intercept)
    
    def make_BayesianLastLayerSqueezed(self, basemodel_output, num_class, len_dataset, mu, sig, Sig, s_init, intercept):
        return BayesianLastLayerSqueezed(basemodel_output, num_class, len_dataset, mu, sig, Sig, s_init, intercept)
    
    def forward(self, x, method):
        x = self.basemodel.forward(x)
        if self.binary_squeezed:
            out = self.BayesianLastLayerSqueezed(x, method)
            return out
        outs = list()
        dir(self)
        for index, num_class in enumerate(self.num_classes):
            fun = eval("self.BayesianLastLayer_" + str(index))
            out = fun(x, method)
            outs.append(out)
        return outs
    
    def init_covar(self, method):
        if self.binary_squeezed:
            getattr(self, f"BayesianLastLayerSqueezed").init_covar(method)
        else:
            for index, num_class in enumerate(self.num_classes):
                getattr(self, f"BayesianLastLayer_{index}").init_covar(method)

    def update_covar(self, method):
        if self.binary_squeezed:
            getattr(self, f"BayesianLastLayerSqueezed").update_covar(method)
        else:
            for index, num_class in enumerate(self.num_classes):
                getattr(self, f"BayesianLastLayer_{index}").update_covar(method)
    
def BuildMultiLabelModelBayesian(basemodel, basemodel_output, num_classes, binary_squeezed, len_dataset, mu, sig, Sig, s_init, intercept):
    return MultiLabelModelBayesian(basemodel, basemodel_output, num_classes, binary_squeezed, len_dataset, mu, sig, Sig, s_init, intercept)
