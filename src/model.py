import torch
import torch.nn as nn
import torch.nn.functional as F

from objective import KL, KL_mvn, neg_ELL_MC_MH, neg_ELL_TB_MH, KL_MH, neg_ELL_MC_mvn_MH, neg_ELL_TB_mvn_MH, KL_mvn_MH, neg_ELL_MC, neg_ELL_TB, neg_ELL_MC_mvn, neg_ELL_TB_mvn

##### SOFTMAX
import sys
import os

def load_vbll(vbll_path):
    sys.path.append(os.path.abspath(vbll_path)) #os.path.join(vbll_path, '..')))
    try:
        import vbll
        print("vbll found")
    except:
        print("vbll not found")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vbll"])
        import vbll

"""## Base model"""

class LLModel(nn.Module):

    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None):
        """
        Parameters:
        ----------
            p : int
                Dimensionality of the input features after processing by the backbone network.
            K : int
                Number of outputs (labels).
        """
        super().__init__()
        print(f"[LLModel] beta={beta} input_dim={p} output_dim={K} intercept={intercept}")
        self.intercept = intercept
        if intercept:
            p += 1
        self.p = p
        self.K = K
        self.backbone = backbone
        self.beta = beta

        self.params = []
        return p

    def process(self, X_batch):
        if self.backbone is not None:
            X_processed = self.backbone(X_batch)
        else:
            X_processed = X_batch

        X_processed = X_processed.to(torch.double)

        if self.intercept:
            X_processed = torch.cat((torch.ones(X_processed.size()[0], 1, device=X_processed.device), X_processed), 1)
        return X_processed

    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        raise NotImplementedError("[LLModel] train_loss not implemented")
    
    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        raise NotImplementedError("[LLModel] test_loss not implemented")

    def predict(self, X, threshold=0.5):
        preds = self.forward(X)
        return (preds > threshold).float(), preds
    
    def forward(self, X):
        raise NotImplementedError("[LLModel] forward mechanism not implemented")

class LLModelCC(LLModel):
    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None, chain_order=None):
        LLModel.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone)
        self.chain_order = chain_order if chain_order is not None else list(range(K))
        print("[LLModelCC] chain_order=", self.chain_order)
        self.chain_order = torch.tensor(self.chain_order, dtype=torch.long)

    def process_chain(self, X_batch):
        raise NotImplementedError("[LLModelCC] process_chain not implemented")
        # X_processed = self.process(X_batch)
        # return X_processed[:, self.chain_order]

"""## Sigmoid-likelihood model"""

class LogisticVI(LLModel):
    """
    Variational Inference for Logistic Regression with Multiple Outputs.

    This class implements variational inference for logistic regression with support for multiple outputs (multi-label classification).
    It uses the original functions for computing the KL divergence and expected negative log-likelihood, and builds multihead versions by calling these functions.

    The class assumes that the data loader and training loop are handled externally.
    It provides methods to compute the ELBO and perform optimization steps given batches of data.

    Parameters:
    ----------
    p : int
        Dimensionality of the input features after processing by the backbone network.
    K : int
        Number of outputs (labels).
    method : int, optional
        Method to use for approximating the ELBO:
        - 0: Proposed bound, diagonal covariance variational family.
        - 1: Proposed bound, full covariance variational family.
        - 4: Monte Carlo, diagonal covariance variational family.
        - 5: Monte Carlo, full covariance variational family.
        Default is 0.
    mu : torch.Tensor, optional
        Prior means for each output. Shape (p, K).
    sig : torch.Tensor, optional
        Prior standard deviations for each output. Shape (p, K).
    Sig : list of torch.Tensor, optional
        Prior covariance matrices for each output. List of K tensors, each of shape (p, p).
    m_init : torch.Tensor, optional
        Initial means of the variational distributions. Shape (p, K).
    s_init : torch.Tensor, optional
        Initial standard deviations (or lower-triangular parameters) of the variational distributions. Shape depends on method.
    l_max : float, optional
        Maximum value of l for the proposed bound. Default is 12.0.
    adaptive_l : bool, optional
        Whether to adaptively increase l during training. Default is False.
    n_samples : int, optional
        Number of samples for Monte Carlo estimation. Default is 500.
    backbone : torch.nn.Module, optional
        Backbone network to transform input features.
    """

    @property
    def mu_list(self):
        return self.prior_mu.expand(self.K, self.p)
        # return torch.full((self.K, self.p), self.prior_mu.item(), dtype=torch.double, requires_grad=self.prior_mu.requires_grad)

    @property
    def prior_scale(self):
        return torch.exp(self.u_sig)

    @property
    def sig_list(self):
        return self.prior_scale.expand(self.K, self.p)
        # return torch.full((self.K, self.p), self.prior_scale.item(), dtype=torch.double, requires_grad=self.u_sig.requires_grad)

    @property
    def Sig_list(self):
        ps = self.prior_scale
        return [torch.eye(self.p, dtype=torch.double, device=ps.device) * ps for _ in range(self.K)]

    def __init__(self, p, K, method=0, beta=1.0, intercept=False, # mu=None, sig=None, Sig=None, 
                 m_init=None, s_init=None, 
                 prior_mean_learnable=False, prior_scale=1.0, prior_scale_learnable=False,
                 posterior_mean_init_scale=1.0, posterior_var_init_add=0.0,
                 incorrect_straight_sigmoid=False, sigmoid_mc_computation=False, sigmoid_mc_n_samples=100,
                 l_max=12.0, adaptive_l=False, n_samples=500, backbone=None):
        p = super().__init__(p, K, beta=beta, intercept=intercept, backbone=backbone)
        print(f"[LogisticVI] method={method} l_max={l_max} adaptive_l={adaptive_l} n_samples={n_samples}")

        self.method = method
        self.l_max = l_max
        self.adaptive_l = adaptive_l
        self.n_samples = n_samples

        self.incorrect_straight_sigmoid = incorrect_straight_sigmoid
        self.sigmoid_mc_computation = sigmoid_mc_computation
        self.sigmoid_mc_n_samples = sigmoid_mc_n_samples

        # Initialize prior parameters
        self.prior_mu = nn.Parameter(torch.tensor(0.0, dtype=torch.double), requires_grad=prior_mean_learnable)

        self.u_sig = nn.Parameter(torch.log(torch.tensor(prior_scale, dtype=torch.double)), requires_grad=prior_scale_learnable)

        # Initialize variational parameters
        if m_init is None:
            self.m_list = [torch.randn(p, dtype=torch.double) * posterior_mean_init_scale for _ in range(K)]
        else:
            self.m_list = [m_init[:, val_k] for i_k, val_k in enumerate(range(K))]

        if s_init is None:
            if method in [0, 4]:
                self.u_list = [torch.tensor([-1.0 + posterior_var_init_add] * p, dtype=torch.double) for _ in range(K)]
                self.s_list = [torch.exp(u) for u in self.u_list]
            elif method in [1, 5]:
                self.u_list = []
                for _ in range(K):
                    u = torch.ones(int(p * (p + 1) / 2), dtype=torch.double) * (1.0 / p)
                    u.requires_grad = True
                    self.u_list.append(u)
        else:
            if method in [0, 4]:
                self.s_list = [s_init[:, val_k] for i_k, val_k in enumerate(range(K))]
                self.u_list = [torch.log(s) for s in self.s_list]
            elif method in [1, 5]:
                self.u_list = s_init  # Should be list of u tensors for each output

        # Set requires_grad=True for variational parameters
        for m in self.m_list:
            m.requires_grad = True
        for u in self.u_list:
            u.requires_grad = True

        # Collect parameters for optimization
        self.params = nn.ParameterList([self.prior_mu, self.u_sig] + list(self.m_list) + list(self.u_list))
        if self.backbone is not None:
            self.params += list(self.backbone.parameters())

        # Initialize l_terms for adaptive l
        if adaptive_l:
            self.l_terms = float(int(l_max / 2))
        else:
            self.l_terms = l_max

    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        data_size = data_size or X_batch.shape[0]
        return -self.compute_ELBO(X_batch, y_batch, data_size, verbose=verbose)

    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        data_size = data_size or X_batch.shape[0]
        return -self.compute_ELBO(X_batch, y_batch, data_size, verbose=verbose, other_beta=0.0)

    def compute_ELBO(self, X_batch, y_batch, data_size, verbose=False, other_beta=None):
        """
        Compute the Evidence Lower Bound (ELBO) for a batch of data.

        Parameters:
        ----------
        X_batch : torch.Tensor
            Batch of input data. Shape (batch_size, input_dim).
        y_batch : torch.Tensor
            Batch of target variables. Shape (batch_size, K).

        Returns:
        -------
        ELBO : torch.Tensor
            The computed ELBO for the batch.
        """
        X_processed = self.process(X_batch)
        batch_size = X_batch.shape[0]

        # Prepare lists for variational parameters and priors
        m_list = [m.to(X_batch.device) for m in self.m_list]
        mu_list = [mu.to(X_batch.device) for mu in self.mu_list]
        y_list = [y_batch[:, val_k] for i_k, val_k in enumerate(range(self.K))]

        if self.method in [0, 4]:
            s_list = [torch.exp(u).to(X_batch.device) for u in self.u_list]
            sig_list = [sig.to(X_batch.device) for sig in self.sig_list]

            if self.method == 0:
                likelihood = -neg_ELL_TB_MH(m_list, s_list, y_list, X_processed, l_max=self.l_terms)
                KL_div = KL_MH(m_list, s_list, mu_list, sig_list)
            else:
                likelihood = -neg_ELL_MC_MH(m_list, s_list, y_list, X_processed.to(X_batch.device), n_samples=self.n_samples)
                KL_div = KL_MH(m_list, s_list, mu_list, sig_list)

        elif self.method in [1, 5]:
            L_list = []
            u_list = [u.to(X_batch.device) for u in self.u_list]
            for u in u_list:
                L = torch.zeros(self.p, self.p, dtype=torch.double, device=X_batch.device)
                tril_indices = torch.tril_indices(self.p, self.p, 0).to(X_batch.device)
                L[tril_indices[0], tril_indices[1]] = u
                L_list.append(L)

            S_list = [L @ L.t() for L in L_list]
            Sig_list = [Sig.to(X_batch.device) for Sig in self.Sig_list]
            if self.method == 1:
                likelihood = -neg_ELL_TB_mvn_MH(m_list, S_list, y_list, X_processed, l_max=self.l_terms)
                KL_div = KL_mvn_MH(m_list, S_list, mu_list, Sig_list)
            else:
                likelihood = -neg_ELL_MC_mvn_MH(m_list, S_list, y_list, X_processed, n_samples=self.n_samples)
                KL_div = KL_mvn_MH(m_list, S_list, mu_list, self.Sig_list)

        else:
            raise ValueError("Method not recognized")

        mean_log_lik = likelihood/batch_size
        mean_kl_div = KL_div/data_size
        beta = other_beta or self.beta
        ELBO = mean_log_lik - beta*mean_kl_div
        if verbose:
            print(f"ELBO={ELBO:.2f} mean_log_lik={mean_log_lik:.2f} mean_kl_div={mean_kl_div:.2f}")
        return ELBO

    def compute_negative_log_likelihood(self, X, y, mc = False, n_samples = 1000):
        X_processed = self.process(X)
        m_list = [m.to(X.device) for m in self.m_list]
        y_list = [y[:, val_k] for i_k, val_k in enumerate(range(self.K))]
        if self.method in [0, 4]:
            s_list = [torch.exp(u).to(X.device) for u in self.u_list]
            if mc:
                nlls = [neg_ELL_MC(m, s, y, X_processed, n_samples=n_samples) for m, s, y in zip(m_list, s_list, y_list)]
            else:
                nlls = [neg_ELL_TB(m, s, y, X_processed, l_max=self.l_terms) for m, s, y in zip(m_list, s_list, y_list)]
        elif self.method in [1, 5]:
            L_list = []
            u_list = [u.to(X.device) for u in self.u_list]
            for u in u_list:
                L = torch.zeros(self.p, self.p, dtype=torch.double, device=X.device)
                tril_indices = torch.tril_indices(self.p, self.p, 0).to(X.device)
                L[tril_indices[0], tril_indices[1]] = u
                L_list.append(L)

            S_list = [L @ L.t() for L in L_list]
            if mc:
                nlls = [neg_ELL_MC_mvn(m, S, y, X_processed, n_samples=n_samples) for m, S, y in zip(m_list, S_list, y.T)]
            else:
                nlls = [neg_ELL_TB_mvn(m, S, y, X_processed, l_max=self.l_terms) for m, S, y in zip(m_list, S_list, y.T)]
        return torch.tensor(nlls)

    def get_confidences(self, preds):
        return torch.max(torch.stack([preds, 1 - preds]), dim=0)[0]

    def expected_sigmoid_multivariate(self, X, m, u, mc=False, n_samples=None):
        assert not mc or n_samples is not None, "n_samples must be provided for Monte Carlo estimation"
        m = m.to(X.device)
        M = X @ m
        if self.incorrect_straight_sigmoid:
            return torch.sigmoid(M)
        if self.method in [0, 4]:
            s = torch.exp(u).to(X.device)
            scaling_factor_diag = torch.sum(m**2 * s)
            if not mc:
                scaling_factor = torch.sqrt(1 + (torch.pi / 8) * scaling_factor_diag)
                # scaling_factor = torch.sqrt(1 + (torch.pi / 8) * (m.T @ S @ m))
                expected_sigmoid = torch.sigmoid(M / scaling_factor)
            else:
                S = torch.sqrt(torch.sum(X ** 2 * (s ** 2), dim=1)) # ref objective.py 142
                for i in range(len(self.sigmoid_mc_n_samples)):
                    norm = torch.distributions.Normal(loc=M, scale=S)
                    samples = norm.sample((n_samples,))
                    sigmoid_samples = torch.sigmoid(samples)
                    expected_sigmoid = sigmoid_samples.mean(dim=0)
        elif self.method in [1, 5]:
            L = torch.zeros(u.size(0), u.size(0), dtype=torch.double, device=X.device)
            tril_indices = torch.tril_indices(L.size(0), L.size(1), offset=0).to(X.device)
            L[tril_indices[0], tril_indices[1]] = u.to(X.device)
            cov = L @ L.T
            if not mc:
                scaling_factor = 1 / torch.sqrt(1 + (torch.pi / 8) * (m.T @ cov @ m))
                expected_sigmoid = torch.sigmoid(M * scaling_factor)
            else:
                mvn = torch.distributions.MultivariateNormal(loc=M, covariance_matrix=cov)
                samples = mvn.sample((n_samples,))
                sigmoid_samples = torch.sigmoid(samples)
                expected_sigmoid = sigmoid_samples.mean(dim=0)
        return expected_sigmoid

    def forward(self, X):
        """
        Predict probabilities for each output given input data.

        Parameters:
        ----------
        X : torch.Tensor
            Input data. Shape (n_samples, input_dim).

        Returns:
        -------
        preds : torch.Tensor
            Predicted probabilities for each output. Shape (n_samples, K).
        """
        X_processed = self.process(X)
        u_list = [u.to(X_processed.device) for u in self.u_list]

        preds = []
        for m, u in zip(self.m_list, u_list):
            pred = self.expected_sigmoid_multivariate(X_processed, m, u, mc=self.sigmoid_mc_computation, n_samples=self.sigmoid_mc_n_samples)
            preds.append(pred.unsqueeze(1))

        preds = torch.cat(preds, dim=1)  # Shape: (n_samples, K)
        return preds

"""## CC Sigmoid Deterministic model"""

class LogisticVICC(LLModelCC, LogisticVI):

    @property
    def mu_list(self):
        return [self.prior_mu.expand(self.p + val_k) for i_k, val_k in enumerate(self.chain_order)]

    @property
    def prior_scale(self):
        return torch.exp(self.u_sig)

    @property
    def sig_list(self):
        return [self.prior_scale.expand(self.p + val_k) for i_k, val_k in enumerate(self.chain_order)]

    @property
    def Sig_list(self):
        ps = self.prior_scale
        return [torch.eye(self.p, dtype=torch.double, device=ps.device) * ps for _ in range(self.K)]

    def __init__(self, p, K, method=0, beta=1.0, intercept=False, backbone=None, m_init=None, s_init=None, 
                 prior_mean_learnable=False, prior_scale=1.0, prior_scale_learnable=False,
                 posterior_mean_init_scale=1.0, posterior_var_init_add=0.0,
                 incorrect_straight_sigmoid=False, sigmoid_mc_computation=False, sigmoid_mc_n_samples=100,
                 l_max=12.0, adaptive_l=False, n_samples=500, chain_order=None):
        LLModelCC.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone, chain_order=chain_order)
        LogisticVI.__init__(self, p, K, method=method, beta=beta, intercept=intercept, backbone=backbone, m_init=m_init, s_init=s_init,
                            prior_mean_learnable=prior_mean_learnable, prior_scale=prior_scale, prior_scale_learnable=prior_scale_learnable,
                            posterior_mean_init_scale=posterior_mean_init_scale, posterior_var_init_add=posterior_var_init_add,
                            incorrect_straight_sigmoid=incorrect_straight_sigmoid, sigmoid_mc_computation=sigmoid_mc_computation, sigmoid_mc_n_samples=sigmoid_mc_n_samples,
                            l_max=l_max, adaptive_l=adaptive_l, n_samples=n_samples)
        print(f"[LogisticVICC]")

        self.prior_mu = nn.Parameter(torch.tensor(0.0, dtype=torch.double), requires_grad=prior_mean_learnable)
        self.u_sig = nn.Parameter(torch.log(torch.tensor(prior_scale, dtype=torch.double)), requires_grad=prior_scale_learnable)

        if m_init is None:
            m_list = [torch.randn(self.p+val_k, dtype=torch.double) for i_k, val_k in enumerate(self.chain_order)]
        else:
            if isinstance(m_init, torch.Tensor):
                m_list = []
                for i_k in range(m_init.shape[1]):
                    m = m_init[:, i_k]
                    if m.shape[0] == self.p:
                        extended_m = torch.randn(self.p+self.chain_order[i_k], dtype=torch.double)
                        extended_m[:self.p] = m
                        m_list.append(nn.Parameter(extended_m))
                    elif m.shape[0] == self.p+self.K:
                        m_list.append(nn.Parameter(m[:self.p+self.chain_order[i_k]]))
            elif isinstance(m_init, list):
                m_list = []
                assert len(m_init) == self.K
                for i_k, m in enumerate(m_init):
                    if m.shape[0] == self.p+self.chain_order[i_k]:
                        m_list.append(nn.Parameter(m))
                    elif m.shape[0] == self.p:
                        extended_m = torch.randn(self.p+self.chain_order[i_k], dtype=torch.double)
                        extended_m[:self.p] = m
                        m_list.append(nn.Parameter(extended_m))

        if s_init is None:
            if method in [0, 4]:
                self.u_list = [torch.tensor([-1.0 + posterior_var_init_add] * (self.p+self.chain_order[i_k]), dtype=torch.double) for i_k in range(self.K)]
                self.s_list = [torch.exp(u) for u in self.u_list]
            elif method in [1, 5]:
                self.u_list = []
            for i_k in range(self.K):
                u = torch.ones(int((self.p+self.chain_order[i_k]) * (self.p+self.chain_order[i_k] + 1) / 2), dtype=torch.double) * (1.0 / (self.p+self.chain_order[i_k]))
                u.requires_grad = True
                self.u_list.append(u)
        else:
            if method in [0, 4]:
                s_list = []
                if isinstance(s_init, torch.Tensor):
                    for i_k in range(s_init.shape[1]):
                        s = s_init[:, k]
                        if s.shape[0] == self.p:
                            extended_s = torch.randn(self.p+self.chain_order[i_k], dtype=torch.double)
                            extended_s[:self.p] = s
                            s_list.append(extended_s)
                        elif s.shape[0] == self.p+self.chain_order[i_k]:
                            s_list.append(s)
                elif isinstance(s_init, list):
                    assert len(s_init) == self.K
                    for i_k, s in enumerate(s_init):
                        if s.shape[0] == self.p+self.chain_order[i_k]:
                            s_list.append(s)
                        elif s.shape[0] == self.p:
                            extended_s = torch.randn(self.p+self.chain_order[i_k], dtype=torch.double)
                            extended_s[:self.p] = s
                            s_list.append(extended_s)
                self.s_list = [s_list[:, i_k] for i_k in range(self.K)]
                self.u_list = [torch.log(s) for s in self.s_list]
            elif method in [1, 5]:
                s_list = []
                if isinstance(s_init, torch.Tensor):
                    if len(s_init) == int(self.p * (self.p + 1) / 2):
                        for i_k, s in enumerate(s_init):
                            x = torch.ones(int((self.p+self.chain_order[i_k]) * (self.p+self.chain_order[i_k] + 1) / 2), dtype=torch.double) * (1.0 / (self.p+self.chain_order[i_k]))
                            x[:s.shape[0]] = s
                            s_list.append(x)
                    elif len(s_init) == self.p+self.K-1:
                        for k, s in enumerate(s_init):
                            s_list.append(s[:self.p+self.chain_order[i_k]])
                elif isinstance(s_init, list):
                    assert len(s_init) == self.K
                    for i_k, s in enumerate(s_init):
                        if s.shape[0] == self.p+self.chain_order[i_k]:
                            s_list.append(s)
                        elif s.shape[0] == self.p:
                            extended_s = torch.randn(self.p+self.chain_order[i_k], dtype=torch.double)
                            extended_s[:self.p] = s
                            s_list.append(extended_s)
                self.u_list = s_list

        self.m_list = m_list
        self.heads = m_list # ref or leave?

        for m in self.m_list:
            m.requires_grad = True
        for u in self.u_list:
            u.requires_grad = True

        self.params = nn.ParameterList(self.m_list)
        if self.backbone is not None:
            self.params += list(self.backbone.parameters())

    def forward(self, X_batch):
        X_processed = self.process(X_batch)
        X_processed = X_processed.to(torch.double)

        preds = []
        for i, k in enumerate(self.chain_order):
            relevant_head = (self.chain_order == i).nonzero().item() # if numpy then np.nonzero(self.chain_order == i)[0].item()
            if i == 0:
                pred_after_sigmoid = self.expected_sigmoid_multivariate(X_processed, self.m_list[relevant_head], self.u_list[relevant_head].to(X_processed.device), mc=self.sigmoid_mc_computation, n_samples=self.sigmoid_mc_n_samples)
                # pred = X_processed @ self.heads[relevant_head].to(X_processed.device)
            else:
                prev_preds = torch.cat(preds, dim=1)
                # pred = torch.cat((X_processed, prev_preds), dim=1) @ self.heads[relevant_head].to(X_processed.device)
                pred_after_sigmoid = self.expected_sigmoid_multivariate(torch.cat((X_processed, prev_preds), dim=1), self.m_list[relevant_head], self.u_list[relevant_head].to(X_processed.device), mc=self.sigmoid_mc_computation, n_samples=self.sigmoid_mc_n_samples)
            # preds.append(pred.unsqueeze(1))
            preds.append(pred_after_sigmoid.unsqueeze(1))
        # preds = [torch.sigmoid(pred) for pred in preds]
        return torch.cat(preds, dim=1)

    def compute_ELBO(self, X_batch, y_batch, data_size, verbose=False, other_beta=None):
        X_processed = self.process(X_batch)
        batch_size = X_batch.shape[0]

        m_list = [m.to(X_batch.device) for m in self.m_list]
        mu_list = [mu.to(X_batch.device) for mu in self.mu_list]
        y_list = [y_batch[:, k] for k in range(self.K)]

        likelihood = torch.tensor(0.0, dtype=torch.double, device=X_batch.device)
        KL_div = torch.tensor(0.0, dtype=torch.double, device=X_batch.device)

        preds = []
        for i_k, val_k in enumerate(self.chain_order):
            relevant_head = (self.chain_order == i_k).nonzero().item()
            if i_k == 0:
                X = X_processed
            else:
                X = torch.cat((X_processed, torch.cat(preds, dim=1)), dim=1)
            # pred = X @ m_list[relevant_head].to(X_processed.device)
            pred_after_sigmoid = self.expected_sigmoid_multivariate(X, m_list[relevant_head], self.u_list[relevant_head].to(X.device), mc=self.sigmoid_mc_computation, n_samples=self.sigmoid_mc_n_samples)
            # preds.append(pred.unsqueeze(1))
            preds.append(pred_after_sigmoid.unsqueeze(1))

            if self.method in [0, 4]:
                s = torch.exp(self.u_list[relevant_head].to(X_batch.device))
                sig = self.sig_list[relevant_head].to(X_batch.device)
                if self.method == 0:
                    likelihood += -neg_ELL_TB(m_list[relevant_head], s, y_list[relevant_head], X, l_max=self.l_terms)
                    KL_div += KL(m_list[relevant_head], s, mu_list[relevant_head], sig)
                else:
                    likelihood += -neg_ELL_MC(m_list[relevant_head], s, y_list[relevant_head], X, n_samples=self.n_samples)
                    KL_div += KL(m_list[relevant_head], s, mu_list[relevant_head], sig)

            elif self.method in [1, 5]:

                u = self.u_list[relevant_head].to(X_batch.device)
                L = torch.zeros(self.p, self.p, dtype=torch.double, device=X_batch.device)
                tril_indices = torch.tril_indices(self.p, self.p, 0).to(X_batch.device)
                L[tril_indices[0], tril_indices[1]] = u
                S = L @ L.t()

                Sig = self.Sig_list[relevant_head].to(X_batch.device)
                if self.method == 1:
                    likelihood += -neg_ELL_TB_mvn(m_list[relevant_head], S, y_list[relevant_head], X, l_max=self.l_terms)
                    KL_div += KL_mvn(m_list[relevant_head], S, mu_list[relevant_head], Sig)
                else:
                    likelihood += -neg_ELL_MC_mvn(m_list[relevant_head], S, y_list[relevant_head], X, n_samples=self.n_samples)
                    KL_div += KL_mvn(m_list[relevant_head], S, mu_list[relevant_head], Sig)
            else:
                raise ValueError("Method not recognized")

        mean_log_lik = likelihood/batch_size
        mean_kl_div = KL_div/data_size
        beta = other_beta or self.beta
        ELBO = mean_log_lik - beta*mean_kl_div
        if verbose:
            print(f"ELBO={ELBO:.2f} mean_log_lik={mean_log_lik:.2f} mean_kl_div={mean_kl_div:.2f}")
        return ELBO

    def compute_negative_log_likelihood(self, X, y, mc = False, n_samples = 1000):
        X_processed = self.process(X)
        m_list = [m.to(X.device) for m in self.m_list]
        y_list = [y[:, k] for k in range(self.K)]
        likelihood = []

        preds = []
        for i_k, val_k in enumerate(self.chain_order):
            relevant_head = (self.chain_order == i_k).nonzero().item()
            if i_k == 0:
                XX = X_processed
            else:
                XX = torch.cat((X_processed, torch.cat(preds, dim=1)), dim=1)
            # pred = XX @ m_list[relevant_head].to(X.device)
            pred_after_sigmoid = self.expected_sigmoid_multivariate(XX, m_list[relevant_head], self.u_list[relevant_head].to(X.device), mc=self.sigmoid_mc_computation, n_samples=self.sigmoid_mc_n_samples)
            # preds.append(pred.unsqueeze(1))
            preds.append(pred_after_sigmoid.unsqueeze(1))

            if self.method in [0, 4]:
                s = torch.exp(self.u_list[relevant_head].to(X.device))

                if mc:
                    cur_likelihood = -neg_ELL_MC(m_list[relevant_head], s, y_list[relevant_head], XX, n_samples=n_samples)
                else:
                    cur_likelihood = -neg_ELL_TB(m_list[relevant_head], s, y_list[relevant_head], XX, l_max=self.l_terms)

            elif self.method in [1, 5]:
                u = self.u_list[relevant_head].to(X.device)
                L = torch.zeros(self.p, self.p, dtype=torch.double, device=X.device)
                tril_indices = torch.tril_indices(self.p, self.p, 0).to(X.device)
                L[tril_indices[0], tril_indices[1]] = u
                S = L @ L.t()

                if mc:
                    cur_likelihood = -neg_ELL_MC_mvn(m_list[relevant_head], S, y_list[relevant_head], XX, n_samples=n_samples)
                else:
                    cur_likelihood = -neg_ELL_TB_mvn(m_list[relevant_head], S, y_list[relevant_head], XX, l_max=self.l_terms)
            else:
                raise ValueError("Method not recognized")
            likelihood.append(cur_likelihood)

        return torch.tensor(likelihood)


"""## Sigmoid Deterministic model"""

class LogisticPointwise(LLModel):
    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None, m_init=None, mu=None, Sig=None):
        p = super().__init__(p, K, beta=beta, intercept=intercept, backbone=backbone)
        print(f"[LogisticPointwise]")

        # Initialize prior parameters
        if mu is None:
            self.mu_list = [torch.zeros(self.p+k, dtype=torch.double) for k in range(K)]
        else:
            self.mu_list = [mu[:, self.p+k] for k in range(K)]

        if Sig is None:
            self.Sig_list = [torch.eye(p, dtype=torch.double) for _ in range(K)]
        else:
            self.Sig_list = Sig

        if m_init is None:
            self.m_list = [nn.Parameter(torch.randn(p, dtype=torch.double)) for _ in range(K)]
        else:
            self.m_list = [nn.Parameter(m_init[:, k]) for k in range(K)]

        self.params = nn.ParameterList(self.m_list)
        if self.backbone is not None:
            self.params += list(self.backbone.parameters())

    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        data_size = data_size or X_batch.shape[0]

        preds = self.forward(X_batch)
        assert preds.shape == y_batch.shape, f"preds.shape={preds.shape} != y_batch.shape={y_batch.shape}"
        loss = nn.BCELoss(reduction='mean')
        mean_bce = loss(preds, y_batch)
        mean_reg = self.regularization() / data_size if self.beta else torch.tensor(0.0)

        if verbose:
            print(f"mean_bce_loss={mean_bce:.2f}  mean_reg={mean_reg:.2f}")

        return mean_bce

    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False, other_beta=None):
        data_size = data_size or X_batch.shape[0]

        preds = self.forward(X_batch)
        assert preds.shape == y_batch.shape, f"preds.shape={preds.shape} != y_batch.shape={y_batch.shape}"
        loss = nn.BCELoss(reduction='mean')
        mean_bce = loss(preds, y_batch)

        beta = other_beta or self.beta
        mean_reg = self.regularization() / data_size if beta else 0.0

        if verbose:
            print(f"mean_bce_loss={mean_bce:.2f}  mean_reg={mean_reg:.2f}")
        return mean_bce + beta * mean_reg

    def regularization(self):
        log_prob = 0.
        for m, prior_mu, prior_Sig in zip(self.m_list, self.mu_list, self.Sig_list):
            d = torch.distributions.MultivariateNormal(loc=prior_mu.to(m.device), covariance_matrix=prior_Sig.to(m.device))
            log_prob += d.log_prob(m)
        return -log_prob

    def forward(self, X):
        """
        Predict probabilities for each output given input data.

        Parameters:
        ----------
        X : torch.Tensor
            Input data. Shape (n_samples, input_dim).

        Returns:
        -------
        preds : torch.Tensor
            Predicted probabilities for each output. Shape (n_samples, K).
        """
        X_processed = self.process(X)

        preds = []
        for m in self.m_list:
            pred = torch.sigmoid(X_processed @ m.to(X_processed.device))
            preds.append(pred.unsqueeze(1))

        preds = torch.cat(preds, dim=1)  # Shape: (n_samples, K)
        return preds

    def compute_negative_log_likelihood(self, X, y, mc = True, n_samples = 1000):
        """
        Compute the negative log likelihood of the data given the predictions.

        Parameters:
        ----------
        preds : torch.Tensor
            Predicted probabilities for each output. Shape (n_samples, K).
        y : torch.Tensor
            Target variables. Shape (n_samples, K).
        mc, n_samples : bool, int (optional) Dumb arguments

        Returns:
        -------
        nll : torch.Tensor
            The computed negative log likelihood for each attribute. Shape (K).
        """
        preds = self.forward(X)
        loss = nn.BCELoss(reduction='none')
        # likelihood = torch.exp(-loss(preds, y))
        # mean_likelihood = torch.mean(likelihood, dim=0)
        nll = torch.mean(loss(preds, y), dim=0)
        return nll

    def get_confidences(self, preds):
        return torch.max(torch.stack([preds, 1 - preds]), dim=0)[0]

""" Logistic Pointwise CC model """

class LogisticPointwiseCC(LLModelCC, LogisticPointwise):
    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None, m_init=None, mu=None, Sig=None, chain_order=None):
        LLModelCC.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone, chain_order=chain_order)
        LogisticPointwise.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone, m_init=m_init, mu=mu, Sig=Sig)
        print(f"[LogisticPointwiseCC]")

        if m_init is None:
            #FIXME
            m_list = [nn.Parameter(torch.randn(self.p+val_k, dtype=torch.double)) for i_k, val_k in enumerate(self.chain_order)]
        else:
            if isinstance(m_init, torch.Tensor):
                m_list = []
                for k in range(m_init.shape[1]):
                    m = m_init[:, k]
                    if m.shape[0] == self.p:
                        extended_m = torch.randn(self.p+self.chain_order[k], dtype=torch.double)
                        extended_m[:self.p+self.chain_order[k]] = m
                        m_list.append(nn.Parameter(extended_m))
                    elif m.shape[0] == self.p+self.K:
                        m_list.append(nn.Parameter(m[:self.p+self.chain_order[k]]))
            elif isinstance(m_init, list):
                m_list = []
                assert len(m_init) == self.K
                for k, m in enumerate(m_init):
                    if m.shape[0] == self.p+self.chain_order[k]:
                        m_list.append(nn.Parameter(m))
                    elif m.shape[0] == self.p:
                        extended_m = torch.randn(self.p+self.chain_order[k], dtype=torch.double)
                        extended_m[:self.p] = m
                        m_list.append(nn.Parameter(extended_m))

        self.m_list = m_list
        self.heads = m_list # ref or leave?

        for m in self.m_list:
            m.requires_grad = True

        self.params = nn.ParameterList(self.m_list)
        if self.backbone is not None:
            self.params += list(self.backbone.parameters())

    # def regularization(self):
    #     log_prob = 0.
    #     for m, prior_mu, prior_Sig in zip(self.m_list, self.mu_list, self.Sig_list):
    #         # simple prior distribution
    #         d = torch.distributions.MultivariateNormal(loc=prior_mu.to(m.device), covariance_matrix=prior_Sig.to(m.device))
    #         log_prob += d.log_prob(m)
    #     return -log_prob
    
    def forward(self, X_batch):
        X_processed = self.process(X_batch)
        X_processed = X_processed.to(torch.double)
        logits = []
        for i, k in enumerate(self.chain_order):
            if i == 0:
                logit = torch.sigmoid(X_processed @ self.heads[(self.chain_order == i).nonzero().item()]).to(X_processed.device)
            else:
                prev_logits = torch.cat(logits, dim=1)
                logit = torch.sigmoid(torch.cat((X_processed, prev_logits), dim=1) @ self.heads[(self.chain_order == i).nonzero().item()]).to(X_processed.device)
            logits.append(logit.unsqueeze(1))
        
        return torch.cat(logits, dim=1)

"""## VBLL models"""

class SoftmaxVBLL(LLModel):
    def __init__(self, p, K, beta, vbll_cfg, num_classes_lst=None, intercept=False, backbone=None):
        p = super().__init__(p, K, beta, intercept=intercept, backbone=backbone)
        print(f"[SoftmaxVBLL] vbll_cfg={vbll_cfg}")
        vbll_cfg.REGULARIZATION_WEIGHT = self.beta
        self.vbll_cfg = vbll_cfg

        if num_classes_lst is None:
            self.num_classes_single = 2
            self.num_classes_lst = [self.num_classes_single] * self.K
        else:
            self.num_classes_single = num_classes_lst[0]
            self.num_classes_lst = num_classes_lst


        self.heads = nn.ModuleList([self.make_output_layer(num_hidden=self.p, num_classes=self.num_classes_lst[k]) for k in range(self.K)])

        # Collect parameters for optimization
        self.params = []
        if self.backbone is not None:
            self.params += list(self.backbone.parameters())
        for head in self.heads:
            self.params += list(head.parameters())

    def make_output_layer(self, **kwargs):
        load_vbll(self.vbll_cfg.PATH)
        if self.vbll_cfg.TYPE == "disc":
            return self._make_disc_vbll_layer(cfg=self.vbll_cfg, **kwargs).double()

        elif self.vbll_cfg.TYPE == "gen":
            return self._make_gen_vbll_layer(cfg=self.vbll_cfg, **kwargs).double()

        else:
            raise ValueError(f"Unknown VBLL type={self.vbll_cfg.TYPE}!")

    def _make_disc_vbll_layer(self, num_hidden, num_classes, cfg):
        """ VBLL Discriminative classification head. """
        import vbll
        return vbll.DiscClassification( num_hidden,
                                        num_classes,
                                        self.beta,
                                        softmax_bound=cfg.SOFTMAX_BOUND,
                                        # return_empirical=cfg.RETURN_EMPIRICAL,
                                        # softmax_bound_empirical=cfg.SOFTMAX_BOUND_EMPIRICAL,
                                        parameterization = cfg.PARAMETRIZATION,
                                        return_ood=cfg.RETURN_OOD,
                                        prior_scale=cfg.PRIOR_SCALE,
                                       )

    def _make_gen_vbll_layer(self, num_hidden, num_classes, cfg):
        """ VBLL Generative classification head. """
        import vbll
        return vbll.GenClassification(  num_hidden,
                                        num_classes,
                                        self.beta,
                                        softmax_bound=cfg.SOFTMAX_BOUND,
                                        # return_empirical=cfg.RETURN_EMPIRICAL,
                                        # softmax_bound_empirical=cfg.SOFTMAX_BOUND_EMPIRICAL,
                                        parameterization = cfg.PARAMETRIZATION,
                                        return_ood=cfg.RETURN_OOD,
                                        prior_scale=cfg.PRIOR_SCALE)

    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        data_size = data_size or X_batch.shape[0]
        X_processed = self.process(X_batch)

        loss = 0.
        for i, (head, y) in enumerate(zip(self.heads, y_batch.T)):
            loss1 = head(X_processed).train_loss_fn(y.long())
            loss += loss1
            if verbose:
                print(f"head={i} loss={loss1:.2f}")

        return loss

    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        data_size = data_size or X_batch.shape[0]
        X_processed = self.process(X_batch)

        loss = 0.
        for i, (head, y) in enumerate(zip(self.heads, y_batch.T)):
            loss1 = head(X_processed).val_loss_fn(y.long())
            loss += loss1
            if verbose:
                print(f"head={i} loss={loss1:.2f}")

        return loss

    def forward(self, X):
        """
        Predict probabilities for each output given input data.

        Parameters:
        ----------
        X : torch.Tensor
            Input data. Shape (n_samples, input_dim).

        Returns:
        -------
        preds : torch.Tensor
            Predicted probabilities for each output. Shape (n_samples, K).
        """
        X_processed = self.process(X)

        preds = []
        for head in self.heads:
            pred = head(X_processed).predictive.probs
            preds.append(pred) # .unsqueeze(1)

        return torch.stack(preds, dim=1)

    def predict(self, X, threshold=None):
        preds = self.forward(X)
        all_preds = []
        for i_k in range(self.K):
            max_class = torch.argmax(preds[:, i_k, :], dim=-1)
            all_preds.append(max_class)
        return torch.stack(all_preds, dim=1), preds

    def compute_negative_log_likelihood(self, X, y, mc = False, n_samples = 1000):
        X_processed = self.process(X)
        nlls = []
        for head, y in zip(self.heads, y.T):
            nll = head(X_processed).val_loss_fn(y.long())
            nlls.append(nll)
        return torch.stack(nlls)
    
    def get_confidences(self, preds):
        return torch.max(preds, dim=-1)[0]

""" ## Softmax VBLL CC model """

class SoftmaxVBLLCC(LLModelCC, SoftmaxVBLL):
    def __init__(self, p, K, beta, vbll_cfg, num_classes_lst=None, intercept=False, backbone=None, chain_order=None):
        LLModelCC.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone, chain_order=chain_order)
        SoftmaxVBLL.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone, num_classes_lst=num_classes_lst, vbll_cfg=vbll_cfg)
        print(f"[SoftmaxVBLLCC]")

        self.heads = nn.ModuleList([self.make_output_layer(num_hidden=self.p+sum(self.num_classes_lst[:k]), num_classes=self.num_classes_lst[k]) for k in range(self.K)])
        self.heads = nn.ModuleList([self.heads[(self.chain_order == i_k).nonzero().item()] for i_k, val_k in enumerate(self.chain_order)])

        self.params = nn.ParameterList()
        if self.backbone is not None:
            self.params += list(self.backbone.parameters())
        for head in nn.ModuleList(self.heads):
            self.params += list(head.parameters())
    
    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        data_size = data_size or X_batch.shape[0]
        X_processed = self.process(X_batch)

        logits = []
        loss = 0.
        for i_k, val_k in enumerate(self.chain_order):
            relevant_head = val_k # (self.chain_order == i_k).nonzero().item()
            if i_k == 0:
                out = self.heads[relevant_head](X_processed)
                loss1 = out.train_loss_fn(y_batch.T[relevant_head].long())
            else:
                prev_logits = torch.cat(logits, dim=1)
                out = self.heads[relevant_head](torch.cat((X_processed, prev_logits), dim=1))
                loss1 = out.train_loss_fn(y_batch.T[relevant_head].long())
            logits.append(out.predictive.probs)
            loss += loss1
            if verbose:
                print(f"head={i_k} loss={loss1:.2f}")
        return loss

    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        data_size = data_size or X_batch.shape[0]
        X_processed = self.process(X_batch)

        logits = []
        loss = 0.
        for i_k, val_k in enumerate(self.chain_order):
            relevant_head = val_k # (self.chain_order == i).nonzero().item()
            if i_k == 0:
                out = self.heads[relevant_head](X_processed)
                loss1 = self.heads[relevant_head](X_processed).val_loss_fn(y_batch.T[relevant_head].long())
            else:
                prev_logits = torch.cat(logits, dim=1)
                out = self.heads[relevant_head](torch.cat((X_processed, prev_logits), dim=1))
                loss1 = self.heads[relevant_head](torch.cat((X_processed, prev_logits), dim=1)).val_loss_fn(y_batch.T[relevant_head].long())
            logits.append(out.predictive.probs)
            loss += loss1
            if verbose:
                print(f"head={i_k} loss={loss1:.2f}")
            return loss
    
    def forward(self, X):
        X_processed = self.process(X)
        logits = []
        for i_k, val_k in enumerate(self.chain_order):
            if i_k == 0:
                logit = self.heads[val_k](X_processed).predictive.probs
            else:
                logit = self.heads[val_k](torch.cat((X_processed, torch.cat(logits, dim=1)), dim=1)).predictive.probs
            logits.append(logit)
        return torch.stack(logits, dim=1)
    
    def compute_negative_log_likelihood(self, X, y, mc=False, n_samples=1000):
        X_processed = self.process(X)
        nlls = []
        logits = []
        for i_k, val_k in enumerate(self.chain_order):
            relevant_head = val_k
            head = self.heads[relevant_head]
            y_k = y[:, relevant_head]
            if i_k == 0:
                out = head(X_processed)
            else:
                out = head(torch.cat((X_processed, torch.cat(logits, dim=1)), dim=1))
            nll = out.val_loss_fn(y_k.long())
            nlls.append(nll)
            logit = out.predictive.probs
            logits.append(logit)
        return torch.stack(nlls)

"""## Softmax Pointwise model"""

class SoftmaxPointwise(LLModel):
    def __init__(self, p, K, beta=0.0, num_classes_lst=None, intercept=False, backbone=None):
        p = super().__init__(p, K, beta=beta, intercept=intercept, backbone=backbone)
        print(f"[SoftmaxPointwise]")
        if num_classes_lst is None:
            self.num_classes_single = 2
            self.num_classes_lst = [self.num_classes_single] * self.K
        else:
            self.num_classes_single = num_classes_lst[0]
            self.num_classes_lst = num_classes_lst

        self.heads = nn.ModuleList([self.make_output_layer(num_classes=self.num_classes_lst[k]) for k in range(self.K)])
        self.loss = nn.CrossEntropyLoss(reduction='mean')

        # Collect parameters for optimization
        self.params = []
        if self.backbone is not None:
            self.params += list(self.backbone.parameters())
        for head in self.heads:
            self.params += list(head.parameters())

    def make_output_layer(self, num_classes):
        return nn.Linear(self.p, num_classes).to(torch.double)

    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        data_size = data_size or X_batch.shape[0]
        X_processed = self.process(X_batch)

        total_loss = 0.0

        for i, (head,y) in enumerate(zip(self.heads,y_batch.T)):
            pred = head(X_processed)
            loss_head = self.loss(pred, y.to(torch.long)) 
            total_loss += loss_head
            if verbose:
                print(f"head={i} loss={loss_head:.2f}")

        return total_loss

    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        return self.train_loss(X_batch, y_batch, data_size, verbose)

    def forward(self, X):
        """
        Predict probabilities for each output given input data.

        Parameters:
        ----------
        X : torch.Tensor
            Input data. Shape (n_samples, input_dim).

        Returns:
        -------
        preds : torch.Tensor
            Predicted probabilities for each output. Shape (n_samples, K, C).
        """
        X_processed = self.process(X)

        preds = []
        for head in self.heads:
            pred = head(X_processed)
            pred = F.log_softmax(pred, dim=1)
            preds.append(pred)
  
        preds = torch.stack(preds, dim=1)  # Shape: (n_samples, K, C)
        return preds

    def predict(self, X, threshold=None):
        preds = self.forward(X)
        all_preds = []
        for i in range(self.K):
            max_class = torch.argmax(preds[:, i, :], dim=-1)
            all_preds.append(max_class)
        return torch.stack(all_preds, dim=1), preds
    
    def get_confidences(self, preds):
        return torch.max(preds, dim=-1)[0]

    def compute_negative_log_likelihood(self, X_batch, y_batch, mc=True):
        logits = self.forward(X_batch)
        nlls = []
        for val_k in range(self.K):
            relevant_head = val_k
            y = y_batch.T[relevant_head]
            logit = logits[:, relevant_head, :]
            probabilities = F.softmax(logit, dim=1)
            true_class_probs = probabilities.gather(1, y.unsqueeze(1).to(torch.long)).squeeze().to(torch.float)
            log_likelihood = torch.log(true_class_probs)
            nlls.append(-log_likelihood.sum())
            # nll = F.cross_entropy(logit, y.to(torch.long))
        return torch.stack(nlls)
    
""" # Softmax Pointwise CC model """

class SoftmaxPointwiseCC(LLModelCC, SoftmaxPointwise):
    def __init__(self, p, K, beta=0.0, intercept=False, backbone=None, num_classes_lst=None, chain_order=None):
        LLModelCC.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone, chain_order=chain_order)
        SoftmaxPointwise.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone, num_classes_lst=num_classes_lst)
        print(f"[SoftmaxPointwiseCC]")

        if num_classes_lst is None:
            self.num_classes_lst = [2] * self.K  # Default to binary classification for each head
        else:
            self.num_classes_lst = num_classes_lst

        self.heads = nn.ModuleList([self.make_output_layer(in_features=self.p+sum(self.num_classes_lst[:k]), num_classes=self.num_classes_lst[k]) for k in range(self.K)])
        self.heads = nn.ModuleList([self.heads[(self.chain_order == i_k).nonzero().item()] for i_k, val_k in enumerate(self.chain_order)])
        self.loss = nn.CrossEntropyLoss(reduction='mean')

        self.params = nn.ParameterList()
        if self.backbone is not None:
            self.params += list(self.backbone.parameters())
        for head in nn.ModuleList(self.heads):
            self.params += list(head.parameters())

    def make_output_layer(self, num_classes, in_features=None):
        if in_features is None:
            in_features = self.p
        return nn.Linear(in_features, num_classes).to(torch.double)

    def predict(self, X, threshold=None):
        preds = self.forward(X)
        all_preds = []
        for i in range(self.K):
            max_class = torch.argmax(preds[:, i, :], dim=-1)
            all_preds.append(max_class)
        return torch.stack(all_preds, dim=1), preds

    def forward(self, X):
        X_processed = self.process(X)
        logits = []
        for i_k, val_k in enumerate(self.chain_order):
            if i_k == 0:
                logit = self.heads[val_k](X_processed)
            else:
                prev_logits = torch.cat(logits, dim=1)
                logit = self.heads[val_k](torch.cat((X_processed, prev_logits), dim=1))
            logits.append(logit)
        return torch.stack(logits, dim=1)
    
    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        data_size = data_size or X_batch.shape[0]
        X_processed = self.process(X_batch)
        total_loss = 0.
        logits = []
        for i_k, val_k in enumerate(self.chain_order):
            if i_k == 0:
                logit = self.heads[val_k](X_processed)
                loss_head = self.loss(logit, y_batch.T[val_k].to(torch.long)) 
            else:
                prev_logits = torch.cat(logits, dim=1)
                logit = self.heads[val_k](torch.cat((X_processed, prev_logits), dim=1))
                loss_head = self.loss(logit, y_batch.T[val_k].to(torch.long)) 
            logits.append(logit)
            total_loss += loss_head
            if verbose:
                print(f"head={i_k} loss={loss_head:.2f}")
        return total_loss

    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        return self.train_loss(X_batch, y_batch, data_size, verbose)


def create_model(cfg, backbone = None):
    """
    p = cfg.get("p", 64)
    K = cfg.get("K", 6)
    method = cfg.get("method", 0)
    beta = cfg.get("beta", 0.0)
    intercept = cfg.get("intercept", False)
    backbone = cfg.get("backbone", None)
    vbll_cfg = cfg.get("vbll_cfg", None)
    """

    model_classes = {
        "logisticvi": LogisticVI,
        "logisticpointwise": LogisticPointwise,
        "softmaxvbll": SoftmaxVBLL,
        "softmaxpointwise": SoftmaxPointwise,
        "cc_logisticvi": LogisticVICC,
        "cc_logisticpointwise": LogisticPointwiseCC,
        "cc_softmaxvbll": SoftmaxVBLLCC,
        "cc_softmaxpointwise": SoftmaxPointwiseCC,
    }
    if cfg.model_type not in model_classes:
        raise ValueError(f"Unknown model_type={cfg.model_type}")

    #return model_classes[model_type](p, K, method=method, beta=beta, intercept=intercept, backbone=backbone, vbll_cfg=vbll_cfg)
    model_class = model_classes[cfg.model_type]
    model_args = {k: v for k, v in vars(cfg).items() if k in model_class.__init__.__code__.co_varnames}
    print(model_args)
    return model_class(**model_args, backbone=backbone)
