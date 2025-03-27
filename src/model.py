import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from objective import KL, KL_mvn, neg_ELL_MC_MH, neg_ELL_TB_MH, KL_MH, neg_ELL_MC_mvn_MH, neg_ELL_TB_mvn_MH, KL_mvn_MH, neg_ELL_MC, neg_ELL_TB, neg_ELL_MC_mvn, neg_ELL_TB_mvn

"""## Load VBLL"""
def load_vbll(vbll_path):
    sys.path.append(os.path.abspath(vbll_path)) # currently VBLL v0.4.0.1
    try:
        import vbll
        print("vbll found")
    except:
        print("vbll not found")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vbll"])
        import vbll

"""# Base models"""

"""## Base vanilla model"""
class LLModel(nn.Module):
    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None):
        """
        Parameters:
        ----------
            p : int
                Dimensionality of the input features after processing by the backbone network.
            K : int
                Number of outputs (attributes).
            beta: float, optional
                Regularization parameter. Default is 1.0.
            intercept : bool, optional
                Whether to include an intercept term in the model. Default is False.
            backbone : torch.nn.Module, optional
                Backbone network to transform input features. Default is None (no preprocessing).
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
        self.params = self.get_learnable_parameters()
        return p
    
    def get_learnable_parameters(self):
        params = nn.ParameterList([])
        if self.backbone is not None:
            params += list(self.backbone.parameters())
        return params

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

"""## Base CC model"""
class LLModelCC(LLModel):
    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None, chain_order=None, chain_type="logit"):
        """
        Parameters:
        ----------
            p : int
                Dimensionality of the input features after processing by the backbone network.
            K : int
                Number of outputs (attributes).
            beta: float, optional
                Regularization parameter. Default is 1.0.
            intercept : bool, optional
                Whether to include an intercept term in the model. Default is False.
            backbone : torch.nn.Module, optional
                Backbone network to transform input features. Default is None (no preprocessing).
            chain_order : list of int, optional
                Order of the chain. Default is None (from 0 on).
            chain_type : str, optional
                Type of the chain. Default is "logit". Choose from ["logit", "probability", "prediction", "true"].
        """
        LLModel.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone)
        self.chain_order = chain_order if chain_order is not None else list(range(K)) # TODO: [backlog] make graph instead of list
        assert len(self.chain_order) == self.K, f"chain_order must have length {self.K}"
        print("[LLModelCC] chain_order=", self.chain_order)
        self.chain_order = torch.tensor(self.chain_order, dtype=torch.long)
        self.chain_type = chain_type

    def process_chain(self, X_batch):
        raise NotImplementedError("[LLModelCC] process_chain not implemented")

"""## Sigmoid models"""

"""## Sigmoid-pointwise model"""
class LogisticPointwise(LLModel):
    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None, m_init=None, 
                 prior_mu=None, prior_Sig=None, prior_mean_learnable=False, prior_scale_learnable=False):
        """
        Parameters:
        ----------
            p : int
                Dimensionality of the input features after processing by the backbone network.
            K : int
                Number of outputs (attributes).
            beta: float, optional
                Regularization parameter. Default is 1.0.
            intercept : bool, optional
                Whether to include an intercept term in the model. Default is False.
            backbone : torch.nn.Module, optional
                Backbone network to transform input features. Default is None (no preprocessing).
            m_init : torch.Tensor, optional
                Initial means of the variational distributions. Shape (p, K). Default is None (random initialization).
            prior_mu : torch.Tensor, optional
                Prior means for each output. Shape (p, K). Default is None (zero means).
            prior_Sig : list of torch.Tensor, optional
                Prior covariance matrices for each output. List of K tensors, each of shape (p, p). Default is None (identity matrices).
            prior_mean_learnable : bool, optional
                Whether the prior means are learnable. Default is False.
            prior_scale_learnable : bool, optional
                Whether the prior scales are learnable. Default is False.
        """
        p = super().__init__(p, K, beta=beta, intercept=intercept, backbone=backbone)
        print(f"[LogisticPointwise] input_dim={p} output_dim={K} beta={beta}")

        self.loss = nn.BCELoss(reduction='mean')

        if prior_mu is None:
            self.prior_mu_list = [torch.zeros(self.p, dtype=torch.double) for k in range(self.K)]
        else:
            assert isinstance(prior_mu, torch.Tensor), "mu must be a torch.Tensor"
            assert prior_mu.shape[0] == self.K, f"mu must have shape ({self.K}, p)"
            # assert prior_mu.shape[1] == self.p, f"mu must have shape ({self.K}, {self.p})" turn off for CC
            self.prior_mu_list = [prior_mu[k] for k in range(self.K)]

        if prior_Sig is None:
            self.prior_Sig_list = [torch.eye(self.p, dtype=torch.double) for _ in range(self.K)]
        else:
            assert isinstance(prior_Sig, torch.Tensor) or isinstance(prior_Sig, list), "Sig must be a list of tensors"
            if isinstance(prior_Sig, torch.Tensor):
                assert prior_Sig.shape[0] == self.K, f"Sig must have shape ({self.K}, p, p)"
                # assert prior_Sig.shape[1] == self.p, f"Sig must have shape ({self.K}, {self.p}, {self.p})" # turn off for CC
            else:
                assert len(prior_Sig) == self.K, f"Sig must contain {self.K} tensors"
                for i, sig in enumerate(prior_Sig):
                    assert isinstance(sig, torch.Tensor), f"Sig[{i}] must be a torch.Tensor"
                    # assert sig.shape == (self.p, self.p), f"Sig[{i}] must have shape ({self.p}, {self.p})" # turn off for CC
            self.prior_Sig_list = prior_Sig

        if m_init is None:
            self.m_list = [nn.Parameter(torch.randn(self.p, dtype=torch.double)) for _ in range(self.K)]
        else:
            assert isinstance(m_init, torch.Tensor) or isinstance(m_init, list), "m_init must be a torch.Tensor or list of torch.Tensor"
            assert len(m_init) == self.K, f"m_init must contain {self.K} tensors"
            if isinstance(m_init, torch.Tensor):
                assert m_init.shape[0] == self.K, f"m_init must have shape ({self.K}, p)"
                # assert m_init.shape[1] == self.p, f"m_init must have shape ({self.K}, {self.p})" # turn off for CC
            else:
                for i, m in enumerate(m_init):
                    assert isinstance(m, torch.Tensor), f"m_init[{i}] must be a torch.Tensor"
                    assert m.shape == (self.p,), f"m_init[{i}] must have shape ({self.p},)"
            self.m_list = [nn.Parameter(m_init[:, k]) for k in range(self.K)]


        self.params = self.get_learnable_parameters()
    
    def get_learnable_parameters(self):
        params = nn.ParameterList(self.m_list)
        if self.backbone is not None:
            params += list(self.backbone.parameters())
        if self.prior_mean_learnable:
            self.prior_mu_list = [nn.Parameter(mu) for mu in self.prior_mu_list]
            params += nn.ParameterList(self.prior_mu_list)
        if self.prior_scale_learnable:
            self.prior_Sig_list = [nn.Parameter(Sig) for Sig in self.prior_Sig_list]
            params += nn.ParameterList(self.prior_Sig_list)
        return params

    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the training loss [BCE] for a batch of data.
        
        Parameters:
        ----------
        X_batch : torch.Tensor
            Batch of input data. Shape (batch_size, input_dim).
        y_batch : torch.Tensor
            Batch of target variables. Shape (batch_size, K).
        data_size : int, optional
            Total size of the dataset. Default is None (batch size).
        verbose : bool, optional
            Whether to print the loss. Default is False.
        """
        data_size = data_size or X_batch.shape[0]

        preds = self.forward(X_batch)
        assert preds.shape == y_batch.shape, f"preds.shape={preds.shape} (from forward) != y_batch.shape={y_batch.shape} (data)"
        mean_bce = self.loss(preds, y_batch)
        mean_reg = self.regularization() / data_size if self.beta else torch.tensor(0.0, device=mean_bce.device)
        if verbose:
            print(f"mean_bce_loss={mean_bce:.2f} {'mean_reg={mean_reg:.2f}' if self.beta else ''}")
        if self.beta:
            return mean_bce + self.beta * mean_reg
        return mean_bce

    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False, other_beta=None):
        """
        Compute the test loss [BCE] for a batch of data. See train loss."""
        data_size = data_size or X_batch.shape[0]

        preds = self.forward(X_batch)
        assert preds.shape == y_batch.shape, f"preds.shape={preds.shape} != y_batch.shape={y_batch.shape}"
        mean_bce = self.loss(preds, y_batch)

        beta = other_beta or self.beta
        mean_reg = self.regularization() / data_size if beta else torch.tensor(0.0, device=mean_bce.device)

        if verbose:
            print(f"mean_bce_loss={mean_bce:.2f} {'mean_reg={mean_reg:.2f}' if beta else ''}")
        return mean_bce + beta * mean_reg

    def regularization(self):
        """
        Compute the regularization term for the model. This is the KL divergence between the variational distributions and the priors. 
        For pointwise logistic regression, this is the negative log probability of the variational parameters under the prior distribution which is equivalent to L2 regularization.
        
        Returns:
        -------
        log_prob : torch.Tensor
            The computed regularization term. Shape (1).
        """
        log_prob = 0.
        for i, (m, prior_mu, prior_Sig) in enumerate(zip(self.m_list, self.prior_mu_list, self.prior_Sig_list)):
            try:
                d = torch.distributions.MultivariateNormal(loc=prior_mu.to(m.device), covariance_matrix=prior_Sig.to(m.device))
                log_prob += d.log_prob(m)
            except:
                print(f"Error in regularization for output {i}")
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
        for i, m in enumerate(self.m_list):
            pred = torch.sigmoid(X_processed @ m.to(X_processed.device))
            preds.append(pred.unsqueeze(1))

        preds = torch.cat(preds, dim=1)
        assert preds.shape == (X.shape[0], self.K), f"preds.shape={preds.shape} != (X.shape[0], {self.K})"
        return preds
    
    def predict(self, X):
        """
        Predict binary labels for each output given input data.
        
        Parameters:
        ----------
        X : torch.Tensor
            Input data. Shape (n_samples, input_dim).
            
        Returns:
        -------
        predictions : torch.Tensor
            Predicted binary labels for each output. Shape (n_samples, K).
        preds : torch.Tensor
            Predicted probabilities for each output. Shape (n_samples, K).
        """
        preds = self.forward(X)
        return (preds > 0.5).float(), preds

    def compute_negative_log_likelihood(self, X, y, mc = False, n_samples = 1000):
        """
        Compute the negative log likelihood of the data given the predictions.

        Parameters:
        ----------
        X : torch.Tensor
            Predicted probabilities for each output. Shape (n_samples, K).
        y : torch.Tensor
            Target variables. Shape (n_samples, K).
        mc: bool, optional [Dumb argument]
            Whether to use Monte Carlo estimation. Default is False.
        n_samples : int (optional) [Dumb argument]
            Number of samples for Monte Carlo estimation. Default is 1000.

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
        assert nll.shape == (self.K,), f"nll.shape={nll.shape} != (K={self.K})"
        return nll

    def get_confidences(self, preds):
        """
        Compute the confidence of the predictions. This is the maximum of the predicted probability and its complement.
        
        Parameters:
        ----------
        preds : torch.Tensor
            Predicted probabilities for each output. Shape (n_samples, K)."""
        return torch.max(torch.stack([preds, 1 - preds]), dim=0)[0]

"""## Logistic-pointwise CC model """
class LogisticPointwiseCC(LLModelCC, LogisticPointwise):
    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None, m_init=None, 
                 prior_mu=None, prior_Sig=None, prior_mean_learnable=False, prior_scale_learnable=False,
                 chain_order=None, chain_type="logit"):
        """
        Parameters:
        ----------
            p : int
                Dimensionality of the input features after processing by the backbone network.
            K : int
                Number of outputs (attributes).
            beta: float, optional
                Regularization parameter. Default is 1.0.
            intercept : bool, optional
                Whether to include an intercept term in the model. Default is False.
            backbone : torch.nn.Module, optional
                Backbone network to transform input features. Default is None (no preprocessing).
            m_init : torch.Tensor | list, optional
                Initial means of the variational distributions. Shape (K, p | p + K). Default is None (random initialization).
            prior_mu : torch.Tensor, optional
                Prior means for each output. Shape (K, p | p + K). Default is None (zero means).
            prior_Sig : list of torch.Tensor, optional
                Prior covariance matrices for each output. List of K tensors, each of shape (p, p) | (p+K, p+K). Default is None (identity matrices).
            prior_mean_learnable : bool, optional
                Whether the prior means are learnable. Default is False.
            prior_scale_learnable : bool, optional
                Whether the prior scales are learnable. Default is False.
            chain_order : list of int, optional
                Order of the chain. Default is None (from 0 on).
            chain_type : str, optional
                Type of the chain. Default is "logit". Choose from ["logit", "probability", "prediction", "true"]."""
        LLModelCC.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone, chain_order=chain_order, chain_type=chain_type)
        LogisticPointwise.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone, m_init=m_init, 
                                   prior_mu=prior_mu, prior_Sig=prior_Sig, prior_mean_learnable=prior_mean_learnable, prior_scale_learnable=prior_scale_learnable)
        print(f"[LogisticPointwiseCC]")

        if m_init is None:
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
        for m in self.m_list:
            m.requires_grad = True

        self.params = self.get_learnable_parameters()
    
    def forward(self, X_batch):
        X_processed = self.process(X_batch)
        X_processed = X_processed.to(torch.double)
        prev_list = []
        for i, k in enumerate(self.chain_order):
            if i == 0:
                logit = (X_processed @ self.heads[(self.chain_order == i).nonzero().item()]).to(X_processed.device)
                probability = torch.sigmoid(logit)
            else:
                prev_cat = torch.cat(prev_list, dim=1)
                logit = (torch.cat((X_processed, prev_cat), dim=1) @ self.heads[(self.chain_order == i).nonzero().item()]).to(X_processed.device)
                probability = torch.sigmoid(logit)
            if self.chain_type == "logit":
                prev_list.append(logit.unsqueeze(1))
            elif self.chain_type == "probability":
                prev_list.append(probability.unsqueeze(1))
            elif self.chain_type == "prediction":
                prev_list.append((probability > 0.5).float().unsqueeze(1))
            # elif self.chain_type == "true":
            #     prev_list.append(y_batch[:, k].unsqueeze(1))
        
        return torch.cat(probability, dim=1)


"""## Sigmoid-logistic (VI-PER) model"""
class LogisticVI(LLModel):

    @property
    def prior_mu_list(self):
        """
        Return the prior means for each output.
        """
        return self.prior_mu.expand(self.K, self.p)
        # return torch.full((self.K, self.p), self.prior_mu.item(), dtype=torch.double, requires_grad=self.prior_mu.requires_grad)

    @property
    def prior_scale(self):
        """
        Return the prior scale for the standard deviations.
        """
        return torch.exp(self.prior_u_sig)

    @property
    def prior_Sig_list(self):
        """
        Return the prior covariance matrices for each output.
        """
        ps = self.prior_scale
        return [torch.eye(self.p, dtype=torch.double, device=ps.device) * ps for _ in range(self.K)]

    @property
    def s_list(self):
        """
        Return the standard deviations for each output.
        """
        return [torch.exp(u) for u in self.u_list]
    
    @property
    def L_single(self, u):
        """
        Return the covariance matrix for a single output.
        """
        L = torch.zeros(self.p, self.p, dtype=torch.double, device=u.device)
        tril_indices = torch.tril_indices(self.p, self.p, 0).to(u.device)
        L[tril_indices[0], tril_indices[1]] = u
        return L @ L.t()
        

    @property
    def S_list(self):
        """
        Return the covariance matrices for each output.
        """
        return [self.L_single(u) for u in self.u_list]  

    def __init__(self, p, K, method=0, l_max=12.0, adaptive_l=False, n_samples=500, beta=1.0, intercept=False, 
                 prior_mu=None, prior_u_sig=None, prior_mean_learnable=False, prior_scale_init=1.0, prior_scale_learnable=False,
                 m_init=None, s_init=None, posterior_mean_init_scale=1.0, posterior_var_init_add=0.0,
                 incorrect_straight_sigmoid=False, sigmoid_mc_computation=False, sigmoid_mc_n_samples=100,
                 backbone=None):
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
        l_max : float, optional
            Maximum value of l for the proposed bound. Default is 12.0.
        adaptive_l : bool, optional
            Whether to adaptively increase l during training. Default is False.
        n_samples : int, optional
            Number of samples for Monte Carlo estimation. Default is 500.
        beta : float, optional
            Regularization parameter (in ELBO). Default is 1.0.
        intercept : bool, optional
            Whether to include an intercept term in the model. Default is False.
        prior_mu : torch.Tensor, optional
            Prior means for each output. Shape (p, K). Default is None (zero means).
        prior_u_sig : torch.Tensor, optional
            Prior standard deviations for each output. Shape (p,). Default is None (unit standard deviations).
        prior_mean_learnable : bool, optional
            Whether the prior means are learnable. Default is False.
        prior_scale_init : float, optional
            Prior scale for the standard deviations. Default is 1.0.
        prior_scale_learnable : bool, optional
            Whether the prior scales are learnable. Default is False.
        m_init : torch.Tensor, optional
            Initial means of the variational distributions. Shape (p, K). Default is None (random initialization).
        s_init : torch.Tensor, optional
            Initial standard deviations (or lower-triangular parameters) of the variational distributions. Shape depends on method.
        posterior_mean_init_scale : float, optional
            Scale for the random initialization of the variational means. Default is 1.0.
        posterior_var_init_add : float, optional
            Value to add to the initial standard deviations. Default is 0.0.
        incorrect_straight_sigmoid : bool, optional
            Whether to use the incorrect straight-through sigmoid estimator. Default is False (hence use probit https://arxiv.org/pdf/2002.10118)
        sigmoid_mc_computation : bool, optional
            Whether to use Monte Carlo estimation for the sigmoid function. Default is False.
        sigmoid_mc_n_samples : int, optional
            Number of samples for Monte Carlo estimation. Default is 100.
        backbone : torch.nn.Module, optional
            Backbone network to transform input features. Default is None (no preprocessing).
        """
        p = super().__init__(p, K, beta=beta, intercept=intercept, backbone=backbone)
        print(f"[LogisticVI] method={method} l_max={l_max} adaptive_l={adaptive_l} n_samples={n_samples}")

        self.method = method
        self.l_max = l_max
        self.adaptive_l = adaptive_l
        self.n_samples = n_samples

        self.prior_mean_learnable = prior_mean_learnable
        self.prior_scale_learnable = prior_scale_learnable

        self.incorrect_straight_sigmoid = incorrect_straight_sigmoid
        self.sigmoid_mc_computation = sigmoid_mc_computation
        self.sigmoid_mc_n_samples = sigmoid_mc_n_samples

        # Initialize prior parameters
        if prior_mu is None:
            self.prior_mu = nn.Parameter(torch.tensor(0.0, dtype=torch.double), requires_grad=self.prior_mean_learnable)
        else:
            assert isinstance(prior_mu, torch.Tensor), "mu must be a torch.Tensor"
            assert prior_mu.shape == (self.p, self.K), f"mu must have shape ({self.p}, {self.K})"
            self.prior_mu = nn.Parameter(prior_mu, requires_grad=self.prior_mean_learnable)
        if prior_u_sig is None:
            self.prior_u_sig = nn.Parameter(torch.log(torch.tensor(prior_scale_init, dtype=torch.double)), requires_grad=self.prior_scale_learnable)
        else:
            assert isinstance(prior_u_sig, torch.Tensor), "u_sig must be a torch.Tensor"
            assert prior_u_sig.shape == (self.p,), f"u_sig must have shape ({self.p},)"
            self.prior_u_sig = nn.Parameter(prior_u_sig, requires_grad=self.prior_scale_learnable)

        # Initialize variational parameters
        if m_init is None:
            self.m_list = [torch.randn(self.p, dtype=torch.double) * posterior_mean_init_scale for _ in range(K)]
        else:
            assert isinstance(m_init, torch.Tensor), "m_init must be a torch.Tensor"
            assert m_init.shape == (self.K, self.p), f"m_init must have shape ({self.K}, {self.p})"
            self.m_list = [m_init[val_k, :] for val_k in range(self.K)]

        if s_init is None:
            if method in [0, 4]:
                self.u_list = [torch.tensor([-1.0 + posterior_var_init_add] * self.p, dtype=torch.double) for _ in range(self.K)]
                # self.s_list = [torch.exp(u) for u in self.u_list]
            elif method in [1, 5]:
                self.u_list = []
                for _ in range(self.K):
                    u = torch.ones(int(self.p * (self.p + 1) / 2), dtype=torch.double) * (1.0 / self.p)
                    u.requires_grad = True
                    self.u_list.append(u)
        else:
            if method in [0, 4]:
                assert isinstance(s_init, torch.Tensor), "s_init must be a torch.Tensor"
                assert s_init.shape == (self.K, self.p), f"s_init must have shape ({self.K}, {self.p})"
                self.s_list = [s_init[i_k, :] for i_k in range(self.K)]
                self.u_list = [torch.log(s) for s in self.s_list]
            elif method in [1, 5]:
                assert isinstance(s_init, torch.Tensor), "s_init must be a torch.Tensor"
                assert s_init.shape == (self.K, self.p * (self.p + 1) // 2), f"s_init must have shape ({self.K}, {self.p * (self.p + 1) // 2})"
                self.u_list = [s_init[i_k, :] for i_k in range(self.K)]

        # Set requires_grad=True for variational parameters
        for m in self.m_list:
            m.requires_grad = True
        for u in self.u_list:
            u.requires_grad = True

        self.params = self.get_learnable_parameters()

        # Initialize l_terms for adaptive l
        if adaptive_l:
            self.l_terms = float(int(l_max / 2))
        else:
            self.l_terms = l_max

    def get_learnable_parameters(self):
        params = nn.ParameterList(list(self.m_list) + list(self.u_list))
        if self.prior_mean_learnable:
            params.append(self.prior_mu)
        if self.prior_scale_learnable:
            params.append(self.prior_u_sig)
        if self.backbone is not None:
            params += list(self.backbone.parameters())
        return params

    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the training loss [ELBO] for a batch of data. Reference to compute_ELBO
        
        Parameters:
        ----------
        X_batch : torch.Tensor
            Batch of input data. Shape (batch_size, input_dim).
        y_batch : torch.Tensor
            Batch of target variables. Shape (batch_size, K).
        data_size : int, optional
            Total size of the dataset. Default is None (batch size).
        verbose : bool, optional
            Whether to print the loss. Default is False."""
        data_size = data_size or X_batch.shape[0]
        return -self.compute_ELBO(X_batch, y_batch, data_size, verbose=verbose)

    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the test loss [ELBO] for a batch of data. See train loss. Beta is set to 0.0.
        
        Parameters:
        ----------
        X_batch : torch.Tensor
            Batch of input data. Shape (batch_size, input_dim).
        y_batch : torch.Tensor
            Batch of target variables. Shape (batch_size, K).
        data_size : int, optional
            Total size of the dataset. Default is None (batch size).
        verbose : bool, optional
            Whether to print the loss. Default is False."""
        data_size = data_size or X_batch.shape[0]
        return -self.compute_ELBO(X_batch, y_batch, data_size, verbose=verbose, other_beta=0.0)

    def compute_ELBO(self, X_batch, y_batch, data_size, verbose=False, other_beta=None):
        """
        Compute the Evidence Lower Bound (ELBO) for a batch of data. Reference to objective.py

        Parameters:
        ----------
        X_batch : torch.Tensor
            Batch of input data. Shape (batch_size, input_dim).
        y_batch : torch.Tensor
            Batch of target variables. Shape (batch_size, K).
        data_size : int
            Total size of the dataset.
        verbose : bool, optional
            Whether to print the loss. Default is False.
        other_beta : float, optional
            Regularization parameter. Default is None (use self.beta).

        Returns:
        -------
        ELBO : torch.Tensor
            The computed ELBO for the batch.
        """
        X_processed = self.process(X_batch)
        batch_size = X_batch.shape[0]

        m_list = [m.to(X_batch.device) for m in self.m_list]
        prior_mu_list = [mu.to(X_batch.device) for mu in self.prior_mu_list]
        y_list = [y_batch[:, val_k] for i_k, val_k in enumerate(range(self.K))]

        if self.method in [0, 4]:
            s_list = [s.to(X_batch.device) for s in self.s_list]
            prior_Sig_list = [sig.to(X_batch.device) for sig in self.prior_Sig_list]

            if self.method == 0:
                likelihood = -neg_ELL_TB_MH(m_list, s_list, y_list, X_processed, l_max=self.l_terms)
                KL_div = KL_MH(m_list, s_list, prior_mu_list, prior_Sig_list)
            else:
                likelihood = -neg_ELL_MC_MH(m_list, s_list, y_list, X_processed.to(X_batch.device), n_samples=self.n_samples)
                KL_div = KL_MH(m_list, s_list, prior_mu_list, prior_Sig_list)

        elif self.method in [1, 5]:
            S_list = [S.to(X_batch.device) for S in self.S_list]
            prior_Sig_list = [Sig.to(X_batch.device) for Sig in self.prior_Sig_list]
            if self.method == 1:
                likelihood = -neg_ELL_TB_mvn_MH(m_list, S_list, y_list, X_processed, l_max=self.l_terms)
                KL_div = KL_mvn_MH(m_list, S_list, prior_mu_list, prior_Sig_list)
            else:
                likelihood = -neg_ELL_MC_mvn_MH(m_list, S_list, y_list, X_processed, n_samples=self.n_samples)
                KL_div = KL_mvn_MH(m_list, S_list, prior_mu_list, self.prior_Sig_list)

        else:
            raise ValueError("Method not recognized")

        mean_log_lik = likelihood/batch_size
        mean_kl_div = KL_div/data_size
        beta = other_beta or self.beta
        ELBO = mean_log_lik - beta*mean_kl_div
        if verbose:
            print(f"ELBO={ELBO:.2f} mean_log_lik={mean_log_lik:.2f} mean_kl_div={mean_kl_div:.2f}")
        assert ELBO.shape == torch.Size([1]), f"ELBO.shape={ELBO.shape} != (1)"
        return ELBO

    def compute_negative_log_likelihood(self, X, y, mc = False, n_samples = 1000):
        """
        Compute the negative log likelihood of the data given the predictions. Reference to objective.py
        
        Parameters:
        ----------
        X : torch.Tensor
            Predicted probabilities for each output. Shape (n_samples, K).
        y : torch.Tensor
            Target variables. Shape (n_samples, K).
        mc: bool, optional
            Whether to use Monte Carlo estimation. Default is False.
        n_samples : int, optional
            Number of samples for Monte Carlo estimation. Default is 1000.

        Returns:
        -------
        nlls : list of torch.Tensor
            The computed negative log likelihood for each attribute. Shape (K,)
        """
        X_processed = self.process(X)
        m_list = [m.to(X.device) for m in self.m_list]
        y_list = [y[:, i_k] for i_k in range(self.K)]
        if self.method in [0, 4]:
            s_list = [s.to(X.device) for s in self.s_list]
            if mc:
                nlls = [neg_ELL_MC(m, s, y, X_processed, n_samples=n_samples) for m, s, y in zip(m_list, s_list, y_list)]
            else:
                nlls = [neg_ELL_TB(m, s, y, X_processed, l_max=self.l_terms) for m, s, y in zip(m_list, s_list, y_list)]
        elif self.method in [1, 5]:
            S_list = [S.to(X.device) for S in self.S_list]
            if mc:
                nlls = [neg_ELL_MC_mvn(m, S, y, X_processed, n_samples=n_samples) for m, S, y in zip(m_list, S_list, y.T)]
            else:
                nlls = [neg_ELL_TB_mvn(m, S, y, X_processed, l_max=self.l_terms) for m, S, y in zip(m_list, S_list, y.T)]
        assert len(nlls) == self.K, f"nlls must have length {self.K}"
        return torch.tensor(nlls)

    def get_confidences(self, preds):
        """
        Compute the confidence of the predictions. This is the maximum of the predicted probability and its complement.
        
        Parameters:
        ----------
        preds : torch.Tensor
            Predicted probabilities for each output. Shape (n_samples, K).
        """
        return torch.max(torch.stack([preds, 1 - preds]), dim=0)[0]

    @torch.no_grad
    def expected_sigmoid_multivariate(self, X, m, u, mc=False, n_samples=None):
        """
        Compute the expected sigmoid function for a multivariate normal distribution. 
        This works for both diagonal and full covariance matrices, but only for one output at a time.
        Reference to objective.py (142) and probit (https://arxiv.org/pdf/2002.10118)
        
        Parameters:
        ----------
        X : torch.Tensor
            Input data. Shape (n_samples, input_dim).
        m : torch.Tensor
            Mean of the distribution. Shape (input_dim).
        u : torch.Tensor
            Standard deviations of the distribution. Shape (input_dim).
        mc : bool, optional
            Whether to use Monte Carlo estimation. Default is False.
        n_samples : int, optional
            Number of samples for Monte Carlo estimation. Default is None.
        
        Returns:
        -------
        after_activation : torch.Tensor
            The expected sigmoid function for the distribution. Shape (n_samples).
        before_activation : torch.Tensor
            Before applying the sigmoid function. Not working properly for MC. Shape (n_samples).
        """
        assert (not mc or n_samples is not None), "n_samples must be provided for Monte Carlo estimation"
        m = m.to(X.device)
        u = u.to(X.device)
        
        M = X @ m # just take mean of the distribution
        if self.incorrect_straight_sigmoid:
            return torch.sigmoid(M), M

        if self.method in [0, 4]:
            s = torch.exp(u)
            if not mc: # probit approximation
                scaling_factor_diag = torch.einsum("bi,i,bi->b", X, s**2, X)
                assert scaling_factor_diag.shape == torch.Size([X.shape[0]])
                scaling_factor = torch.sqrt(1 + (torch.pi / 8) * scaling_factor_diag)
                M_corrected = M / scaling_factor
                expected_sigmoid = torch.sigmoid(M_corrected)
            else: # ref objective.py 142
                S = torch.sqrt(torch.sum(X**2 * s**2, dim=1))
                S = torch.sqrt(S)
                norm = torch.distributions.Normal(loc=M, scale=S)
                samples = norm.rsample(n_samples)
                sigmoid_samples = torch.sigmoid(samples)
                expected_sigmoid = sigmoid_samples.mean(dim=0)
                M_corrected = M

        elif self.method in [1, 5]:
            cov = self.L_single(u)
            if not mc: # probit approximation
                scaling_factor_nondiag = torch.einsum("bi,ij,bj->b", X, cov, X)
                assert scaling_factor_nondiag.shape == torch.Size([X.shape[0]])
                scaling_factor = 1 / torch.sqrt(1 + (torch.pi / 8) * scaling_factor_nondiag)
                M_corrected = M * scaling_factor
                expected_sigmoid = torch.sigmoid(M_corrected)
            else: #TODO: check this later
                mvn = torch.distributions.MultivariateNormal(
                    loc=M, covariance_matrix=cov
                )
                samples = mvn.rsample(n_samples)
                sigmoid_samples = torch.sigmoid(samples)
                expected_sigmoid = sigmoid_samples.mean(dim=0)
                M_corrected = M

        assert expected_sigmoid.shape == torch.Size([X.shape[0]])
        assert M_corrected.shape == torch.Size([X.shape[0]])
        return expected_sigmoid, M_corrected

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
            probs, logits = self.expected_sigmoid_multivariate(X_processed, m, u, mc=self.sigmoid_mc_computation, n_samples=self.sigmoid_mc_n_samples)
            preds.append(probs.unsqueeze(1))

        preds = torch.cat(preds, dim=1)
        assert preds.shape == (X.shape[0], self.K), f"preds.shape={preds.shape} != (X.shape[0], {self.K})"
        return preds

"""## Sigmoid-logistic CC (VI-PER) model"""
class LogisticVICC(LLModelCC, LogisticVI):
    
    @property
    def prior_mu_list(self):
        """
        Return the prior means for each output. Depending on the chain order the dimensionality differs.
        """
        return [self.prior_mu.expand(self.p + val_k) for _, val_k in enumerate(self.chain_order)]

    @property
    def prior_Sig_list(self):
        """
        Return the prior covariance matrices for each output. Depending on the chain order the dimensionality differs.
        """
        ps = self.prior_scale
        return [ps.expand(self.p + val_k) for _, val_k in enumerate(self.chain_order)]

    @property
    def s_list(self):
        """
        Return the standard deviations for each output.
        """
        return [torch.exp(u) for u in self.u_list]
    
    @property
    def L_single(self, u, i_k):
        """
        Return the covariance matrix for a single output.
        """
        L = torch.zeros(self.p + self.chain_order[i_k], self.p + self.chain_order[i_k], dtype=torch.double, device=u.device)
        tril_indices = torch.tril_indices(self.p + self.chain_order[i_k], self.p + self.chain_order[i_k], 0).to(u.device)
        L[tril_indices[0], tril_indices[1]] = u
        return L @ L.t()

    def __init__(self, p, K, method=0, l_max=12.0, adaptive_l=False, n_samples=500, beta=1.0, intercept=False, backbone=None, 
                 prior_mu=None, prior_u_sig=None, prior_mean_learnable=False, prior_scale_init=1.0, prior_scale_learnable=False,
                 m_init=None, s_init=None,
                 posterior_mean_init_scale=1.0, posterior_var_init_add=0.0,
                 incorrect_straight_sigmoid=False, sigmoid_mc_computation=False, sigmoid_mc_n_samples=100,
                 chain_order=None, chain_type="logit"):
        LLModelCC.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone, chain_order=chain_order, chain_type=chain_type)
        LogisticVI.__init__(self, p, K, method=method, l_max=l_max, adaptive_l=adaptive_l, n_samples=n_samples, beta=beta, intercept=intercept,  
                            prior_mu=prior_mu, prior_u_sig=prior_u_sig, prior_mean_learnable=prior_mean_learnable, prior_scale_init=prior_scale_init, prior_scale_learnable=prior_scale_learnable,
                            m_init=m_init, s_init=s_init, posterior_mean_init_scale=posterior_mean_init_scale, posterior_var_init_add=posterior_var_init_add,
                            incorrect_straight_sigmoid=incorrect_straight_sigmoid, sigmoid_mc_computation=sigmoid_mc_computation, sigmoid_mc_n_samples=sigmoid_mc_n_samples,
                            backbone=backbone)
        print(f"[LogisticVICC]")

        # self.prior_mu = nn.Parameter(torch.tensor(0.0, dtype=torch.double), requires_grad=prior_mean_learnable)
        # self.prior_u_sig = nn.Parameter(torch.log(torch.tensor(prior_scale_init, dtype=torch.double)), requires_grad=prior_scale_learnable)

        if m_init is None:
            m_list = [torch.randn(self.p+val_k, dtype=torch.double) for _, val_k in enumerate(self.chain_order)]
        else:
            if isinstance(m_init, torch.Tensor):
                assert m_init.shape == (self.K, self.p) or m_init.shape == (self.K, self.p + self.K)
                m_list = []
                for i_k in range(self.K):
                    m = m_init[i_k]
                    if m.shape[0] == self.p:
                        extended_m = torch.randn(self.p + self.chain_order[i_k], dtype=torch.double)
                        extended_m[:self.p] = m
                        m_list.append(nn.Parameter(extended_m))
                    elif m.shape[0] == self.p + self.K:
                        m_list.append(nn.Parameter(m[:self.p + self.chain_order[i_k]]))
                    else:
                        raise ValueError(f"m_init[{i_k}] has wrong shape {m.shape}")
            elif isinstance(m_init, list):
                m_list = []
                assert len(m_init) == self.K
                for i_k, m in enumerate(m_init):
                    if m.shape[0] == self.p + self.chain_order[i_k]:
                        m_list.append(nn.Parameter(m))
                    elif m.shape[0] == self.p:
                        extended_m = torch.randn(self.p + self.chain_order[i_k], dtype=torch.double)
                        extended_m[:self.p] = m
                        m_list.append(nn.Parameter(extended_m))
                    else:
                        raise ValueError(f"m_init[{i_k}] has wrong shape {m.shape}")
        self.m_list = m_list

        if s_init is None:
            if method in [0, 4]:
                self.u_list = [torch.tensor([-1.0 + posterior_var_init_add] * (self.p+self.chain_order[i_k]), dtype=torch.double) for i_k in range(self.K)]
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
                    assert s_init.shape[0] == self.K, f"s_init.shape[0]={s_init.shape[0]} != {self.K}"
                    for i_k in range(s_init.shape[0]):
                        s = s_init[i_k]
                        if s.shape[0] == self.p:
                            extended_s = torch.randn(self.p + self.chain_order[i_k], dtype=torch.double)
                            extended_s[:self.p] = s
                            s_list.append(extended_s)
                        elif s.shape[0] == self.p + self.chain_order[i_k]:
                            s_list.append(s)
                        else:
                            raise ValueError(f"s_init[{i_k}] has wrong shape {s.shape}")
                elif isinstance(s_init, list):
                    assert len(s_init) == self.K
                    for i_k, s in enumerate(s_init):
                        if s.shape[0] == self.p + self.chain_order[i_k]:
                            s_list.append(s)
                        elif s.shape[0] == self.p:
                            extended_s = torch.randn(self.p + self.chain_order[i_k], dtype=torch.double)
                            extended_s[:self.p] = s
                            s_list.append(extended_s)
                        else:
                            raise ValueError(f"s_init[{i_k}] has wrong shape {s.shape}")
                self.s_list = s_list
                self.u_list = [torch.log(s) for s in self.s_list]
            elif method in [1, 5]:
                s_list = []
                if isinstance(s_init, torch.Tensor):
                    assert s_init.shape[0] == self.K, f"s_init.shape[0]={s_init.shape[0]} != {self.K}"
                    for i_k in range(s_init.shape[0]):
                        s = s_init[i_k]
                        if len(s) == int(self.p * (self.p + 1) / 2):
                            extended_s = torch.ones(int((self.p + self.chain_order[i_k]) * (self.p + self.chain_order[i_k] + 1) / 2), dtype=torch.double) * (1.0 / (self.p + self.chain_order[i_k]))
                            extended_s[:len(s)] = s
                            s_list.append(extended_s)
                        elif len(s) == int((self.p + self.chain_order[i_k]) * (self.p + self.chain_order[i_k] + 1) / 2):
                            s_list.append(s)
                        else:
                            raise ValueError(f"s_init[{i_k}] has wrong shape {s.shape}")
                elif isinstance(s_init, list):
                    assert len(s_init) == self.K
                    for i_k, s in enumerate(s_init):
                        if len(s) == int((self.p + self.chain_order[i_k]) * (self.p + self.chain_order[i_k] + 1) / 2):
                            s_list.append(s)
                        elif len(s) == int(self.p * (self.p + 1) / 2):
                            extended_s = torch.ones(int((self.p + self.chain_order[i_k]) * (self.p + self.chain_order[i_k] + 1) / 2), dtype=torch.double) * (1.0 / (self.p + self.chain_order[i_k]))
                            extended_s[:len(s)] = s
                            s_list.append(extended_s)
                        else:
                            raise ValueError(f"s_init[{i_k}] has wrong shape {s.shape}")
                self.u_list = s_list # ??

        # Set requires_grad=True for variational parameters
        for m in self.m_list:
            m.requires_grad = True
        for u in self.u_list:
            u.requires_grad = True

        self.params = self.get_learnable_parameters()

    def get_learnable_parameters(self):
        params = nn.ParameterList(self.m_list + self.u_list)
        if self.prior_mean_learnable:
            params.append(self.prior_mu)
        if self.prior_scale_learnable:
            params.append(self.prior_u_sig)
        if self.backbone is not None:
            params += list(self.backbone.parameters())
        return params

    def forward(self, X_batch):
        """
        Predict probabilities for each output given input data.

        Parameters:
        ----------
        X_batch : torch.Tensor
            Input data. Shape (batch_size, input_dim).
        """
        X_processed = self.process(X_batch)
        X_processed = X_processed.to(torch.double)

        preds = []
        prev_list = []
        for i_k, val_k in enumerate(self.chain_order):
            i_relevant = (self.chain_order == i_k).nonzero().item()
            if i_k == 0:
                X = X_processed
            else:
                prev_cat = torch.cat(prev_list, dim=1)
                X = torch.cat((X_processed, prev_cat), dim=1)
            probability, logit = self.expected_sigmoid_multivariate(X, self.m_list[i_relevant], self.u_list[i_relevant].to(X_processed.device), mc=self.sigmoid_mc_computation, n_samples=self.sigmoid_mc_n_samples)
            preds.append(probability.unsqueeze(1))
            if self.chain_type == "logit":
                prev_list.append(logit.unsqueeze(1))
            elif self.chain_type == "probability":
                prev_list.append(probability.unsqueeze(1))
            elif self.chain_type == "prediction":
                prev_list.append((probability > 0.5).float().unsqueeze(1))
            # elif self.chain_type == "true":
            #     prev_list.append(y_batch[:, k].unsqueeze(1))
        out = torch.cat(preds, dim=1)
        assert out.shape == (X_batch.shape[0], self.K), f"out.shape={out.shape} != (X_batch.shape[0], {self.K})"
        return out

    def compute_ELBO(self, X_batch, y_batch, data_size, verbose=False, other_beta=None):
        """
        Compute the Evidence Lower Bound (ELBO) for a batch of data. Reference to objective.py
        
        Parameters:
        ----------
        X_batch : torch.Tensor
            Batch of input data. Shape (batch_size, input_dim).
        y_batch : torch.Tensor
            Batch of target variables. Shape (batch_size, K).
        data_size : int
            Total size of the dataset.
        verbose : bool, optional
            Whether to print the loss. Default is False.
        other_beta : float, optional
            Regularization parameter. Default is None (use self.beta).
        """
        X_processed = self.process(X_batch)
        batch_size = X_batch.shape[0]

        m_list = [m.to(X_batch.device) for m in self.m_list]
        prior_mu_list = [mu.to(X_batch.device) for mu in self.prior_mu_list]
        y_list = [y_batch[:, k] for k in range(self.K)]

        likelihood = torch.tensor(0.0, dtype=torch.double, device=X_batch.device)
        KL_div = torch.tensor(0.0, dtype=torch.double, device=X_batch.device)

        prev_list = []
        for i_k, val_k in enumerate(self.chain_order):
            i_relevant = (self.chain_order == i_k).nonzero().item()
            if i_k == 0:
                X = X_processed
            else:
                X = torch.cat((X_processed, torch.cat(prev_list, dim=1)), dim=1)
            probability, logit = self.expected_sigmoid_multivariate(X, m_list[i_relevant], self.u_list[i_relevant], mc=self.sigmoid_mc_computation, n_samples=self.sigmoid_mc_n_samples)
            if self.chain_type == "logit":
                prev_list.append(logit.unsqueeze(1))
            elif self.chain_type == "probability":
                prev_list.append(probability.unsqueeze(1))
            elif self.chain_type == "prediction":
                prev_list.append((probability > 0.5).float().unsqueeze(1))
            # elif self.chain_type == "true":
            #     prev_list.append(y_batch[:, k].unsqueeze(1))

            if self.method in [0, 4]:
                s = torch.exp(self.u_list[i_relevant].to(X_batch.device))
                sig = self.prior_Sig_list[i_relevant].to(X_batch.device)
                if self.method == 0:
                    likelihood += -neg_ELL_TB(m_list[i_relevant], s, y_list[i_relevant], X, l_max=self.l_terms)
                    KL_div += KL(m_list[i_relevant], s, prior_mu_list[i_relevant], sig)
                else:
                    likelihood += -neg_ELL_MC(m_list[i_relevant], s, y_list[i_relevant], X, n_samples=self.n_samples)
                    KL_div += KL(m_list[i_relevant], s, prior_mu_list[i_relevant], sig)

            elif self.method in [1, 5]:

                u = self.u_list[i_relevant].to(X_batch.device)
                S = self.L_single(u, i_relevant).to(X_batch.device)
                Sig = self.prior_Sig_list[i_relevant].to(X_batch.device)
                if self.method == 1:
                    likelihood += -neg_ELL_TB_mvn(m_list[i_relevant], S, y_list[i_relevant], X, l_max=self.l_terms)
                    KL_div += KL_mvn(m_list[i_relevant], S, prior_mu_list[i_relevant], Sig)
                else:
                    likelihood += -neg_ELL_MC_mvn(m_list[i_relevant], S, y_list[i_relevant], X, n_samples=self.n_samples)
                    KL_div += KL_mvn(m_list[i_relevant], S, prior_mu_list[i_relevant], Sig)
            else:
                raise ValueError("Method not recognized")

        mean_log_lik = likelihood/batch_size
        mean_kl_div = KL_div/data_size
        beta = other_beta or self.beta
        ELBO = mean_log_lik - beta*mean_kl_div
        if verbose:
            print(f"ELBO={ELBO:.2f} mean_log_lik={mean_log_lik:.2f} mean_kl_div={mean_kl_div:.2f}")
        return ELBO

    def compute_negative_log_likelihood(self, X_batch, y_batch, mc = False, n_samples = 1000):
        """
        Compute the negative log likelihood of the data given the predictions. Reference to objective.py
        
        Parameters:
        ----------
        X_batch : torch.Tensor
            Batch of input data. Shape (batch_size, input_dim).
        y_batch : torch.Tensor
            Batch of target variables. Shape (batch_size, K).
        mc: bool, optional
            Whether to use Monte Carlo estimation. Default is False.
        n_samples : int, optional
            Number of samples for Monte Carlo estimation. Default is 1000.
        """
        X_processed = self.process(X_batch)
        m_list = [m.to(X.device) for m in self.m_list]
        y_list = [y_batch[:, k] for k in range(self.K)]
        likelihood = []

        prev_list = []
        for i_k, val_k in enumerate(self.chain_order):
            i_relevant = (self.chain_order == i_k).nonzero().item()
            if i_k == 0:
                X = X_processed
            else:
                X = torch.cat((X_processed, torch.cat(prev_list, dim=1)), dim=1)
            probability, logit = self.expected_sigmoid_multivariate(X, m_list[i_relevant], self.u_list[i_relevant].to(X.device), mc=self.sigmoid_mc_computation, n_samples=self.sigmoid_mc_n_samples)
            if self.chain_type == "logit":
                prev_list.append(logit.unsqueeze(1))
            elif self.chain_type == "probability":
                prev_list.append(probability.unsqueeze(1))
            elif self.chain_type == "prediction":
                prev_list.append((probability > 0.5).float().unsqueeze(1))
            # elif self.chain_type == "true":
            #     prev_list.append(y_batch[:, k].unsqueeze(1))

            if self.method in [0, 4]:
                s = torch.exp(self.u_list[i_relevant].to(X.device))
                if mc:
                    cur_likelihood = -neg_ELL_MC(m_list[i_relevant], s, y_list[i_relevant], X, n_samples=n_samples)
                else:
                    cur_likelihood = -neg_ELL_TB(m_list[i_relevant], s, y_list[i_relevant], X, l_max=self.l_terms)

            elif self.method in [1, 5]:
                u = self.u_list[i_relevant].to(X.device)
                S = self.L_single(u, i_relevant).to(X.device)
                if mc:
                    cur_likelihood = -neg_ELL_MC_mvn(m_list[i_relevant], S, y_list[i_relevant], X, n_samples=n_samples)
                else:
                    cur_likelihood = -neg_ELL_TB_mvn(m_list[i_relevant], S, y_list[i_relevant], X, l_max=self.l_terms)
            else:
                raise ValueError("Method not recognized")
            assert cur_likelihood.shape == torch.Size([1]), f"cur_likelihood.shape={cur_likelihood.shape} != (1)"
            likelihood.append(cur_likelihood)
        assert len(likelihood) == self.K, f"likelihood must have length {self.K}"
        return torch.tensor(likelihood)

""" # Softmax models """

"""## Softmax-pointwise model"""
class SoftmaxPointwise(LLModel):
    def __init__(self, p, K, beta=0.0, num_classes_lst=None, intercept=False, backbone=None):
        """
        Initialize the Softmax-pointwise model.
        
        Parameters:
        ----------
        p : int
            Input dimensionality.
        K : int
            Number of outputs.
        beta : float, optional
            Regularization parameter. Default is 0.0. For pointwise variant equivalent to L2 regularization.
        num_classes_lst : list of int, optional
            Number of classes for each output. Default is None (binary classification). Currently works for fixed common number of classes.
        intercept : bool, optional
            Whether to include an intercept term. Default is False.
        backbone : nn.Module, optional
            Backbone model. Default is None.
        """
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

        self.params = self.get_learnable_parameters()

    def get_learnable_parameters(self):
        params = []
        if self.backbone is not None:
            params += list(self.backbone.parameters())
        for head in self.heads:
            params += list(head.parameters())
        return nn.ParameterList(params)

    def make_output_layer(self, num_classes):
        return nn.Linear(self.p, num_classes).to(torch.double)

    # TODO: regularization
    # def regularization(self):
    #     """
    #     Compute the L2 regularization term.
    #     """
    #     reg = 0.0
    #     for head in self.heads:
    #            for param in head.parameters():
    #                 log_prob += torch.sum(param**2)
    #     return reg

    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the loss for a batch of data. Use CrossEntropy for pointwise classification.
        """
        data_size = data_size or X_batch.shape[0]
        X_processed = self.process(X_batch)

        total_loss = 0.0

        for i, (head,y) in enumerate(zip(self.heads,y_batch.T)):
            pred = head(X_processed)
            loss_head = self.loss(pred, y.to(torch.long))
            assert loss_head.shape == torch.Size([1]), f"loss_head.shape={loss_head.shape} != (1)"
            total_loss += loss_head
            if verbose:
                print(f"head={i} loss={loss_head:.2f}")

        reg_loss = self.regularization()
        if verbose and self.beta:
            print(f"reg_loss={reg_loss:.2f}")
        total_loss += self.beta * reg_loss

        return total_loss

    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False, other_beta=None):
        """
        Compute the loss for a batch of data. Reference to train_loss.
        """
        return self.train_loss(X_batch, y_batch, data_size, verbose)

    def forward(self, X_batch):
        """
        Compute logits for each output given input data.

        Parameters:
        ----------
        X_batch : torch.Tensor
            Input data. Shape (n_samples, input_dim).

        Returns:
        -------
        preds : torch.Tensor
            Predicted probabilities for each output. Shape (n_samples, K, C).
        """
        X_processed = self.process(X_batch)

        logits = []
        for head in self.heads:
            logit = head(X_processed)
            logits.append(logit)
            
        logits = torch.stack(logits, dim=1)
        assert logits.shape == (X_batch.shape[0], self.K, self.num_classes_single), f"logits.shape={logits.shape} != (X_batch.shape[0], {self.K}, {self.num_classes_single})"
        return logits

    def predict(self, X_batch, threshold=None):
        """
        Predict the class for each output given input data.
        
        Parameters:
        ----------
        X_batch : torch.Tensor
            Input data. Shape (n_samples, input_dim).
        threshold : float, optional [inactive]
            Threshold for binary classification. Default is None."""
        logits = self.forward(X_batch)
        all_preds = []
        for i in range(self.K):
            max_class = torch.argmax(logits[:, i, :], dim=-1)
            assert max_class.shape == torch.Size([X_batch.shape[0]])
            all_preds.append(max_class)
        out = torch.stack(all_preds, dim=1)
        assert out.shape == (X_batch.shape[0], self.K), f"out.shape={out.shape} != (X_batch.shape[0], {self.K})"
        return out, logits
    
    def get_confidences(self, preds):
        return torch.max(preds, dim=-1)[0]

    def compute_negative_log_likelihood(self, X_batch, y_batch, mc=True):
        """
        Compute the negative log likelihood of the data given the predictions. Compute the negative log likelihood for each output.
        Likelihood is computed using CrossEntropy loss.
        
        Parameters:
        ----------
        X_batch : torch.Tensor
            Input data. Shape (n_samples, input_dim).
        y_batch : torch.Tensor
            Target variables. Shape (n_samples, K).
        mc : bool, optional
            Whether to use Monte Carlo estimation. Default is True. [inactive]
        """
        logits = self.forward(X_batch)
        nlls = []
        for val_k in range(self.K):
            relevant_head = val_k
            y = y_batch.T[relevant_head]
            logit = logits[:, relevant_head, :]
            probabilities = F.softmax(logit, dim=1)
            true_class_probs = probabilities.gather(1, y.unsqueeze(1).to(torch.long)).squeeze().to(torch.float)
            assert true_class_probs.shape == torch.Size([X_batch.shape[0]]), f"true_class_probs.shape"
            log_likelihood = torch.log(true_class_probs)
            nll = -log_likelihood.sum()
            assert nll.shape == torch.Size([1]), f"nll.shape={nll.shape} != (1)"
            nlls.append(nll)
        return torch.stack(nlls)


""" # Softmax-pointwise CC model """
class SoftmaxPointwiseCC(LLModelCC, SoftmaxPointwise):
    def __init__(self, p, K, beta=0.0, intercept=False, backbone=None, num_classes_lst=None, chain_order=None, chain_type="logit"):
        """
        Initialize the Softmax-pointwise model with chain structure. Reference to SoftmaxPointwise.
        
        Parameters:
        ----------
        p : int
            Input dimensionality.
        K : int
            Number of outputs.
        beta : float, optional
            Regularization parameter. Default is 0.0. For pointwise variant equivalent to L2 regularization.
        intercept : bool, optional
            Whether to include an intercept term. Default is False.
        backbone : nn.Module, optional
            Backbone model. Default is None.
        num_classes_lst : list of int, optional
            Number of classes for each output. Default is None (binary classification). Currently works for fixed common number of classes.
        chain_order : list of int, optional
            Chain order for the outputs. Default is None (no chain structure).
        chain_type : str, optional
            Type of the chain structure. Default is "logit".
        """
        LLModelCC.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone, chain_order=chain_order, chain_type=chain_type)
        SoftmaxPointwise.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone, num_classes_lst=num_classes_lst)
        print(f"[SoftmaxPointwiseCC] chain_order={self.chain_order} chain_type={self.chain_type}")

        if num_classes_lst is None:
            self.num_classes_lst = [2] * self.K  # Default to binary classification for each head
        else:
            self.num_classes_lst = num_classes_lst

        self.heads = nn.ModuleList([self.make_output_layer(in_features=self.p+sum(self.num_classes_lst[:k]), num_classes=self.num_classes_lst[k]) for k in range(self.K)])
        self.heads = nn.ModuleList([self.heads[(self.chain_order == i_k).nonzero().item()] for i_k, val_k in enumerate(self.chain_order)])
        self.loss = nn.CrossEntropyLoss(reduction='mean')

        self.params = self.get_learnable_parameters()

    def make_output_layer(self, num_classes, in_features=None):
        if in_features is None:
            in_features = self.p
        return nn.Linear(in_features, num_classes).to(torch.double)

    def forward(self, X_batch):
        """
        Pass data through the model and return the logits.

        Parameters:
        ----------
        X_batch : torch.Tensor
            Input data. Shape (n_samples, input_dim).
        
        """
        X_processed = self.process(X_batch)
        prev_list = []
        logits = []
        for i_k, val_k in enumerate(self.chain_order):
            if i_k == 0:
                X = X_processed
            else:
                X = torch.cat((X_processed, prev_list), dim=1)
            logit = self.heads[val_k](X)
            logits.append(logit)
            if self.chain_type == "logit":
                prev_list.append(logit)
            elif self.chain_type == "probability":
                prev_list.append(F.softmax(logit, dim=1))
            elif self.chain_type == "prediction":
                prev_list.append(logit.argmax(dim=1))
            # elif self.chain_type == "true":
            #     prev_list.append(y_batch[:, k])
        return torch.stack(logits, dim=1)
    
    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the loss for a batch of data. Use CrossEntropy for pointwise classification.
        
        Parameters:
        ----------
        X_batch : torch.Tensor
            Batch of input data. Shape (batch_size, input_dim).
        y_batch : torch.Tensor
            Batch of target variables. Shape (batch_size, K).
        data_size : int, optional
            Total size of the dataset.
        verbose : bool, optional
            Whether to print the loss. Default is False.
        """
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
            assert logit.shape == torch.Size([X_batch.shape[0], self.num_classes_lst[val_k]]), f"logit.shape={logit.shape} != (X_batch.shape[0], {self.num_classes_lst[val_k]})"
            assert loss_head.shape == torch.Size([1]), f"loss_head.shape={loss_head.shape} != (1)"
            logits.append(logit)
            total_loss += loss_head
            if verbose:
                print(f"head={i_k} loss={loss_head:.2f}")

        reg_loss = self.regularization()
        if verbose:
            print(f"reg_loss={reg_loss:.2f}")
        total_loss += self.beta * reg_loss

        return total_loss

    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the loss for a batch of data. Reference to train_loss.
        """
        return self.train_loss(X_batch, y_batch, data_size, verbose)


# TODO: Choose version of VBLL and adapt this documentation to the chosen version.
# TODO: In VBLL it is easy to return probs. How to adapt it to go for logits?
# TODO: Refactor accordingly.
"""## Softmax VBLL models"""
class SoftmaxVBLL(LLModel):
    def __init__(self, p, K, beta, vbll_cfg, num_classes_lst=None, intercept=False, backbone=None):
        """
        Initialize the Softmax VBLL model. Reference to https://github.com/matekrk/vbll
        
        Parameters:
        ----------
        p : int
            Input dimensionality.
        K : int
            Number of outputs.
        beta : float
            Regularization parameter.
        vbll_cfg : dict
            Configuration for VBLL. Reference to https://github.com/matekrk/vbll
        num_classes_lst : list of int, optional
            Number of classes for each output. Default is None (binary classification). Currently works for fixed common number of classes.
        intercept : bool, optional
            Whether to include an intercept term. Default is False.
        backbone : nn.Module, optional
            Backbone model. Default is None (no backbone).
        """
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

        self.params = self.get_learnable_parameters()

    def get_learnable_parameters(self):
        params = []
        if self.backbone is not None:
            params += list(self.backbone.parameters())
        for head in self.heads:
            params += list(head.parameters())
        return nn.ParameterList(params)

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

    def forward(self, X_batch):
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
        X_processed = self.process(X_batch)

        probs = []
        for head in self.heads:
            distr = head(X_processed).predictive
            probs = distr.probs
            probs.append(probs)
        return torch.stack(probs, dim=1)

    def predict(self, X_batch, threshold=None):
        preds = self.forward(X_batch)
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
    def __init__(self, p, K, beta, vbll_cfg, num_classes_lst=None, intercept=False, backbone=None, chain_order=None, chain_type="probability"):
        LLModelCC.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone, chain_order=chain_order, chain_type=chain_type)
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

"""# General / Utils model"""
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

    model_class = model_classes[cfg.model_type]
    model_args = {k: v for k, v in vars(cfg).items() if k in model_class.__init__.__code__.co_varnames}
    print(model_args)
    return model_class(**model_args, backbone=backbone)
