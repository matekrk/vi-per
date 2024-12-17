import torch
import torch.nn as nn
import torch.nn.functional as F

from objective import ELL_MC_MH, ELL_TB_MH, KL_MH, ELL_MC_mvn_MH, ELL_TB_mvn_MH, KL_mvn_MH

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

        if self.intercept:
            X_processed = torch.cat((torch.ones(X_processed.size()[0], 1, device=X_processed.device), X_processed), 1)
        return X_processed

    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        raise ValueError("[LLModel] train_loss not implemented")
    
    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        raise ValueError("[LLModel] test_loss not implemented")

    def predict(self, X):
        raise ValueError("[LLModel] predict not implemented")
    
    def forward(self, X):
        self.predict(X)

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

    def __init__(self, p, K, method=0, beta=1.0, intercept=False,
                 mu=None, sig=None, Sig=None, m_init=None, s_init=None, 
                 prior_scale=1.0, posterior_mean_init_scale=1.0, posterior_var_init_add=0.0,
                 l_max=12.0, adaptive_l=False, n_samples=500, backbone=None):
        p = super().__init__(p, K, beta=beta, intercept=intercept, backbone=backbone)

        self.method = method
        self.l_max = l_max
        self.adaptive_l = adaptive_l
        self.n_samples = n_samples

        # Initialize prior parameters
        if mu is None:
            self.mu_list = [torch.zeros(p, dtype=torch.double) for _ in range(K)]
        else:
            self.mu_list = [mu[:, k] for k in range(K)]

        if sig is None:
            self.sig_list = [torch.ones(p, dtype=torch.double) * prior_scale for _ in range(K)]
        else:
            self.sig_list = [sig[:, k] for k in range(K)]

        if Sig is None:
            self.Sig_list = [torch.eye(p, dtype=torch.double) * prior_scale for _ in range(K)]
        else:
            self.Sig_list = Sig  # List of K covariance matrices of shape (p, p)

        # Initialize variational parameters
        if m_init is None:
            self.m_list = [torch.randn(p, dtype=torch.double) * posterior_mean_init_scale for _ in range(K)]
        else:
            self.m_list = [m_init[:, k] for k in range(K)]

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
                self.s_list = [s_init[:, k] for k in range(K)]
                self.u_list = [torch.log(s) for s in self.s_list]
            elif method in [1, 5]:
                self.u_list = s_init  # Should be list of u tensors for each output

        # Set requires_grad=True for variational parameters
        for m in self.m_list:
            m.requires_grad = True
        for u in self.u_list:
            u.requires_grad = True

        # Collect parameters for optimization
        self.params = nn.ParameterList(list(self.m_list) + list(self.u_list))
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
        y_list = [y_batch[:, k] for k in range(self.K)]

        if self.method in [0, 4]:
            s_list = [torch.exp(u).to(X_batch.device) for u in self.u_list]
            sig_list = [sig.to(X_batch.device) for sig in self.sig_list]

            if self.method == 0:
                likelihood = -ELL_TB_MH(m_list, s_list, y_list, X_processed, l_max=self.l_terms)
                KL_div = KL_MH(m_list, s_list, mu_list, sig_list)
            else:
                likelihood = -ELL_MC_MH(m_list, s_list, y_list, X_processed, n_samples=self.n_samples)
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
                likelihood = -ELL_TB_mvn_MH(m_list, S_list, y_list, X_processed, l_max=self.l_terms)
                KL_div = KL_mvn_MH(m_list, S_list, mu_list, Sig_list)
            else:
                likelihood = -ELL_MC_mvn_MH(m_list, S_list, y_list, X_processed, n_samples=self.n_samples)
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


    def predict(self, X):
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

"""## Sigmoid Deterministic model"""

class LogisticPointwise(LLModel):

    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None, m_init=None, mu=None, Sig=None):
        p = super().__init__(p, K, beta=beta, intercept=intercept, backbone=backbone)

        # Initialize prior parameters
        if mu is None:
            self.mu_list = [torch.zeros(p, dtype=torch.double) for _ in range(K)]
        else:
            self.mu_list = [mu[:, k] for k in range(K)]

        if Sig is None:
            self.Sig_list = [torch.eye(p, dtype=torch.double) for _ in range(K)]
        else:
            self.Sig_list = Sig  # List of K covariance matrices of shape (p, p)

        # Initialize variational parameters
        if m_init is None:
            self.m_list = [torch.randn(p, dtype=torch.double) for _ in range(K)]
        else:
            self.m_list = [m_init[:, k] for k in range(K)]

        # Set requires_grad=True for variational parameters
        for m in self.m_list:
            m.requires_grad = True

        # Collect parameters for optimization
        self.params = nn.ParameterList(self.m_list)
        if self.backbone is not None:
            self.params += list(self.backbone.parameters())

    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        data_size = data_size or X_batch.shape[0]

        preds = self.predict(X_batch)
        assert preds.shape == y_batch.shape, f"preds.shape={preds.shape} != y_batch.shape={y_batch.shape}"
        loss = nn.BCELoss(reduction='mean')
        mean_bce = loss(preds, y_batch)
        mean_reg = self.regularization() / data_size

        if verbose:
            print(f"mean_bce_loss={mean_bce:.2f}  mean_reg={mean_reg:.2f}")

        return mean_bce + self.beta * mean_reg

    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False, other_beta=None):
        data_size = data_size or X_batch.shape[0]

        preds = self.predict(X_batch)
        assert preds.shape == y_batch.shape, f"preds.shape={preds.shape} != y_batch.shape={y_batch.shape}"
        loss = nn.BCELoss(reduction='mean')
        mean_bce = loss(preds, y_batch)
        mean_reg = self.regularization() / data_size

        if verbose:
            print(f"mean_bce_loss={mean_bce:.2f}  mean_reg={mean_reg:.2f}")

        beta = other_beta or self.beta
        return mean_bce + beta * mean_reg

    def regularization(self):
        log_prob = 0.
        for m, prior_mu, prior_Sig in zip(self.m_list, self.mu_list, self.Sig_list):
            # simple prior distribution
            d = torch.distributions.MultivariateNormal(loc=prior_mu, covariance_matrix=prior_Sig)
            log_prob += d.log_prob(m)
        return -log_prob

    def predict(self, X):
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

"""## VBLL models"""

class SoftmaxVBLL(LLModel):

    def __init__(self, p, K, beta, vbll_cfg, intercept=False, backbone=None):
        p = super().__init__(p, K, beta, intercept=intercept, backbone=backbone)
        vbll_cfg.REGULARIZATION_WEIGHT = self.beta
        self.vbll_cfg = vbll_cfg

        self.heads = nn.ModuleList([self.make_output_layer(num_hidden=p, num_classes=2) for k in range(K)])

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

    def predict(self, X):
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
            pred = head(X_processed).predictive.probs[:, 1]
            preds.append(pred.unsqueeze(1))

        preds = torch.cat(preds, dim=1)  # Shape: (n_samples, K)
        return preds

class SoftmaxPointwise(LLModel):

    def __init__(self, p, K, beta=0.0, num_classes_lst=None, intercept=False, backbone=None):
        p = super().__init__(p, K, beta=beta, intercept=intercept, backbone=backbone)
        self.num_classes_lst = [2] * K #hardcoded

        self.heads = nn.ModuleList([self.make_output_layer(num_classes=self.num_classes_lst[k]) for k in range(K)])

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

        loss = nn.CrossEntropyLoss(reduction='mean')
        total_loss = 0.0

        for i, (head,y) in enumerate(zip(self.heads,y_batch.T)):
            pred = head(X_processed)
            pred = F.log_softmax(pred, dim=1)
            loss_head = loss(pred, y.to(torch.long)) 
            total_loss += loss_head
            if verbose:
                print(f"head={i} loss={loss_head:.2f}")

        return total_loss

    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        data_size = data_size or X_batch.shape[0]
        X_processed = self.process(X_batch)

        loss = nn.CrossEntropyLoss(reduction='mean')
        total_loss = 0.0

        for i, (head,y) in enumerate(zip(self.heads,y_batch.T)):
            pred = head(X_processed)
            pred = F.log_softmax(pred, dim=1)
            loss_head = loss(pred, y.to(torch.long)) 
            total_loss += loss_head
            if verbose:
                print(f"head={i} loss={loss_head:.2f}")

        return total_loss

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

    def predict(self, X):
      preds = self.forward(X)
      all_preds = []
      for i in range(self.K):
        max_class = torch.argmax(preds[:, i, :], dim=-1)
        all_preds.append(max_class)
      return torch.stack(all_preds, dim=1)
    
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
        "softmaxpointwise": SoftmaxPointwise
    }
    if cfg.model_type not in model_classes:
        raise ValueError(f"Unknown model_type={cfg.model_type}")

    #return model_classes[model_type](p, K, method=method, beta=beta, intercept=intercept, backbone=backbone, vbll_cfg=vbll_cfg)
    model_class = model_classes[cfg.model_type]
    model_args = {k: v for k, v in vars(cfg).items() if k in model_class.__init__.__code__.co_varnames}
    return model_class(**model_args, backbone=backbone)
