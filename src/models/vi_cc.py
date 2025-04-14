import torch
import torch.nn as nn
from .generic import LLModelCC
from objective import KL, KL_mvn, neg_ELL_MC_MH, neg_ELL_TB_MH, KL_MH, neg_ELL_MC_mvn_MH, neg_ELL_TB_mvn_MH, KL_mvn_MH, neg_ELL_MC, neg_ELL_TB, neg_ELL_MC_mvn, neg_ELL_TB_mvn
from .vi_classic import LogisticVI

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
        if self.method in [0, 4]:
            return [torch.ones(self.p + val_k, dtype=torch.double, device=ps.device) * ps for _, val_k in enumerate(self.chain_order)]
        elif self.method in [1, 5]:
            return [torch.eye(self.p + val_k, dtype=torch.double, device=ps.device) * ps for _, val_k in enumerate(self.chain_order)]

    @property
    def s_list(self):
        """
        Return the standard deviations for each output.
        """
        return [torch.exp(u) for u in self.u_list]
    
    def S_single(self, i_k):
        """
        [method dep. on i] Return the covariance matrix for a single output.
        Uses Cholesky decomposition to construct the covariance matrix from the lower triangular matrix.
        """
        u = self.u_list[i_k]
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
        """
        Initialize an instance of the LogisticVICC class.
        Args:
            p (int): Dimensionality of the input features after processing by the backbone network.
            K (int): Number of outputs (attributes).
            method (int, optional): Variational inference method to use. Defaults to 0.
            l_max (float, optional): Maximum value for the latent variable. Defaults to 12.0.
            adaptive_l (bool, optional): Whether to use adaptive latent variable scaling. Defaults to False.
            n_samples (int, optional): Number of Monte Carlo samples for variational inference. Defaults to 500.
            beta (float, optional): Regularization parameter. Defaults to 1.0.
            intercept (bool, optional): Whether to include an intercept term in the model. Defaults to False.
            backbone (torch.nn.Module, optional): Backbone network to transform input features. Defaults to None.
            prior_mu (torch.Tensor, optional): Mean of the prior distribution. Defaults to None.
            prior_u_sig (torch.Tensor, optional): Log-scale of the prior distribution. Defaults to None.
            prior_mean_learnable (bool, optional): Whether the prior mean is learnable. Defaults to False.
            prior_scale_init (float, optional): Initial scale for the prior distribution. Defaults to 1.0.
            prior_scale_learnable (bool, optional): Whether the prior scale is learnable. Defaults to False.
            m_init (torch.Tensor or list, optional): Initial values for the variational mean parameters. Defaults to None.
            s_init (torch.Tensor or list, optional): Initial values for the variational scale parameters. Defaults to None.
            posterior_mean_init_scale (float, optional): Scaling factor for initializing the posterior mean. Defaults to 1.0.
            posterior_var_init_add (float, optional): Additional value for initializing the posterior variance. Defaults to 0.0.
            incorrect_straight_sigmoid (bool, optional): Whether to use an incorrect straight sigmoid computation. Defaults to False.
            sigmoid_mc_computation (bool, optional): Whether to use Monte Carlo computation for sigmoid. Defaults to False.
            sigmoid_mc_n_samples (int, optional): Number of Monte Carlo samples for sigmoid computation. Defaults to 100.
            chain_order (list of int, optional): Order of the chain. Defaults to None.
            chain_type (str, optional): Type of the chain. Must be one of ["logit", "probability", "prediction", "true"]. Defaults to "logit".
        Raises:
            AssertionError: If the shape of `m_init` or `s_init` does not match the expected dimensions.
            ValueError: If the shape of `m_init` or `s_init` is invalid for a specific chain order.
        Notes:
            - The `m_list` and `u_list` attributes are initialized based on the provided `m_init` and `s_init` values.
            - The `method` parameter determines the type of variational inference used (e.g., diagonal or full covariance).
            - The `chain_order` parameter specifies the sequence in which the outputs are processed.
            - The `chain_type` parameter defines how the chain is interpreted or used in the model.
        """
        LLModelCC.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone, chain_order=chain_order, chain_type=chain_type)
        LogisticVI.__init__(self, p, K, method=method, l_max=l_max, adaptive_l=adaptive_l, n_samples=n_samples, beta=beta, intercept=intercept,  
                            prior_mu=prior_mu, prior_u_sig=prior_u_sig, prior_mean_learnable=prior_mean_learnable, prior_scale_init=prior_scale_init, prior_scale_learnable=prior_scale_learnable,
                            m_init=m_init, s_init=s_init, posterior_mean_init_scale=posterior_mean_init_scale, posterior_var_init_add=posterior_var_init_add,
                            incorrect_straight_sigmoid=incorrect_straight_sigmoid, sigmoid_mc_computation=sigmoid_mc_computation, sigmoid_mc_n_samples=sigmoid_mc_n_samples,
                            backbone=backbone)
        print(f"[LogisticVICC]")

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
        self.m_list = nn.ParameterList(m_list)

        if s_init is None:
            if method in [0, 4]:
                self.u_list = nn.ParameterList([nn.Parameter(torch.tensor([-1.0 + posterior_var_init_add] * (self.p+self.chain_order[i_k]), dtype=torch.double)) for i_k in range(self.K)])
            elif method in [1, 5]:
                self.u_list = nn.ParameterList([])
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
                u_list = nn.ParameterList([])
                if isinstance(s_init, torch.Tensor):
                    assert s_init.shape[0] == self.K, f"s_init.shape[0]={s_init.shape[0]} != {self.K}"
                    for i_k in range(s_init.shape[0]):
                        s = s_init[i_k]
                        if len(s) == int(self.p * (self.p + 1) / 2):
                            extended_s = torch.ones(int((self.p + self.chain_order[i_k]) * (self.p + self.chain_order[i_k] + 1) / 2), dtype=torch.double) * (1.0 / (self.p + self.chain_order[i_k]))
                            extended_s[:len(s)] = s
                            u_list.append(extended_s)
                        elif len(s) == int((self.p + self.chain_order[i_k]) * (self.p + self.chain_order[i_k] + 1) / 2):
                            u_list.append(s)
                        else:
                            raise ValueError(f"s_init[{i_k}] has wrong shape {s.shape}")
                elif isinstance(s_init, list):
                    assert len(s_init) == self.K
                    for i_k, s in enumerate(s_init):
                        if len(s) == int((self.p + self.chain_order[i_k]) * (self.p + self.chain_order[i_k] + 1) / 2):
                            u_list.append(s)
                        elif len(s) == int(self.p * (self.p + 1) / 2):
                            extended_s = torch.ones(int((self.p + self.chain_order[i_k]) * (self.p + self.chain_order[i_k] + 1) / 2), dtype=torch.double) * (1.0 / (self.p + self.chain_order[i_k]))
                            extended_s[:len(s)] = s
                            u_list.append(extended_s)
                        else:
                            raise ValueError(f"s_init[{i_k}] has wrong shape {s.shape}")
        self.u_list = nn.ParameterList(self.u_list)

        # Set requires_grad=True for variational parameters
        for m in self.m_list:
            m.requires_grad = True
        for u in self.u_list:
            u.requires_grad = True

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
        Predict probabilities for each output given the input data.

        Args:
            X_batch (torch.Tensor): Input data tensor with shape (batch_size, input_dim), 
                where `batch_size` is the number of samples and `input_dim` is the dimensionality 
                of the input features.

        Returns:
            torch.Tensor: Predicted probabilities for each output with shape (batch_size, K), 
                where `K` is the number of outputs (attributes).

        Raises:
            AssertionError: If the shape of the predicted probabilities does not match 
                (batch_size, K).

        Notes:
            - The method processes the input data using the `process` method before making predictions.
            - Predictions are computed using the `expected_sigmoid_multivariate` method for each output.
            - Monte Carlo sampling is used for the sigmoid computation, controlled by the 
              `sigmoid_mc_computation` and `sigmoid_mc_n_samples` attributes.
        """
        X_processed = self.process(X_batch)
        X_processed = X_processed.to(torch.double)

        probs_list = []
        prev_list = []
        for i_k, val_k in enumerate(self.chain_order):
            i_relevant = (self.chain_order == i_k).nonzero().item()
            if i_k == 0:
                X = X_processed
            else:
                prev_cat = torch.cat(prev_list, dim=1)
                X = torch.cat((X_processed, prev_cat), dim=1)
            probability, logit = self.expected_sigmoid_multivariate(X, i_relevant, mc=self.sigmoid_mc_computation, n_samples=self.sigmoid_mc_n_samples)
            probs_list.append(probability.unsqueeze(1))
            if self.chain_type == "logit":
                prev_list.append(logit.unsqueeze(1))
            elif self.chain_type == "probability":
                prev_list.append(probability.unsqueeze(1))
            elif self.chain_type == "prediction":
                prev_list.append((probability > 0.5).float().unsqueeze(1))
            # elif self.chain_type == "true":
            #     prev_list.append(y_batch[:, k].unsqueeze(1))
        out = torch.cat(probs_list, dim=1)
        assert out.shape == (X_batch.shape[0], self.K), f"out.shape={out.shape} != (X_batch.shape[0], {self.K})"
        return out

    def compute_ELBO(self, X_batch, y_batch, data_size, verbose=False, other_beta=None):
        """
        Compute the Evidence Lower Bound (ELBO) for a batch of data.

        Args:
            X_batch (torch.Tensor): Batch of input data. Shape (batch_size, input_dim), 
                where `batch_size` is the number of samples and `input_dim` is the dimensionality 
                of the input features.
            y_batch (torch.Tensor): Batch of target variables. Shape (batch_size, K), 
                where `K` is the number of outputs (attributes).
            data_size (int): Total size of the dataset, used to normalize the KL divergence term.
            verbose (bool, optional): Whether to print detailed loss information. Defaults to False.
            other_beta (float, optional): Regularization parameter for the KL term. If None, uses `self.beta`. Defaults to None.

        Returns:
            torch.Tensor: The computed ELBO for the batch. A scalar value.

        Raises:
            ValueError: If the specified `method` is not recognized.

        Notes:
            - The ELBO is computed as the difference between the expected log-likelihood and the scaled KL divergence.
            - The `verbose` parameter can be used to print the ELBO, mean log-likelihood, and mean KL divergence for debugging.
            - The `other_beta` parameter allows overriding the default regularization parameter `self.beta` for specific computations.
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
            probability, logit = self.expected_sigmoid_multivariate(X, i_relevant, mc=self.sigmoid_mc_computation, n_samples=self.sigmoid_mc_n_samples)
            if self.chain_type == "logit":
                prev_list.append(logit.unsqueeze(1))
            elif self.chain_type == "probability":
                prev_list.append(probability.unsqueeze(1))
            elif self.chain_type == "prediction":
                prev_list.append((probability > 0.5).float().unsqueeze(1))
            # elif self.chain_type == "true":
            #     prev_list.append(y_batch[:, k].unsqueeze(1))

            if self.method in [0, 4]:
                s = self.s_list[i_relevant].to(X_batch.device)
                sig = self.prior_Sig_list[i_relevant].to(X_batch.device)
                if self.method == 0:
                    likelihood += -neg_ELL_TB(m_list[i_relevant], s, y_list[i_relevant], X, l_max=self.l_terms)
                    KL_div += KL(m_list[i_relevant], s, prior_mu_list[i_relevant], sig)
                else:
                    likelihood += -neg_ELL_MC(m_list[i_relevant], s, y_list[i_relevant], X, n_samples=self.n_samples)
                    KL_div += KL(m_list[i_relevant], s, prior_mu_list[i_relevant], sig)

            elif self.method in [1, 5]:

                S = self.S_single(i_relevant).to(X_batch.device)
                Sig = self.prior_Sig_list[i_relevant].to(X_batch.device)
                if self.method == 1:
                    likelihood += -neg_ELL_TB_mvn(m_list[i_relevant], S, y_list[i_relevant], X, l_max=self.l_terms)
                    KL_div += KL_mvn(m_list[i_relevant], S, prior_mu_list[i_relevant], Sig)
                else:
                    likelihood += -neg_ELL_MC_mvn(m_list[i_relevant], S, y_list[i_relevant], X, n_samples=self.n_samples)
                    KL_div += KL_mvn(m_list[i_relevant], S, prior_mu_list[i_relevant], Sig)
            else:
                raise ValueError("Method not recognized")

            assert likelihood.shape == torch.Size([]), f"likelihood.shape={likelihood.shape} != ()"
            assert KL_div.shape == torch.Size([]), f"KL_div.shape={KL_div.shape} != ()"

        mean_log_lik = likelihood/batch_size
        mean_kl_div = KL_div/data_size
        beta = other_beta or self.beta
        ELBO = mean_log_lik - beta*mean_kl_div
        if verbose:
            print(f"ELBO={ELBO:.2f} mean_log_lik={mean_log_lik:.2f} mean_kl_div={mean_kl_div:.2f}")
        return ELBO

    def compute_negative_log_likelihood(self, X_batch, y_batch, mc = False, n_samples = 1000):
        """
        Compute the negative log likelihood (NLL) of the data given the predictions.

        Args:
            X_batch (torch.Tensor): Batch of input data. Shape (batch_size, input_dim), 
                where `batch_size` is the number of samples and `input_dim` is the dimensionality 
                of the input features.
            y_batch (torch.Tensor): Batch of target variables. Shape (batch_size, K), 
                where `K` is the number of outputs (attributes).
            mc (bool, optional): Whether to use Monte Carlo estimation for the NLL. Defaults to False.
            n_samples (int, optional): Number of samples for Monte Carlo estimation. Required if `mc` is True. Defaults to 1000.

        Returns:
            torch.Tensor: The computed negative log likelihood for each output. Shape (K,).

        Raises:
            ValueError: If the specified `method` is not recognized.
            AssertionError: If the length of the computed NLLs does not match the number of outputs `K`.

        Notes:
            - The NLL is computed for each output (attribute) separately.
            - If `mc` is True, Monte Carlo estimation is used to compute the NLL.
            - The method supports both diagonal and full covariance matrices, depending on the model configuration.
        """
        X_processed = self.process(X_batch)
        m_list = [m.to(X_batch.device) for m in self.m_list]
        y_list = [y_batch[:, k] for k in range(self.K)]
        likelihood = []

        prev_list = []
        for i_k, val_k in enumerate(self.chain_order):
            i_relevant = (self.chain_order == i_k).nonzero().item()
            if i_k == 0:
                X = X_processed
            else:
                X = torch.cat((X_processed, torch.cat(prev_list, dim=1)), dim=1)
            probability, logit = self.expected_sigmoid_multivariate(X, i_relevant, mc=self.sigmoid_mc_computation, n_samples=self.sigmoid_mc_n_samples)
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
                S = self.S_single(i_relevant).to(X.device)
                if mc:
                    cur_likelihood = -neg_ELL_MC_mvn(m_list[i_relevant], S, y_list[i_relevant], X, n_samples=n_samples)
                else:
                    cur_likelihood = -neg_ELL_TB_mvn(m_list[i_relevant], S, y_list[i_relevant], X, l_max=self.l_terms)
            else:
                raise ValueError("Method not recognized")
            assert cur_likelihood.shape == torch.Size([]), f"cur_likelihood.shape={cur_likelihood.shape} != ()"
            likelihood.append(cur_likelihood)
        assert len(likelihood) == self.K, f"likelihood must have length {self.K}"
        return torch.tensor(likelihood)
