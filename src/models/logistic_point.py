import torch
import torch.nn as nn
from .generic import LLModel, LLModelCC

"""## Sigmoid-pointwise model"""
class LogisticPointwise(LLModel):
    """
    Logistic regression model for pointwise predictions. This model is used for binary classification tasks where each output is independent.
    """
    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None, m_init=None, 
                 prior_mu=None, prior_Sig=None, prior_mean_learnable=False, prior_scale_learnable=False):
        """
        Initialize an instance of the LogisticPointwise model.

        Args:
            p (int): Dimensionality of the input features after processing by the backbone network.
            K (int): Number of outputs (attributes).
            beta (float, optional): Regularization parameter. Defaults to 1.0.
            intercept (bool, optional): Whether to include an intercept term in the model. Defaults to False.
            backbone (torch.nn.Module, optional): Backbone network to transform input features. Defaults to None (no preprocessing).
            m_init (torch.Tensor or list, optional): Initial means of the variational distributions. 
                If a tensor, it should have shape (K, p). If a list, it should contain K tensors, each of shape (p,). Defaults to None (random initialization).
            prior_mu (torch.Tensor or list, optional): Prior means for each output. 
                If a tensor, it should have shape (K, p). If a list, it should contain K tensors, each of shape (p,). Defaults to None (zero means).
            prior_Sig (torch.Tensor or list, optional): Prior covariance matrices for each output. 
                If a tensor, it should have shape (K, p, p). If a list, it should contain K tensors, each of shape (p, p). Defaults to None (identity matrices).
            prior_mean_learnable (bool, optional): Whether the prior means are learnable. Defaults to False.
            prior_scale_learnable (bool, optional): Whether the prior scales are learnable. Defaults to False.

        Returns:
            int: Adjusted input dimensionality (p), incremented by 1 if intercept is True.

        Notes:
            - The `m_init` parameter allows for custom initialization of the variational means. If not provided, random initialization is used.
            - The `prior_mu` and `prior_Sig` parameters define the prior distributions for the model's weights. If not provided, default priors are used.
            - If `intercept` is True, the input dimensionality `p` is incremented by 1 to account for the intercept term.
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
        Compute the training loss (Binary Cross-Entropy) for a batch of data.

        Args:
            X_batch (torch.Tensor): Batch of input data with shape (batch_size, input_dim).
            y_batch (torch.Tensor): Batch of target variables with shape (batch_size, K).
            data_size (int, optional): Total size of the dataset. Defaults to None, in which case the batch size is used.
            verbose (bool, optional): Whether to print the loss details. Defaults to False.

        Returns:
            torch.Tensor: The computed training loss, which includes the BCE loss and the regularization term (if applicable).

        Notes:
            - The `data_size` parameter is used to scale the regularization term. If not provided, the batch size is used.
            - The method computes the BCE loss for the predictions and adds the regularization term scaled by the `beta` parameter.
            - If `verbose` is True, the method prints the BCE loss and the regularization term (if `beta` is non-zero).
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

    @torch.no_grad()
    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False, other_beta=None):
        """
        Compute the test loss (Binary Cross-Entropy) for a batch of data.

        Args:
            X_batch (torch.Tensor): Batch of input data with shape (batch_size, input_dim).
            y_batch (torch.Tensor): Batch of target variables with shape (batch_size, K).
            data_size (int, optional): Total size of the dataset. Defaults to None, in which case the batch size is used.
            verbose (bool, optional): Whether to print the loss details. Defaults to False.
            other_beta (float, optional): Alternative regularization parameter to use instead of the model's `beta`. Defaults to None.

        Returns:
            torch.Tensor: The computed test loss, which includes the BCE loss and the regularization term (if applicable).

        Notes:
            - The `data_size` parameter is used to scale the regularization term. If not provided, the batch size is used.
            - The method computes the BCE loss for the predictions and adds the regularization term scaled by the `beta` parameter or `other_beta` if provided.
            - If `verbose` is True, the method prints the BCE loss and the regularization term (if `beta` or `other_beta` is non-zero).
        """
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
        Compute the regularization term for the model.

        This term represents the KL divergence between the variational distributions of the model parameters 
        and their corresponding prior distributions. For pointwise logistic regression, this is equivalent 
        to the negative log probability of the variational parameters under the prior distribution, which 
        can be interpreted as a form of L2 regularization.

        Returns:
            torch.Tensor: The computed regularization term as a scalar tensor.

        Notes:
            - The regularization term is computed for each output independently and then summed across all outputs.
            - The prior distributions are assumed to be multivariate normal distributions with specified means 
              (`prior_mu_list`) and covariance matrices (`prior_Sig_list`).
            - If an error occurs during the computation for a specific output, a message is printed indicating 
              the output index where the error occurred.
        """
        log_prob = 0.
        for i, (m, prior_mu, prior_Sig) in enumerate(zip(self.m_list, self.prior_mu_list, self.prior_Sig_list)):
            try:
                d = torch.distributions.MultivariateNormal(loc=prior_mu, covariance_matrix=prior_Sig)
                log_prob += d.log_prob(m)
            except:
                print(f"Error in regularization for output {i}")
        return -log_prob

    def forward(self, X_batch):
        """
        Predict probabilities for each output given input data.

        Args:
            X_batch (torch.Tensor): Input data with shape (n_samples, input_dim).

        Returns:
            torch.Tensor: Predicted probabilities for each output with shape (n_samples, K).

        Notes:
            - The method processes the input data using the model's backbone (if any) and computes the 
              predicted probabilities for each output using the logistic regression model.
            - The predictions are independent for each output and are computed using the sigmoid activation function.
            - The output probabilities are in the range [0, 1], where higher values indicate a higher likelihood 
              of the positive class for binary classification tasks.
        """
        X_processed = self.process(X_batch)

        preds = []
        for i, m in enumerate(self.m_list):
            pred = torch.sigmoid(X_processed @ m)
            preds.append(pred.unsqueeze(1))

        preds = torch.cat(preds, dim=1)
        assert preds.shape == (X_batch.shape[0], self.K), f"preds.shape={preds.shape} != (X.shape[0], {self.K})"
        return preds
    
    @torch.no_grad()
    def predict(self, X_batch, threshold=0.5):
        """
        Predict binary labels and probabilities for each output given input data.

        Args:
            X_batch (torch.Tensor): Input data with shape (n_samples, input_dim).
            threshold (float, optional): Threshold for converting probabilities to binary labels. Defaults to 0.5.

        Returns:
            tuple:
            - predictions (torch.Tensor): Predicted binary labels for each output with shape (n_samples, K).
            - preds (torch.Tensor): Predicted probabilities for each output with shape (n_samples, K).

        Notes:
            - The `threshold` parameter is used to determine the binary classification labels. 
              Probabilities greater than the threshold are classified as 1, otherwise 0.
            - The method first computes the predicted probabilities using the `forward` method, 
              and then applies the threshold to obtain binary labels.
        """
        preds = self.forward(X_batch)
        return (preds > threshold).float(), preds

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
        nll = torch.mean(loss(preds, y), dim=0)
        assert nll.shape == (self.K,), f"nll.shape={nll.shape} != (K={self.K})"
        return nll

    def get_confidences(self, preds):
        """
        Compute the confidence scores for the predictions. The confidence is defined as the maximum 
        between the predicted probability and its complement (1 - predicted probability).
        Args:
            preds (torch.Tensor): A tensor containing predicted probabilities for each output. 
                      Shape should be (n_samples, K), where `n_samples` is the number 
                      of samples and `K` is the number of classes or outputs.
        Returns:
            torch.Tensor: A tensor of confidence scores for each prediction. Shape matches the 
                  input `preds` tensor.
        Notes:
            - This method assumes that `preds` contains probabilities in the range [0, 1].
            - The confidence score represents the model's certainty about its predictions.
        """
        return torch.max(torch.stack([preds, 1 - preds]), dim=0)[0]


"""## Logistic-pointwise CC model """
class LogisticPointwiseCC(LLModelCC, LogisticPointwise):
    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None, m_init=None, 
                 prior_mu=None, prior_Sig=None, prior_mean_learnable=False, prior_scale_learnable=False,
                 chain_order=None, chain_type="logit"):
        """
        Initialize an instance of the LogisticPointwiseCC class.

        Args:
            p (int): Dimensionality of the input features after processing by the backbone network.
            K (int): Number of outputs (attributes).
            beta (float, optional): Regularization parameter. Defaults to 1.0.
            intercept (bool, optional): Whether to include an intercept term in the model. Defaults to False.
            backbone (torch.nn.Module, optional): Backbone network to transform input features. Defaults to None (no preprocessing).
            m_init (torch.Tensor or list, optional): Initial means of the variational distributions. 
                If a tensor, it should have shape (K, p) or (K, p + K). If a list, it should contain K tensors, 
                each of shape (p,) or (p + chain_order[k],). Defaults to None (random initialization).
            prior_mu (torch.Tensor or list, optional): Prior means for each output. 
                If a tensor, it should have shape (K, p) or (K, p + K). If a list, it should contain K tensors, 
                each of shape (p,) or (p + chain_order[k],). Defaults to None (zero means).
            prior_Sig (torch.Tensor or list, optional): Prior covariance matrices for each output. 
                If a tensor, it should have shape (K, p, p) or (K, p + K, p + K). If a list, it should contain K tensors, 
                each of shape (p, p) or (p + chain_order[k], p + chain_order[k]). Defaults to None (identity matrices).
            prior_mean_learnable (bool, optional): Whether the prior means are learnable. Defaults to False.
            prior_scale_learnable (bool, optional): Whether the prior scales are learnable. Defaults to False.
            chain_order (list of int, optional): Order of the chain. Must have length K. Defaults to None (sequential order from 0 to K-1).
            chain_type (str, optional): Type of the chain. Must be one of ["logit", "probability", "prediction", "true"]. Defaults to "logit".

        Raises:
            AssertionError: If the length of `chain_order` does not match the number of outputs `K`.

        Notes:
            - The `chain_order` parameter determines the sequence in which the outputs are processed. 
              By default, it is a sequential list from 0 to K-1.
            - The `chain_type` parameter specifies how the chain is interpreted or used in the model.
            - The `m_init`, `prior_mu`, and `prior_Sig` parameters allow for custom initialization of the model's 
              variational distributions and priors. If not provided, default values are used.
        """
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
    
    def forward(self, X_batch):
        """
        Perform a forward pass through the logistic chain model.
        Args:
            X_batch (torch.Tensor): Input batch of data with shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Predicted probabilities for each output in the chain with shape (batch_size, K).
        Notes:
            - The method processes the input data, applies the logistic chain model, and computes probabilities
              for each output in the chain.
            - The chain order (`self.chain_order`) determines the sequence in which the outputs are processed.
            - The chain type (`self.chain_type`) specifies how the intermediate outputs are used:
            - "logit": Use the raw logits as intermediate outputs.
            - "probability": Use the probabilities as intermediate outputs.
            - "prediction": Use binary predictions (thresholded at 0.5) as intermediate outputs.
        """
        X_processed = self.process(X_batch)
        X_processed = X_processed.to(torch.double)
        m_list = [m.to(X_processed.device) for m in self.m_list]
        prev_list = []
        probabilities = []
        for i, k in enumerate(self.chain_order):
            if i == 0:
                logit = (X_processed @ m_list[(self.chain_order == i).nonzero().item()]).to(X_processed.device)
                probability = torch.sigmoid(logit)
            else:
                prev_cat = torch.cat(prev_list, dim=1)
                logit = (torch.cat((X_processed, prev_cat), dim=1) @ m_list[(self.chain_order == i).nonzero().item()]).to(X_processed.device)
                probability = torch.sigmoid(logit)
            if self.chain_type == "logit":
                prev_list.append(logit.unsqueeze(1))
            elif self.chain_type == "probability":
                prev_list.append(probability.unsqueeze(1))
            elif self.chain_type == "prediction":
                prev_list.append((probability > 0.5).float().unsqueeze(1))
            # elif self.chain_type == "true":
            #     prev_list.append(y_batch[:, k].unsqueeze(1))
            probabilities.append(probability.unsqueeze(1))
        
        return torch.cat(probabilities, dim=1)
