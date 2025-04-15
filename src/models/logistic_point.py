import torch
import torch.nn as nn

from .generic import LLModel

class LogisticModel(LLModel):
    
    NUM_PER_OUTPUT = 1
    
    """
    Logistic Pointwise model with optional Chain of Classifiers (CC) integration.
    Handles both independent and chain-structured outputs.
    """
    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None, 
                 chain_order=None, chain_type=None, nums_per_output=NUM_PER_OUTPUT,
                 prior_mean=None, prior_log_scale=None, prior_mean_learnable=False, prior_scale_learnable=False):
        """
        Initialize the LogisticModel.

        Args:
            p (int): Dimensionality of the input features after processing by the backbone network.
            K (int): Number of outputs (attributes).
            beta (float, optional): Regularization parameter. Defaults to 1.0.
            intercept (bool, optional): Whether to include an intercept term in the model. Defaults to False.
            backbone (torch.nn.Module, optional): Backbone network to transform input features. Defaults to None.
            chain_order (list of int, optional): Order of the chain. Defaults to None.
            chain_type (str, optional): Type of the chain. Defaults to None.
            nums_per_output (int, optional): Number of outputs for each output. Defaults to 1 (binary classification).
            prior_mean (torch.Tensor or list, optional): Prior means for each output. Defaults to None.
            prior_log_scale (torch.Tensor or list, optional): Prior log scales for each output. Defaults to None.
            prior_mean_learnable (bool, optional): Whether the prior means are learnable. Defaults to False.
            prior_scale_learnable (bool, optional): Whether the prior scales are learnable. Defaults to False.
        """
        p_adjusted = super().__init__(p, K, beta, intercept, backbone, chain_order, chain_type, nums_per_output)
        print(f"[PointwiseModel] input_dim={p_adjusted} output_dim={K} beta={beta} chain_order={chain_order} chain_type={chain_type}")

        self.chain = self.chain_order is not None
        self.loss = nn.BCELoss(reduction='mean')

        # Initialize prior means and log scales
        self.prior_mean_list = self._initialize_prior(prior_mean, K, p_adjusted, default_value=0.0)
        self.prior_log_scale_list = self._initialize_prior(prior_log_scale, K, p_adjusted, default_value=0.0)

        # Make priors learnable if specified
        self.prior_mean_list = nn.ParameterList([nn.Parameter(mu, requires_grad=prior_mean_learnable) for mu in self.prior_mean_list])
        self.prior_log_scale_list = nn.ParameterList([nn.Parameter(log_scale, requires_grad=prior_scale_learnable) for log_scale in self.prior_log_scale_list])

        # Initialize weights (means)
        self.delta_dirac_weight_list = nn.ParameterList([nn.Parameter(torch.zeros(p_adjusted)) for _ in range(K)])

    def _initialize_prior(self, prior, K, p, default_value):
        """
        Initialize prior parameters.

        Args:
            prior (torch.Tensor or list, optional): Prior values.
            K (int): Number of outputs.
            p (int): Dimensionality of the input features.
            default_value (float): Default value for initialization.

        Returns:
            list: List of prior tensors.
        """
        if prior is None:
            return [torch.full((p,), default_value) for _ in range(K)]
        elif isinstance(prior, torch.Tensor):
            assert prior.shape == (K, p), f"Prior tensor must have shape ({K}, {p})."
            return [prior[i] for i in range(K)]
        elif isinstance(prior, list):
            assert len(prior) == K, f"Length of prior list must be equal to K={K}."
            for i in range(K):
                if isinstance(prior[i], torch.Tensor):
                    assert prior[i].shape == (p,), f"Each prior tensor must have shape (p,) where p={p}."
                    prior[i] = prior[i]
                else:
                    assert isinstance(prior[i], float), f"Each prior must be a float or a tensor."
                    prior[i] = torch.full((p,), prior[i])
            return prior
        else:
            raise ValueError("Invalid prior format.")

    def get_learnable_parameters(self, named=True):
        """
        Get learnable parameters of the model.

        Args:
            named (bool, optional): Whether to return named parameters. Defaults to True.

        Returns:
            nn.ParameterList: List of learnable parameters.
        """
        if named:
            named_params = []
            if self.backbone:
                named_params.extend(self.backbone.named_parameters())
            # named_params.append(("delta_dirac_weight_list", self.delta_dirac_weight_list))
            # named_params.append(("prior_mean_list", self.prior_mean_list))
            # named_params.append(("prior_log_scale_list", self.prior_log_scale_list))
            named_params.extend((f"delta_dirac_weight_{i}", param) for i, param in enumerate(self.delta_dirac_weight_list))
            named_params.extend((f"prior_mean_{i}", param) for i, param in enumerate(self.prior_mean_list))
            named_params.extend((f"prior_log_scale_{i}", param) for i, param in enumerate(self.prior_log_scale_list))
            return named_params
        else:
            params = nn.ModuleList()
            if self.backbone:
                params.extend(self.backbone.parameters())
            params.extend(self.delta_dirac_weight_list)
            params.extend(self.prior_mean_list)
            params.extend(self.prior_log_scale_list)
            return params

    def forward(self, X_batch, y_batch = None):
        """
        Perform a forward pass through the model.

        Args:
            X_batch (torch.Tensor): Input batch of data with shape (batch_size, input_dim).
            y_batch (torch.Tensor, optional): Target batch of data with shape (batch_size, K). Defaults to None.

        Returns:
            torch.Tensor: Predicted probabilities for each output with shape (batch_size, K).
        """
        X_processed = self.process(X_batch)
        prev_list = []
        preds = []

        for i_k in range(self.K):
            if self.chain:
                X_current = self.chain_order.process_chain(X_processed, prev_list, i_k)
            else:
                X_current = X_processed

            logits = torch.matmul(X_current, self.delta_dirac_weight_list[i_k])
            probs = torch.sigmoid(logits)
            preds.append(probs.unsqueeze(1))

            if self.chain:
                y = y_batch[:, i_k] if y_batch is not None else None
                prev_list = self.chain_order.update_chain(prev_list, logits, probs, y)

        preds = torch.cat(preds, dim=1)
        assert preds.shape == (X_batch.shape[0], self.K), f"preds.shape={preds.shape} != (X.shape[0], {self.K})"
        return preds

    def regularization(self):
        """
        Compute the KL divergence regularization term.

        Returns:
            torch.Tensor: The computed regularization term as a scalar tensor.
        """
        kl_div = 0.0
        for delta_dirac_weight, prior_mean, prior_log_scale in zip(self.delta_dirac_weight_list, self.prior_mean_list, self.prior_log_scale_list):
            prior_scale = torch.exp(prior_log_scale)
            kl_div += -0.5 * torch.sum(1 + 2 * prior_log_scale - ((delta_dirac_weight - prior_mean) ** 2 + 1) / prior_scale ** 2)
        return kl_div

    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the training loss.

        Args:
            X_batch (torch.Tensor): Batch of input data.
            y_batch (torch.Tensor): Batch of target variables.
            data_size (int, optional): Total size of the dataset. Defaults to None.
            verbose (bool, optional): Whether to print loss details. Defaults to False.

        Returns:
            torch.Tensor: The computed training loss.
        """
        data_size = data_size or X_batch.shape[0]
        preds = self.forward(X_batch)
        mean_bce = self.loss(preds, y_batch)
        mean_reg = self.regularization() / data_size if self.beta else torch.tensor(0.0, device=mean_bce.device)
        if verbose:
            print(f"[Train Loss] BCE: {mean_bce.item()}, Regularization: {mean_reg.item()}")
        return mean_bce + self.beta * mean_reg

    @torch.no_grad()
    def predict(self, X_batch, threshold=0.5):
        """
        Predict binary outputs for the input data.

        Args:
            X_batch (torch.Tensor): Input data.
            threshold (float, optional): Threshold for binary classification. Defaults to 0.5.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Predicted binary outputs (0 or 1).
                - torch.Tensor: Predicted probabilities.
        """
        preds = self.forward(X_batch)
        return (preds > threshold).float(), preds

    @torch.no_grad()
    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the test loss.

        Args:
            X_batch (torch.Tensor): Batch of input data.
            y_batch (torch.Tensor): Batch of target variables.
            data_size (int, optional): Total size of the dataset. Defaults to None.
            verbose (bool, optional): Whether to print loss details. Defaults to False.

        Returns:
            torch.Tensor: The computed test loss.
        """
        data_size = data_size or X_batch.shape[0]
        preds = self.forward(X_batch)
        mean_bce = self.loss(preds, y_batch)
        mean_reg = self.regularization() / data_size if self.beta else torch.tensor(0.0, device=mean_bce.device)
        if verbose:
            print(f"[Test Loss] BCE: {mean_bce.item()}, Regularization: {mean_reg.item()}")
        return mean_bce + self.beta * mean_reg
    
    def compute_negative_log_likelihood(self, X_batch, y_batch, mc = False, n_samples = 1000):
        """
        Compute the negative log likelihood of the data given the predictions.

        Parameters:
        ----------
        X_batch : torch.Tensor
            Predicted probabilities for each output. Shape (n_samples, K).
        y_batch : torch.Tensor
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
        preds = self.forward(X_batch)
        loss = nn.BCELoss(reduction='none')
        nll = torch.mean(loss(preds, y_batch), dim=0)
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

    