"""
vi_diagonal.py - Diagonal covariance variational inference model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseVIModel, VIMethod, get_method
from .objective import KL, neg_ELL_MC, neg_ELL_TB

class DiagonalVIModel(BaseVIModel):
    
    NUM_PER_OUTPUT = 1
    
    """
    Variational Inference model with diagonal covariance matrices.
    Implements the VI-PER (Variational Inference with Probit Estimates Reformulation) method
    with diagonal covariance structure.
    """
    
    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None,
                 chain_order=None, chain_type=None, nums_per_output=NUM_PER_OUTPUT,
                 method=VIMethod.TB_BOUND, 
                 prior_mean=None, prior_mean_learnable=False,
                 prior_log_scale=1.0, prior_scale_learnable=False,
                 adaptive_l=False, l_max=12.0, n_samples=500,
                 posterior_mean_init=None, posterior_scale_init=None, 
                 posterior_mean_init_scale=1.0, posterior_scale_init_add=0.0,
                 incorrect_straight_sigmoid=False,
                 sigmoid_mc_computation=False, sigmoid_mc_n_samples=100):
        """
        Initialize an instance of the DiagonalVIModel class.

        Args:
            p (int): Dimensionality of the input features after processing by the backbone network.
            K (int): Number of outputs (labels).
            beta (float, optional): Regularization parameter (in ELBO). Defaults to 1.0.
            intercept (bool, optional): Whether to include an intercept term in the model. Defaults to False.
            backbone (torch.nn.Module, optional): Backbone network to transform input features. Defaults to None.
            chain_order (list of int, optional): Order of the chain. Defaults to None.
            chain_type (str, optional): Type of the chain. Must be one of ["logit", "probability", "prediction", "truth ground"]. Defaults to None.
            nums_per_output (int, optional): Number of outputs for each output. Defaults to 1.
            method (Union[VIMethod, int], optional): Method to use for approximating the ELBO. Can be an instance of 
                VIMethod or an integer (0 for TB BOUND, 4 for MONTE_CARLO). Defaults to VIMethod.TB_BOUND.
            prior_mean (torch.Tensor, optional): Prior means. Defaults to None (0.0).
            prior_log_scale (torch.Tensor, optional): Prior log standard deviations. Defaults to None (log(prior_scale_init)).
            prior_mean_learnable (bool, optional): Whether the prior means are learnable. Defaults to False.
            prior_log_scale (Union[float, List[float]], optional): Initial value(s) for the prior scale. Defaults to 1.0.
            prior_scale_learnable (bool, optional): Whether the prior scales are learnable. Defaults to False.
            adaptive_l (bool, optional): Whether to adaptively increase l during training. Defaults to False.
            l_max (float, optional): Maximum value of l for the bound. Defaults to 12.0.
            n_samples (int, optional): Number of samples for Monte Carlo estimation. Defaults to 500.
            posterior_mean_init (torch.Tensor, optional): Initial means for variational distribution. Shape (K, p). Defaults to None.
            posterior_scale_init (torch.Tensor, optional): Initial scales for variational distribution. Shape (K, p). Defaults to None.
            posterior_mean_init_scale (float, optional): Scale for random initialization of means. Defaults to 1.0.
            posterior_scale_init_add (float, optional): Value to add to initial log-scales. Defaults to 0.0.
            incorrect_straight_sigmoid (bool, optional): Use simple sigmoid without correction. Defaults to False.
            sigmoid_mc_computation (bool, optional): Use Monte Carlo estimation for sigmoid. Defaults to False.
            sigmoid_mc_n_samples (int, optional): Number of samples for Monte Carlo sigmoid estimation. Defaults to 100.

        Raises:
            ValueError: If an invalid method is provided.
        """
        super().__init__(
            p=p, K=K, beta=beta, intercept=intercept, backbone=backbone,
            chain_order=chain_order, chain_type=chain_type, nums_per_output=nums_per_output,
            prior_mean=prior_mean, prior_mean_learnable=prior_mean_learnable,
            prior_log_scale=prior_log_scale, prior_scale_learnable=prior_scale_learnable,
            adaptive_l=adaptive_l, l_max=l_max, n_samples=n_samples,
            incorrect_straight_sigmoid=incorrect_straight_sigmoid,
            sigmoid_mc_computation=sigmoid_mc_computation, sigmoid_mc_n_samples=sigmoid_mc_n_samples,
        )
        
        # Initialize chain-of-classifier functionality if chain_order is provided
        # self.chain = ChainOfClassifiers(K, chain_order=chain_order, chain_type=chain_type) if chain_order else None

        self.method = get_method(method)
        print(f"[DiagonalVIModel] method={self.method.name} l_max={l_max} adaptive_l={adaptive_l} n_samples={n_samples}")
        
        # Initialize variational parameters
        self._initialize_variational_parameters(
            posterior_mean_init, posterior_scale_init, posterior_mean_init_scale, posterior_scale_init_add
        )
    
    def _initialize_variational_parameters(self, mean_init, scale_init, mean_init_scale, scale_init_add):
        """
        Initialize the variational parameters for the model.
        
        Args:
            mean_init (torch.Tensor, optional): Initial means for variational distribution.
            scale_init (torch.Tensor, optional): Initial scales for variational distribution.
            mean_init_scale (float): Scale for random initialization of means.
            scale_init_add (float): Value to add to initial log-scales.
        """
        # Initialize means
        if mean_init is None:
            # Random initialization scaled by mean_init_scale
            self.posterior_mean_list = nn.ParameterList([
                nn.Parameter(torch.randn(self.p, dtype=torch.double) * mean_init_scale) 
                for _ in range(self.K)
            ])
        else:
            # Use provided initialization
            assert isinstance(mean_init, torch.Tensor), "mean_init must be a torch.Tensor"
            assert mean_init.shape == (self.K, self.p), f"mean_init must have shape ({self.K}, {self.p})"
            self.posterior_mean_list = nn.ParameterList([
                nn.Parameter(mean_init[k].clone()) for k in range(self.K)
            ])
        
        # Initialize log standard deviations
        if scale_init is None:
            # Initialize log standard deviations to -1.0 + scale_init_add
            self.posterior_log_scale_list = nn.ParameterList([
                nn.Parameter(torch.full((self.p,), -1.0 + scale_init_add, dtype=torch.double))
                for _ in range(self.K)
            ])
        else:
            # Use provided standard deviations
            assert isinstance(scale_init, torch.Tensor), "scale_init must be a torch.Tensor"
            assert scale_init.shape == (self.K, self.p), f"scale_init must have shape ({self.K}, {self.p})"
            # Convert to log space for stability
            self.posterior_log_scale_list = nn.ParameterList([
                nn.Parameter(torch.log(scale_init[k].clone())) for k in range(self.K)
            ])
    
    def expected_sigmoid(self, X_processed, i_k, mc=None, n_samples=None):
        """
        Compute the expected sigmoid function for diagonal normal distribution.

        Args:
            X_processed (torch.Tensor): Processed input data of shape (batch_size, p)
            i_k (int): Index of the output to compute
            mc (bool, optional): Whether to use Monte Carlo estimation. Defaults to None (use model default).
            n_samples (int, optional): Number of samples for MC estimation. Defaults to None (use model default).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Expected sigmoid values of shape (batch_size,)
                - Corrected mean values of shape (batch_size,)
        """
        mc = self.sigmoid_mc_computation if mc is None else mc
        n_samples = self.sigmoid_mc_n_samples if n_samples is None else n_samples
        
        # Get parameters for this output
        mean = self.posterior_mean_list[i_k]
        
        # Calculate mean of the distribution
        M = X_processed @ mean
        
        # If using naive sigmoid, return directly
        if self.incorrect_straight_sigmoid:
            return torch.sigmoid(M), M
        
        # Get standard deviations
        log_scale = self.posterior_log_scale_list[i_k]
        scale = torch.exp(log_scale)
        
        if not mc:
            # Probit approximation (TB bound)
            scaling_factor_diag = torch.einsum("bi,i,bi->b", X_processed, scale**2, X_processed)
            scaling_factor = torch.sqrt(1 + (torch.pi / 8) * scaling_factor_diag)
            M_corrected = M / scaling_factor
            expected_sigmoid = torch.sigmoid(M_corrected)
        else:
            # Monte Carlo estimation
            S = torch.sqrt(torch.sum(X_processed**2 * scale**2, dim=1))
            norm_dist = torch.distributions.Normal(loc=M, scale=S)
            samples = norm_dist.rsample((n_samples,))
            sigmoid_samples = torch.sigmoid(samples)
            expected_sigmoid = sigmoid_samples.mean(dim=0)
            M_corrected = M
        
        return expected_sigmoid, M_corrected
    
    def forward(self, X_batch, y_batch = None):
        """
        Perform a forward pass to predict probabilities for each output.

        Args:
            X_batch (torch.Tensor): Input data tensor with shape (n_samples, input_dim)
            y_batch (torch.Tensor, optional): Target variables. Shape (batch_size, K). Defaults to None.
                Used only in chain-of-classifier mode.

        Returns:
            torch.Tensor: Predicted probabilities for each output with shape (n_samples, K)
        """
        X_processed = self.process(X_batch)

        if self.chain:
            prev_list = []
            probs_list = []

            for i_k in range(self.K):
                X = self.chain.process_chain(X_processed, prev_list, i_k)
                y = y_batch[:, i_k] if y_batch is not None else None
                probs, logits = self.expected_sigmoid(X, i_k)
                probs_list.append(probs.unsqueeze(1))
                prev_list = self.chain.update_chain(prev_list, logits, probs, y)

        else:
            probs_list = [self.expected_sigmoid(X_processed, i_k)[0].unsqueeze(1) for i_k in range(self.K)]
        
        probs_out = torch.cat(probs_list, dim=1)
        assert probs_out.shape == (X_batch.shape[0], self.K), \
            f"probs.shape={probs_out.shape} != ({X_batch.shape[0]}, {self.K})"
        return probs_out
    
    def compute_ELBO(self, X_batch, y_batch, data_size, verbose=False, other_beta=None):
        """
        Compute the Evidence Lower Bound (ELBO) for a batch of data.

        Args:
            X_batch (torch.Tensor): Batch of input data. Shape (batch_size, input_dim).
            y_batch (torch.Tensor): Batch of target variables. Shape (batch_size, K).
            data_size (int): Total size of the dataset.
            verbose (bool, optional): Whether to print detailed loss info. Defaults to False.
            other_beta (float, optional): Override for regularization parameter. Defaults to None.

        Returns:
            torch.Tensor: A tensor of ELBO components for each output. Shape (self.K,).
        """
        X_processed = self.process(X_batch)
        batch_size = X_batch.shape[0]
        
        # Prepare target variables
        y_list = [y_batch[:, k] for k in range(self.K)]
        
        # Compute likelihood and KL divergence for each output
        elbo_components = torch.zeros(self.K, dtype=torch.double, device=X_batch.device)
        prev_list = []  # Initialize for chain-of-classifier mode
        for i_k in range(self.K):
            # Update X for chain-of-classifier mode
            if self.chain:
                X = self.chain.process_chain(X_processed, prev_list, i_k)
            else:
                X = X_processed
            
            # Compute expected log-likelihood based on method
            if self.method == VIMethod.TB_BOUND:
                cur_likelihood = -neg_ELL_TB(
                    self.posterior_mean_list[i_k], self.posterior_scale_list[i_k], y_list[i_k], X, l_max=self.l_max
                )
            elif self.method == VIMethod.MONTE_CARLO:
                cur_likelihood = -neg_ELL_MC(
                    self.posterior_mean_list[i_k], self.posterior_scale_list[i_k], y_list[i_k], X, n_samples=self.n_samples
                )
            else:
                raise ValueError(f"Unsupported method: {self.method}")
            
            # Compute KL divergence
            cur_kl = KL(self.posterior_mean_list[i_k], self.posterior_scale_list[i_k], self.prior_mean_list[i_k], self.prior_scale_list[i_k])
            
            # Normalize and compute ELBO component
            mean_log_lik = cur_likelihood / batch_size
            mean_kl_div = cur_kl / data_size
            beta = other_beta if other_beta is not None else self.beta
            elbo_components[i_k] = mean_log_lik - beta * mean_kl_div
            
            # Update chain state
            if self.chain:
                probs, logits = self.expected_sigmoid(X, i_k)
                y = y_list[i_k] if y_batch is not None else None
                prev_list = self.chain.update_chain(prev_list, logits, probs, y)
        
        if verbose:
            total_elbo = elbo_components.sum()
            print(f"ELBO={total_elbo:.2f} components={elbo_components.tolist()}")
        
        return elbo_components
    
    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the training loss (ELBO) for a batch of data.

        Args:
            X_batch (torch.Tensor): Batch of input data. Shape (batch_size, input_dim).
            y_batch (torch.Tensor): Batch of target variables. Shape (batch_size, K).
            data_size (int, optional): Total size of the dataset. Defaults to None, which uses the batch size.
            verbose (bool, optional): Whether to print the loss details. Defaults to False.

        Returns:
            torch.Tensor: The negative ELBO for the batch. A scalar value.

        Notes:
            - The Evidence Lower Bound (ELBO) is computed using the `compute_ELBO` method.
            - The `data_size` parameter is used to normalize the KL divergence term in the ELBO.
            - The returned value is negated because the training process minimizes the loss.
        """
        data_size = data_size or X_batch.shape[0]
        elbo_components = self.compute_ELBO(X_batch, y_batch, data_size, verbose=verbose)
        return -torch.sum(elbo_components)

    @torch.no_grad()
    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the test loss (negative ELBO) for a batch of data.

        Args:
            X_batch (torch.Tensor): Batch of input data. Shape (batch_size, input_dim).
            y_batch (torch.Tensor): Batch of target variables. Shape (batch_size, K).
            data_size (int, optional): Total size of the dataset. Defaults to None, which uses the batch size.
            verbose (bool, optional): Whether to print the loss details. Defaults to False.

        Returns:
            torch.Tensor: The negative ELBO for the batch. A scalar value.

        Notes:
            - The Evidence Lower Bound (ELBO) is computed using the `compute_ELBO` method.
            - The `data_size` parameter is used to normalize the KL divergence term in the ELBO.
            - The returned value is negated because the test process minimizes the loss with `beta` set to 0.0.
        """
        data_size = data_size or X_batch.shape[0]
        elbo_components = self.compute_ELBO(X_batch, y_batch, data_size, verbose=verbose, other_beta=0.0)
        return -torch.sum(elbo_components)
    
    @property
    def posterior_scale_list(self):
        """
        Returns the posterior scale list for the model.

        Returns:
            List[torch.Tensor]: List of posterior scales for each output.
        """
        return [torch.exp(scale) for scale in self.posterior_log_scale_list]
    
    @property
    def prior_scale_list(self):
        """
        Returns the prior scale list for the model.

        Returns:
            List[torch.Tensor]: List of prior scales for each output.
        """
        return [torch.exp(scale) for scale in self.prior_log_scale_list]
