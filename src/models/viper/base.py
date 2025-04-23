"""
vi_base.py - Base module for variational inference models
"""
from enum import Enum
import torch
import torch.nn as nn

from ..generic import LLModel
from ..cc import ChainOfClassifiers

class VIMethod(Enum):
    """Enumeration of variational inference methods."""
    TB_BOUND = 0  # Proposed TB bound
    MONTE_CARLO = 4    # Monte Carlo estimation

def get_method(method):
    # Store the VI method
    if isinstance(method, int):
        # Handle backward compatibility with integer method specification
        if method == 0:
            method = VIMethod.TB_BOUND
        elif method == 4:
            method = VIMethod.MONTE_CARLO
        else:
            raise ValueError(f"Method {method} not supported for VIModel. Use 0 (TB) or 4 (MC).")
    elif isinstance(method, VIMethod):
        method = method
    else:
        raise ValueError(f"Invalid method: {method}. Must be an instance of VIMethod or compatible integer.")
    return method

class BaseVIModel(LLModel):
    
    NUM_PER_OUTPUT = 1
    
    """
    Base class for all variational inference models. Implements common VI functionality.
    """
    
    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None,
                 chain_order=None, chain_type=None, nums_per_output=NUM_PER_OUTPUT,
                 prior_mean=0.0, prior_mean_learnable=False,
                 prior_log_scale=1.0, prior_scale_learnable=False,
                 adaptive_l=False, l_max=12.0, n_samples=500,
                 incorrect_straight_sigmoid=False,
                 sigmoid_mc_computation=False, sigmoid_mc_n_samples=100):
        """
        Initialize the base variational inference model.

        Args:
            p (int): Dimensionality of the input features after processing by the backbone network.
            K (int): Number of outputs (attributes).
            beta (float, optional): Regularization parameter for the KL term in ELBO. Defaults to 1.0.
            intercept (bool, optional): Whether to include an intercept term. Defaults to False.
            backbone (torch.nn.Module, optional): Backbone network to transform input features. Defaults to None.
            chain_order (list of int, optional): Order of the chain. Defaults to None.
            chain_type (str, optional): Type of the chain. Must be one of ["logit", "probability", "prediction", "truth ground"]. Defaults to None.
            nums_per_output (int, optional): Number of outputs for each output. Defaults to 1.
            prior_mean (Union[float, List[float], List[torch.Tensor]], optional): Prior means. Defaults to 0.0.
            prior_mean_learnable (bool, optional): Whether prior mean is learnable. Defaults to False.
            prior_log_scale (Union[float, List[float]], optional): Prior log-scale. Defaults to torch.log(torch.tensor(1.0)).
            prior_scale_learnable (bool, optional): Whether prior scale is learnable. Defaults to False.
            adaptive_l (bool, optional): Whether to adaptively increase l during training. Defaults to False.
            l_max (float, optional): Maximum value of l for the proposed bound. Defaults to 12.0.
            n_samples (int, optional): Number of samples for Monte Carlo estimation. Defaults to 500.
            incorrect_straight_sigmoid (bool, optional): Whether to use incorrect straight-through sigmoid. Defaults to False.
            sigmoid_mc_computation (bool, optional): Whether to use MC for sigmoid computation. Defaults to False.
            sigmoid_mc_n_samples (int, optional): Samples for sigmoid MC estimation. Defaults to 100.

        """
        p_adjusted = super().__init__(p, K, beta, intercept, backbone, chain_order, chain_type, nums_per_output)
        self.p = p_adjusted  # Adjust for intercept
        
        # VI hyperparameters
        self.adaptive_l = adaptive_l
        self.l_max = l_max
        self.n_samples = n_samples
        self.l_terms = float(int(l_max / 2)) if adaptive_l else l_max
        
        # Prior parameters configuration
        self.prior_mean_learnable = prior_mean_learnable
        self.prior_scale_learnable = prior_scale_learnable
        
        # Sigmoid computation settings
        self.incorrect_straight_sigmoid = incorrect_straight_sigmoid
        self.sigmoid_mc_computation = sigmoid_mc_computation
        self.sigmoid_mc_n_samples = sigmoid_mc_n_samples
        
        # Initialize prior parameters
        self.prior_mean_list = self._initialize_prior_parameter(
            prior_mean if prior_mean is not None else torch.tensor(0.0), 
            self.p, self.prior_mean_learnable, default_value=0.0
        )
        self.prior_log_scale_list = self._initialize_prior_parameter(
            prior_log_scale if prior_log_scale is not None else torch.log(torch.tensor(1.0)), 
            self.p, self.prior_scale_learnable, default_value=torch.log(torch.tensor(1.0))
        )
        
        # Variational parameters will be initialized in subclasses
        self.posterior_mean_list = None
        self.posterior_log_scale_list = None
    
    def _initialize_prior_parameter(self, value, size, learnable, default_value):
        """
        Initialize a prior parameter as a tensor or ParameterList.

        Args:
            value (Union[float, torch.Tensor, List[float], List[torch.Tensor]]): Value to initialize the parameter.
            size (int): Size of the parameter.
            learnable (bool): Whether the parameter is learnable.
            default_value (float or torch.Tensor): Default value if `value` is None.

        Returns:
            Union[nn.Parameter, nn.ParameterList]: Initialized parameter. For now nn.ParamaterList.
        """
        if value is None:
            value = default_value

        if isinstance(value, (float, int)):
            tensor = self._expand_to_tensor(value, size)
            # return nn.Parameter(tensor, requires_grad=learnable) #FIXME: do we want that option?
            return nn.ParameterList([nn.Parameter(tensor, requires_grad=learnable) for _ in range(self.K)])
        elif isinstance(value, torch.Tensor):
            if value.dim() == 0:
                return nn.ParameterList([nn.Parameter(torch.full((size,), value.item(), dtype=torch.double), requires_grad=learnable) for _ in range(self.K)])
            elif value.dim() == 1:
                assert value.shape[0] == size, f"Tensor must have shape ({size},)"
                return nn.ParameterList([nn.Parameter(value, requires_grad=learnable) for _ in range(self.K)])
            elif value.dim() == 2:
                assert value.shape[1] == size, f"Tensor must have shape (self.K={self.K}, {size})"
                return nn.ParameterList([nn.Parameter(v, requires_grad=learnable) for v in value])
            else:
                raise ValueError("Tensor must be 0D, 1D, or 2D")
        elif isinstance(value, list):
            assert len(value) == self.K, f"List must have length {self.K}"
            if all(isinstance(v, torch.Tensor) for v in value):
                assert len(value) == size, f"List of tensors must have length {size}"
                return nn.ParameterList([nn.Parameter(v, requires_grad=learnable) for v in value])
            else:
                tensor = self._expand_to_tensor(value, size)
                # return nn.Parameter(tensor, requires_grad=learnable) #FIXME: do we want that option?
                return nn.ParameterList([nn.Parameter(tensor, requires_grad=learnable) for _ in range(self.K)])
        else:
            raise ValueError(f"Value must be a float, int, list of floats, or list of tensors. Got: {type(value)}")
    
    def _expand_to_tensor(self, value, size):
        """
        Expand a scalar or list to a tensor of the specified size.

        Args:
            value (Union[float, List[float]]): Scalar or list to expand.
            size (int): Size of the resulting tensor.

        Returns:
            torch.Tensor: Expanded tensor.
        """
        if isinstance(value, (float, int)):
            return torch.full((size,), float(value), dtype=torch.double)
        elif isinstance(value, list):
            assert len(value) == size, f"List must have length {size}"
            return torch.tensor(value, dtype=torch.double)
        else:
            raise ValueError("Value must be a float, int, or list of floats/ints")
    
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
            named_params.extend((f"posterior_mean_{i}", param) for i, param in enumerate(self.posterior_mean_list))
            named_params.extend((f"posterior_log_scale_{i}", param) for i, param in enumerate(self.posterior_log_scale_list))
            named_params.extend((f"prior_mean_{i}", param) for i, param in enumerate(self.prior_mean_list))
            named_params.extend((f"prior_log_scale_{i}", param) for i, param in enumerate(self.prior_log_scale_list))
            return named_params
        else:
            params = []
            if self.backbone:
                params.extend(self.backbone.parameters())
            params.extend(self.posterior_mean_list)
            params.extend(self.posterior_log_scale_list)
            params.extend(self.prior_mean_list)
            params.extend(self.prior_log_scale_list)
            return params

    @property
    def prior_scale(self):
        """Return the prior scale for the standard deviations."""
        return torch.exp(self.prior_log_scale)
    
    @property
    def prior_scale_list(self):
        """
        Return the prior standard deviations for each output.

        Returns:
            list: List of length K containing standard deviation vectors
        """
        return [torch.exp(log_scale) for log_scale in self.prior_log_scale_list]

    @property
    def posterior_scale_list(self):
        """
        Return the variational standard deviations for each output.

        Returns:
            list: List of length K containing standard deviation vectors
        """
        return [torch.exp(log_scale) for log_scale in self.posterior_log_scale_list]

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
        probs = self.forward(X_batch)
        return (probs > threshold).float(), probs
    
    @torch.no_grad()
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
        return self.compute_ELBO(X_batch, y_batch, data_size=X_batch.shape[0], other_beta=0.0)

    @torch.no_grad()
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
