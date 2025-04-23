import torch
import torch.nn as nn
from .cc import ChainOfClassifiers

"""# Base models"""

"""## Base vanilla model"""
class LLModel(nn.Module):
    """
    Base class for all models. It defines the basic structure and methods that all models should implement.
    """
    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None,
                 chain_order=None, chain_type=None, nums_per_output=1):
        """
        Initialize the LLModel.

        Args:
            p (int): Dimensionality of the input features after processing by the backbone network.
            K (int): Number of outputs (attributes).
            beta (float, optional): Regularization parameter. Defaults to 1.0.
            intercept (bool, optional): Whether to include an intercept term in the model. Defaults to False.
            backbone (torch.nn.Module, optional): Backbone network to transform input features. Defaults to None (no preprocessing).
            chain_order (list of int, optional): Order of the chain. Defaults to None.
            chain_type (str, optional): Type of the chain. Must be one of ["logit", "probability", "prediction", "ground_truth"]. Defaults to None.
            nums_per_output (int, optional): Number of outputs for each output. Defaults to 1 (binary classification, logistic).
        Returns:
            int: Adjusted input dimensionality (p), incremented by 1 if intercept is True.
        """
        super().__init__()
        print(f"[LLModel] beta={beta} input_dim={p} output_dim={K} intercept={intercept} chain_order={chain_order} chain_type={chain_type}")
        
        # Initialize chain-of-classifier functionality if chain_order is provided
        assert chain_order is None or chain_type in ChainOfClassifiers.valid_chain_types, \
            f"Invalid chain_type: {chain_type}. Must be one of {ChainOfClassifiers.valid_chain_types}."
        self.chain = ChainOfClassifiers(K, chain_order, chain_type, nums_per_output) if chain_order else None
        self.chain_type = chain_type
        self.intercept = intercept
        p_adjusted = p
        if intercept:
            p_adjusted += 1
        self.K = K
        if self.chain is not None:
            p_adjusted += self.chain.extra_dim
        self.p = p_adjusted
        self.backbone = backbone
        self.beta = beta
        return p_adjusted
    
    def get_learnable_parameters(self, named=True):
        """
        Get learnable parameters of the model.

        Args:
            named (bool, optional): Whether to return named parameters. Defaults to True.

        Returns:
            nn.ParameterList: List of learnable parameters.
        """
        if named:
            return nn.ParameterList(self.named_parameters())
        else:
            return nn.ParameterList(self.parameters())

    def process(self, X_batch):
        """
        Process the input batch through the backbone network.
        
        Args:
            X_batch (torch.Tensor): Input batch of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Processed batch of shape (batch_size, p).
        """
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
    
    def forward(self, X):
        raise NotImplementedError("[LLModel] forward mechanism not implemented")
    
    def predict(self, X):
        raise NotImplementedError("[LLModel] predict mechanism not implemented")
    
    def compute_negative_log_likelihood(self, X_batch, y_batch):
        raise NotImplementedError("[LLModel] compute_negative_log_likelihood not implemented")
    
    
