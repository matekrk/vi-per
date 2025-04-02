import torch
import torch.nn as nn

"""# Base models"""

"""## Base vanilla model"""
class LLModel(nn.Module):
    """
    Base class for all models. It defines the basic structure and methods that all models should implement.
    """
    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None):
        """
        Initialize the LLModel.

        Args:
            p (int): Dimensionality of the input features after processing by the backbone network.
            K (int): Number of outputs (attributes).
            beta (float, optional): Regularization parameter. Defaults to 1.0.
            intercept (bool, optional): Whether to include an intercept term in the model. Defaults to False.
            backbone (torch.nn.Module, optional): Backbone network to transform input features. Defaults to None (no preprocessing).

        Returns:
            int: Adjusted input dimensionality (p), incremented by 1 if intercept is True.
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
    """
    Base class for all chain models. It defines the basic structure and methods that all chain models should implement.
    """
    def __init__(self, p, K, beta=1.0, intercept=False, backbone=None, chain_order=None, chain_type="logit"):
        """
        Initialize an instance of the LLModelCC class.

        Args:
            p (int): Dimensionality of the input features after processing by the backbone network.
            K (int): Number of outputs (attributes).
            beta (float, optional): Regularization parameter. Defaults to 1.0.
            intercept (bool, optional): Whether to include an intercept term in the model. Defaults to False.
            backbone (torch.nn.Module, optional): Backbone network to transform input features. Defaults to None (no preprocessing).
            chain_order (list of int, optional): Order of the chain. Defaults to None (sequential order from 0 to K-1).
            chain_type (str, optional): Type of the chain. Must be one of ["logit", "probability", "prediction", "true"]. Defaults to "logit".

        Raises:
            AssertionError: If the length of `chain_order` does not match the number of outputs `K`.

        Notes:
            - The `chain_order` parameter determines the sequence in which the outputs are processed. 
              By default, it is a sequential list from 0 to K-1.
            - The `chain_type` parameter specifies how the chain is interpreted or used in the model.
        """
        LLModel.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone)
        self.chain_order = chain_order if chain_order is not None else list(range(K))  # TODO: [backlog] make graph instead of list
        assert len(self.chain_order) == self.K, f"chain_order must have length {self.K}"
        print("[LLModelCC] chain_order=", self.chain_order)
        self.chain_order = torch.tensor(self.chain_order, dtype=torch.long)
        self.chain_type = chain_type

    def process_chain(self, X_batch):
        raise NotImplementedError("[LLModelCC] process_chain not implemented")
