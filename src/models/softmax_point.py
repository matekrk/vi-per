import torch
from .generic import LLModel
import torch.nn as nn

class SoftmaxModel(LLModel):

    NUM_PER_OUTPUT = 2
    
    """
    Softmax-based pointwise model for multi-class classification tasks.
    Supports independent outputs or a chain of classifiers and integrates with a backbone network if provided.
    """
    def __init__(self, p, K, num_classes_lst=None, beta=1.0, intercept=False, backbone=None, chain_type="logit", chain_order=None, nums_per_output=NUM_PER_OUTPUT):
        """
        Initialize the SoftmaxModel.

        Args:
            p (int): Dimensionality of the input features after processing by the backbone network.
            K (int): Number of outputs (attributes).
            num_classes_lst (list of int, optional): Number of classes for each output. Defaults to None (binary classification for all outputs).
            beta (float, optional): Regularization parameter. Defaults to 1.0.
            intercept (bool, optional): Whether to include an intercept term in the model. Defaults to False.
            backbone (torch.nn.Module, optional): Backbone network to transform input features. Defaults to None.
            chain (bool, optional): Whether to use a chain of classifiers. Defaults to False.
            chain_type (str, optional): Type of the chain. Must be one of ["logit", "probability", "prediction", "ground_truth"]. Defaults to "logit".
            chain_order (list or torch.Tensor, optional): Order of the chain. Defaults to sequential order.
            nums_per_output (int, optional): Number of outputs for each output. Defaults to 2 (binary classification, softmax).

        Notes:
            - If `num_classes_lst` is not provided, binary classification (2 classes) is assumed for all outputs.
            - The `backbone` parameter allows for feature extraction or transformation before applying the model's heads.
            - If `chain` is True, the outputs of one classifier are used as inputs to the next.
        """
        p_adjusted = super().__init__(p, K, beta, intercept, backbone, chain_order, chain_type, nums_per_output)
        print(f"[SoftmaxPointwiseModel] input_dim={p_adjusted} output_dim={K} beta={beta} chain_type={chain_type} chain_type={chain_type}")

        self.loss = nn.CrossEntropyLoss(reduction='mean')

        # Initialize number of classes for each output
        if num_classes_lst is None:
            self.num_classes_lst = [2] * K
            # Default to binary classification. 
            # Assume 2 classes for each output.
            # And at least for now the same number of classes for each output.
        else:
            assert len(num_classes_lst) == K, f"num_classes_lst must have length K={K}."
            self.num_classes_lst = num_classes_lst

        # Define output layers (one per output)
        self.heads = nn.ModuleList([self._make_output_layer(p_adjusted, num_classes) for num_classes in self.num_classes_lst])
        self.loss = nn.CrossEntropyLoss(reduction='mean')

        # TODO: add custom initialization

    def _make_output_layer(self, input_dim, num_classes):
        """
        Create a linear output layer for a given number of classes.

        Args:
            input_dim (int): Dimensionality of the input features.
            num_classes (int): Number of classes for the output.

        Returns:
            nn.Module: A linear layer for the output.
        """
        return nn.Linear(input_dim, num_classes).to(torch.double)

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
            for i, head in enumerate(self.heads):
                named_params.extend((f"head_{i}", param) for param in head.parameters())
            return named_params
        else:
            params = []
            if self.backbone:
                params.extend(self.backbone.parameters())
            for head in self.heads:
                params.extend(head.parameters())
            return params

    def forward(self, X_batch, y_batch=None):
        """
        Perform a forward pass through the model.

        Args:
            X_batch (torch.Tensor): Input batch of data with shape (batch_size, input_dim).
            y_batch (torch.Tensor, optional): Ground truth labels for the batch. Required if chain_type is "ground_truth".

        Returns:
            torch.Tensor: Logits for each output with shape (batch_size, K, max_num_classes).
        """
        X_processed = self.process(X_batch)
        logits = []
        prev_list = []

        for i_k, head in enumerate(self.heads):
            if self.chain:
                X_current = self.chain.process_chain(X_processed, prev_list, i_k)
            else:
                X_current = X_processed

            logit = head(X_current)
            logits.append(logit.unsqueeze(1))

            if self.chain:
                y = y_batch[:, i_k] if y_batch is not None else None
                prev_list = self.chain.update_chain(prev_list, logit, torch.softmax(logit, dim=-1), y)

        logits = torch.cat(logits, dim=1)
        return logits

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
        logits = self.forward(X_batch, y_batch)
        total_loss = 0.0

        for i, (logit, y) in enumerate(zip(logits.transpose(1, 0), y_batch.T)):
            loss_head = self.loss(logit, y.to(torch.long))
            total_loss += loss_head
            if verbose:
                print(f"[Train Loss] Head {i}: {loss_head.item()}")

        reg_loss = self.regularization() / data_size if self.beta else torch.tensor(0.0, device=logits.device)
        if verbose:
            print(f"[Train Loss] Regularization: {reg_loss.item()}")
        return total_loss + self.beta * reg_loss

    def regularization(self):
        """
        Compute the L2 regularization term.

        Returns:
            torch.Tensor: The computed regularization term as a scalar tensor.
        """
        reg_loss = 0.0
        for head in self.heads:
            for param in head.parameters():
                reg_loss += torch.sum(param ** 2)
        return reg_loss

    @torch.no_grad()
    def predict(self, X_batch, threshold = None):
        """
        Predict class probabilities and class labels for the input data.

        Args:
            X_batch (torch.Tensor): Input data.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Predicted class labels with shape (batch_size, K).
                - torch.Tensor: Predicted probabilities with shape (batch_size, K, max_num_classes).
        """
        logits = self.forward(X_batch)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return preds, probs
    
    def compute_negative_log_likelihood(self, X_batch, y_batch, mc=True):
        """
        Compute the negative log likelihood (NLL) for the given data and predictions.

        Args:
            X_batch (torch.Tensor): Input data with shape (n_samples, input_dim).
            y_batch (torch.Tensor): Target variables with shape (n_samples, K).
            mc (bool, optional): Whether to use Monte Carlo estimation. Currently inactive. Defaults to True.

        Returns:
            torch.Tensor: A tensor containing the negative log likelihood for each output (attribute).
        """
        logits = self.forward(X_batch)
        nlls = []
        for val_k in range(self.K):
            y = y_batch.T[val_k]
            logit = logits[:, val_k, :]
            probabilities = torch.softmax(logit, dim=1)
            true_class_probs = probabilities.gather(1, y.unsqueeze(1).to(torch.long)).squeeze()
            log_likelihood = torch.log(true_class_probs)
            nll = -log_likelihood.sum()
            nlls.append(nll)
        return torch.stack(nlls)

    def get_confidences(self, probs):
        """
        Compute the confidence scores for the given probabilities.

        Args:
            probs (torch.Tensor): A float tensor, where the last dimension 
                                represents the class probabilities.

        Returns:
            torch.Tensor: A tensor containing the maximum confidence score for each prediction 
                        along the last dimension.
        """
        return torch.max(probs, dim=-1)[0]

    @torch.no_grad()
    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the test loss for a batch of data.

        Args:
            X_batch (torch.Tensor): Input batch of features.
            y_batch (torch.Tensor): Corresponding batch of target labels.
            data_size (int, optional): Total size of the dataset. Defaults to None.
            verbose (bool, optional): If True, prints detailed loss information. Defaults to False.

        Returns:
            torch.Tensor: The total test loss.
        """
        return self.train_loss(X_batch, y_batch, data_size, verbose)
