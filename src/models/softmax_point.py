import torch
import torch.nn as nn
import torch.nn.functional as F
from .generic import LLModel, LLModelCC

"""## Softmax-pointwise model"""
class SoftmaxPointwise(LLModel):
    def __init__(self, p, K, beta=0.0, num_classes_lst=None, intercept=False, backbone=None):
        """
        Initialize an instance of the SoftmaxPointwise class.

        Args:
            p (int): Dimensionality of the input features after processing by the backbone network.
            K (int): Number of outputs (attributes).
            beta (float, optional): Regularization parameter. Defaults to 0.0. For pointwise variant, equivalent to L2 regularization.
            num_classes_lst (list of int, optional): Number of classes for each output. Defaults to None (binary classification for all outputs).
            intercept (bool, optional): Whether to include an intercept term in the model. Defaults to False.
            backbone (torch.nn.Module, optional): Backbone network to transform input features. Defaults to None (no preprocessing).

        Notes:
            - If `num_classes_lst` is not provided, the model assumes binary classification for all outputs.
            - The `backbone` parameter allows for feature extraction or transformation before applying the model's heads.
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

    def get_learnable_parameters(self):
        params = []
        if self.backbone is not None:
            params += list(self.backbone.parameters())
        for head in self.heads:
            params += list(head.parameters())
        return nn.ParameterList(params)

    def make_output_layer(self, num_classes):
        return nn.Linear(self.p, num_classes).to(torch.double)

    # FIXME: is that regularization is ok?
    def regularization(self):
        """
        Compute the L2 regularization term.
        """
        log_prob = 0.0
        for head in self.heads:
               for param in head.parameters():
                    log_prob += torch.sum(param**2)
        return log_prob

    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the training loss for a batch of data using CrossEntropy for pointwise classification.

        Args:
            X_batch (torch.Tensor): Input batch of features with shape (batch_size, num_features).
            y_batch (torch.Tensor): Corresponding batch of target labels with shape (num_heads, batch_size).
            data_size (int, optional): Total size of the dataset. Defaults to the size of `X_batch`.
            verbose (bool, optional): If True, prints detailed loss information for each head and regularization loss. Defaults to False.

        Returns:
            torch.Tensor: The total loss, including the sum of individual head losses and the regularization loss.

        Notes:
            - The method processes the input features using the `process` method before computing the loss.
            - The loss for each head is computed using the `loss` function, and the total loss is the sum of all head losses.
            - If regularization is enabled (i.e., `self.beta` is non-zero), a regularization loss is added to the total loss.
            - The method ensures that the loss for each head is a scalar tensor.
        """
        data_size = data_size or X_batch.shape[0]
        X_processed = self.process(X_batch)

        total_loss = 0.0

        for i, (head,y) in enumerate(zip(self.heads,y_batch.T)):
            pred = head(X_processed)
            loss_head = self.loss(pred, y.to(torch.long))
            assert loss_head.shape == torch.Size([]), f"loss_head.shape={loss_head.shape} != ()" # loss_head.ndim == 0 alternatively
            total_loss += loss_head
            if verbose:
                print(f"head={i} loss={loss_head:.2f}")

        reg_loss = self.regularization() if self.beta else torch.tensor(0.0, dtype=torch.double, device=X_batch.device)
        if verbose:
            print(f"reg_loss={reg_loss:.2f}")
        total_loss += self.beta * reg_loss

        return total_loss

    @torch.no_grad()
    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False, other_beta=None):
        """
        Compute the test loss for a batch of data.

        Args:
            X_batch (torch.Tensor): Input batch of features with shape (batch_size, num_features).
            y_batch (torch.Tensor): Corresponding batch of target labels with shape (num_heads, batch_size).
            data_size (int, optional): Total size of the dataset. Defaults to the size of `X_batch`.
            verbose (bool, optional): If True, prints detailed loss information for each head and regularization loss. Defaults to False.
            other_beta (float, optional): Alternative regularization parameter for testing. Defaults to None (uses `self.beta`).

        Returns:
            torch.Tensor: The total test loss, including the sum of individual head losses and the regularization loss.

        Notes:
            - This method is a reference to `train_loss` and computes the loss in the same way.
            - The `other_beta` parameter allows for testing with a different regularization strength.
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

    @torch.no_grad()
    def predict(self, X_batch, threshold=None):
        """
        Predict the class for each output given the input data.
        Args:
            X_batch (torch.Tensor): Input data with shape (n_samples, input_dim), 
                where `n_samples` is the number of samples and `input_dim` is the dimensionality of the input features.
            threshold (float, optional): Threshold for binary classification. 
                Currently inactive and defaults to None.
        Returns:
            tuple: A tuple containing:
            - out (torch.Tensor): Predicted classes for each output with shape (n_samples, K), 
              where `K` is the number of outputs.
            - logits (torch.Tensor): Raw logits produced by the model with shape (n_samples, K, num_classes).
        Raises:
            AssertionError: If the shape of intermediate or final outputs does not match the expected dimensions.
        Notes:
            - The method processes the input data through the model to compute logits, 
              then determines the predicted class for each output by selecting the class with the highest score.
            - The `threshold` parameter is currently not used and is reserved for future binary classification use cases.
        """
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
        """
        Compute the confidence scores for the given predictions.
        Args:
            preds (torch.Tensor): A tensor containing the predictions, where the last dimension 
                      represents the class probabilities.
        Returns:
            torch.Tensor: A tensor containing the maximum confidence score for each prediction 
                  along the last dimension.
        """

        return torch.max(preds, dim=-1)[0]

    def compute_negative_log_likelihood(self, X_batch, y_batch, mc=True):
        """
        Compute the negative log likelihood (NLL) for the given data and predictions.

        Args:
            X_batch (torch.Tensor): Input data with shape (n_samples, input_dim), 
                where `n_samples` is the number of samples and `input_dim` is the dimensionality of the input features.
            y_batch (torch.Tensor): Target variables with shape (n_samples, K), 
                where `K` is the number of outputs (attributes).
            mc (bool, optional): Whether to use Monte Carlo estimation. Currently inactive. Defaults to True.

        Returns:
            torch.Tensor: A tensor containing the negative log likelihood for each output (attribute).

        Notes:
            - The method computes the NLL for each output using the CrossEntropy loss.
            - The logits for each output are processed through a softmax function to compute probabilities.
            - The NLL is calculated as the negative sum of the log probabilities of the true classes.
            - The method ensures that the NLL for each output is a scalar tensor.
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
            assert nll.shape == torch.Size([]), f"nll.shape={nll.shape} != ()"
            nlls.append(nll)
        return torch.stack(nlls)


""" # Softmax-pointwise CC model """
class SoftmaxPointwiseCC(LLModelCC, SoftmaxPointwise):
    def __init__(self, p, K, beta=0.0, intercept=False, backbone=None, num_classes_lst=None, chain_order=None, chain_type="logit"):
        """
        Initialize an instance of the SoftmaxPointwiseCC class.

        Args:
            p (int): Dimensionality of the input features after processing by the backbone network.
            K (int): Number of outputs (attributes).
            beta (float, optional): Regularization parameter. Defaults to 0.0. For the pointwise variant, equivalent to L2 regularization.
            intercept (bool, optional): Whether to include an intercept term in the model. Defaults to False.
            backbone (torch.nn.Module, optional): Backbone network to transform input features. Defaults to None (no preprocessing).
            num_classes_lst (list of int, optional): Number of classes for each output. Defaults to None (binary classification for all outputs).
            chain_order (list of int, optional): Order of the chain for the outputs. Defaults to None (sequential order from 0 to K-1).
            chain_type (str, optional): Type of the chain structure. Must be one of ["logit", "probability", "prediction"]. Defaults to "logit".

        Raises:
            AssertionError: If the length of `chain_order` does not match the number of outputs `K`.

        Notes:
            - The `chain_order` parameter determines the sequence in which the outputs are processed. 
              By default, it is a sequential list from 0 to K-1.
            - The `chain_type` parameter specifies how the chain is interpreted or used in the model. 
              For example, "logit" uses raw logits, "probability" uses softmax probabilities, and "prediction" uses the predicted class.
            - The `num_classes_lst` parameter allows specifying the number of classes for each output. 
              If not provided, binary classification is assumed for all outputs.
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

    def make_output_layer(self, num_classes, in_features=None):
        if in_features is None:
            in_features = self.p
        return nn.Linear(in_features, num_classes).to(torch.double)

    def forward(self, X_batch):
        """
        Forward pass through the model to compute logits for each output.

        Args:
            X_batch (torch.Tensor): Input data with shape (n_samples, input_dim), 
            where `n_samples` is the number of samples and `input_dim` is the dimensionality of the input features.

        Returns:
            torch.Tensor: A tensor containing the logits for each output with shape (n_samples, K, num_classes), 
            where `K` is the number of outputs and `num_classes` is the number of classes for each output.

        Notes:
            - The method processes the input data through the backbone network (if provided) and then applies 
              the chain structure to compute logits for each output sequentially.
            - The chain structure allows each output to depend on the previous outputs, based on the specified `chain_type`.
            - The `chain_type` parameter determines how the outputs are combined:
            - "logit": Uses raw logits from the previous outputs.
            - "probability": Uses softmax probabilities of the previous outputs.
            - "prediction": Uses the predicted class of the previous outputs.
            - The logits for all outputs are stacked along the second dimension to form the final output tensor.
        """
        X_processed = self.process(X_batch)
        prev_list = []
        logits = []
        for i_k, val_k in enumerate(self.chain_order):
            if i_k == 0:
                X = X_processed
            else:
                prev_cat = torch.cat(prev_list, dim=1)
                X = torch.cat((X_processed, prev_cat), dim=1)
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
        Compute the training loss for a batch of data using CrossEntropy for pointwise classification.

        Args:
            X_batch (torch.Tensor): Input batch of features with shape (batch_size, input_dim), 
                where `batch_size` is the number of samples and `input_dim` is the dimensionality of the input features.
            y_batch (torch.Tensor): Corresponding batch of target labels with shape (batch_size, K), 
                where `K` is the number of outputs (attributes).
            data_size (int, optional): Total size of the dataset. Defaults to the size of `X_batch`.
            verbose (bool, optional): If True, prints detailed loss information for each head and regularization loss. Defaults to False.

        Returns:
            torch.Tensor: The total loss, including the sum of individual head losses and the regularization loss.

        Notes:
            - The method processes the input features using the `process` method before computing the loss.
            - The loss for each head is computed using the `loss` function, and the total loss is the sum of all head losses.
            - If regularization is enabled (i.e., `self.beta` is non-zero), a regularization loss is added to the total loss.
            - The method ensures that the loss for each head is a scalar tensor.
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
            assert loss_head.shape == torch.Size([]), f"loss_head.shape={loss_head.shape} != (1)"
            logits.append(logit)
            total_loss += loss_head
            if verbose:
                print(f"head={i_k} loss={loss_head:.2f}")

        reg_loss = self.regularization() if self.beta else torch.tensor(0.0, dtype=torch.double, device=X_batch.device)
        if verbose:
            print(f"reg_loss={reg_loss:.2f}")
        total_loss += self.beta * reg_loss

        return total_loss

    @torch.no_grad()
    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the test loss for a batch of data.

        This method is a reference to the `train_loss` method and computes the loss in the same way.

        Args:
            X_batch (torch.Tensor): Input batch of features with shape (batch_size, input_dim), 
            where `batch_size` is the number of samples and `input_dim` is the dimensionality of the input features.
            y_batch (torch.Tensor): Corresponding batch of target labels with shape (batch_size, K), 
            where `K` is the number of outputs (attributes).
            data_size (int, optional): Total size of the dataset. Defaults to the size of `X_batch`.
            verbose (bool, optional): If True, prints detailed loss information for each head and regularization loss. Defaults to False.

        Returns:
            torch.Tensor: The total test loss, including the sum of individual head losses and the regularization loss.

        Notes:
            - This method is identical to `train_loss` but is used for testing purposes to maintain semantic clarity.
            - The loss for each head is computed using the `loss` function, and the total loss is the sum of all head losses.
            - If regularization is enabled (i.e., `self.beta` is non-zero), a regularization loss is added to the total loss.
        """
        return self.train_loss(X_batch, y_batch, data_size, verbose)
