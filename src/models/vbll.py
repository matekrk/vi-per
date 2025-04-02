import sys
import os
import torch
import torch.nn as nn
from .generic import LLModel, LLModelCC

"""## Load VBLL"""
def load_vbll(vbll_path):
    sys.path.append(os.path.abspath(vbll_path)) # currently VBLL v0.4.0.2 after 0fcea86800d137a3d9f49853c2570e38462a1a4b
    try:
        import vbll
        print("vbll found")
    except:
        print("vbll not found")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vbll"])
        import vbll

class SoftmaxVBLL(LLModel):
    """
    SoftmaxVBLL model for multi-label classification using VBLL (Variational Bayesian Last Layer).
    """
    def __init__(self, p, K, beta, vbll_cfg, num_classes_lst=None, intercept=False, backbone=None):
        """
        Initialize an instance of the SoftmaxVBLL class.

        Args:
            p (int): Dimensionality of the input features after processing by the backbone network.
            K (int): Number of outputs (attributes).
            beta (float): Regularization parameter for the VBLL model.
            vbll_cfg (object): Configuration object for VBLL, containing parameters such as type, path, and regularization settings.
            num_classes_lst (list of int, optional): Number of classes for each output. Defaults to None, which assumes binary classification for all outputs.
            intercept (bool, optional): Whether to include an intercept term in the model. Defaults to False.
            backbone (torch.nn.Module, optional): Backbone network to transform input features. Defaults to None (no preprocessing).

        Notes:
            - The `vbll_cfg` parameter should be a configuration object compatible with the VBLL library.
            - If `num_classes_lst` is not provided, the model assumes binary classification for all outputs.
            - The `backbone` parameter allows for the integration of a feature extraction network before applying the VBLL layers.
        """
        p = super().__init__(p, K, beta, intercept=intercept, backbone=backbone)
        print(f"[SoftmaxVBLL] vbll_cfg={vbll_cfg}")
        vbll_cfg.REGULARIZATION_WEIGHT = self.beta
        self.vbll_cfg = vbll_cfg

        if num_classes_lst is None:
            self.num_classes_single = 2
            self.num_classes_lst = [self.num_classes_single] * self.K
        else:
            self.num_classes_single = num_classes_lst[0]
            self.num_classes_lst = num_classes_lst

        self.heads = nn.ModuleList([self.make_output_layer(num_hidden=self.p, num_classes=self.num_classes_lst[k]) for k in range(self.K)])

    def get_learnable_parameters(self):
        params = []
        if self.backbone is not None:
            params += list(self.backbone.parameters())
        for head in self.heads:
            params += list(head.parameters())
        return nn.ParameterList(params)

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
                                        return_empirical=cfg.RETURN_EMPIRICAL,
                                        softmax_bound_empirical=cfg.SOFTMAX_BOUND_EMPIRICAL,
                                        parameterization = cfg.PARAMETRIZATION,
                                        return_ood=cfg.RETURN_OOD,
                                        prior_scale=cfg.PRIOR_SCALE,
                                        noise_label=cfg.NOISE_LABEL
                                       )

    def _make_gen_vbll_layer(self, num_hidden, num_classes, cfg):
        """ VBLL Generative classification head. """
        import vbll
        return vbll.GenClassification(  num_hidden,
                                        num_classes,
                                        self.beta,
                                        softmax_bound=cfg.SOFTMAX_BOUND,
                                        return_empirical=cfg.RETURN_EMPIRICAL,
                                        softmax_bound_empirical=cfg.SOFTMAX_BOUND_EMPIRICAL,
                                        parameterization = cfg.PARAMETRIZATION,
                                        return_ood=cfg.RETURN_OOD,
                                        prior_scale=cfg.PRIOR_SCALE,
                                        noise_label=cfg.NOISE_LABEL)

    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the training loss for a batch of data.
        Args:
            X_batch (torch.Tensor): A batch of input features with shape (batch_size, num_features).
            y_batch (torch.Tensor): A batch of target labels with shape (batch_size, num_outputs).
            data_size (int, optional): The total size of the dataset. Defaults to the size of `X_batch`.
            verbose (bool, optional): If True, prints the loss for each head during computation. Defaults to False.
        Returns:
            torch.Tensor: The total training loss computed across all heads.
        Notes:
            - The method processes the input features using the `process` method before computing the loss.
            - Each head in `self.heads` is responsible for computing the loss for one output dimension.
            - The `train_loss_fn` method of each head is used to compute the loss for the corresponding target.
            - If `verbose` is enabled, the loss for each head is printed during the computation.
        """

        data_size = data_size or X_batch.shape[0]
        X_processed = self.process(X_batch)

        loss = 0.
        for i, (head, y) in enumerate(zip(self.heads, y_batch.T)):
            loss1 = head(X_processed).train_loss_fn(y.long())
            loss += loss1
            if verbose:
                print(f"head={i} loss={loss1:.2f}")
        return loss

    @torch.no_grad()
    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the test loss for a batch of data.
        Args:
            X_batch (torch.Tensor): Input features for the batch, with shape (batch_size, num_features).
            y_batch (torch.Tensor): Target labels for the batch, with shape (batch_size, num_outputs).
            data_size (int, optional): Total size of the dataset. Defaults to the batch size if not provided.
            verbose (bool, optional): If True, prints the loss for each head during computation. Defaults to False.
        Returns:
            float: The total loss computed across all heads for the given batch.
        Notes:
            - The method processes the input features using the `process` method before computing the loss.
            - Each head in `self.heads` corresponds to a specific output, and its loss is computed using its `val_loss_fn`.
            - If `verbose` is enabled, the loss for each head is printed during the computation.
        """

        data_size = data_size or X_batch.shape[0]
        X_processed = self.process(X_batch)

        loss = 0.
        for i, (head, y) in enumerate(zip(self.heads, y_batch.T)):
            loss1 = head(X_processed).val_loss_fn(y.long())
            loss += loss1
            if verbose:
                print(f"head={i} loss={loss1:.2f}")
        return loss

    def forward(self, X_batch):
        """
        Perform a forward pass to predict probabilities for each output given input data.

        Args:
            X_batch (torch.Tensor): Input data with shape (n_samples, input_dim).

        Returns:
            torch.Tensor: Predicted probabilities for each output. 
                  Shape (n_samples, K, num_classes), where `num_classes` is the number of classes for each output.

        Notes:
            - The method processes the input features using the `process` method before passing them to the heads.
            - Each head in `self.heads` corresponds to a specific output and computes the predictive distribution.
            - The predicted probabilities for all outputs are stacked along the second dimension.
        """
        X_processed = self.process(X_batch)

        probs = []
        for head in self.heads:
            distr = head(X_processed).predictive
            probs = distr.probs
            probs.append(probs)
        return torch.stack(probs, dim=1)

    @torch.no_grad()
    def predict(self, X_batch, threshold=None):
        """
        Generate predictions for a given batch of input data.
        Args:
            X_batch (torch.Tensor): Input tensor of shape (batch_size, input_dim), 
                where `batch_size` is the number of samples in the batch and `input_dim` 
                is the dimensionality of the input features.
            threshold (float, optional): Threshold value for prediction. Defaults to None. 
                (Currently unused in the implementation.)
        Returns:
            tuple:
            - torch.Tensor: A tensor of shape (batch_size, K) containing the predicted 
              class indices for each of the K outputs.
            - torch.Tensor: A tensor of shape (batch_size, K, num_classes) containing 
              the raw predictions (e.g., logits or probabilities) for each class.
        Notes:
            - The method computes predictions for each of the K outputs by selecting the 
              class with the highest score (argmax) along the last dimension.
            - The `threshold` parameter is included for potential future use but is not 
              currently utilized in the method.
        """

        preds = self.forward(X_batch)
        all_preds = []
        for i_k in range(self.K):
            max_class = torch.argmax(preds[:, i_k, :], dim=-1)
            all_preds.append(max_class)
        return torch.stack(all_preds, dim=1), preds

    def compute_negative_log_likelihood(self, X, y, mc = False, n_samples = 1000):
        """
        Compute the negative log-likelihood (NLL) for the given input data and targets.
        Args:
            X (torch.Tensor): Input data to the model. Shape should be compatible with the model's processing requirements.
            y (torch.Tensor): Target labels corresponding to the input data. Expected shape is (N, K), where N is the number of samples and K is the number of outputs.
            mc (bool, optional): Flag to indicate whether to use Monte Carlo sampling. Defaults to False.
            n_samples (int, optional): Number of Monte Carlo samples to use if `mc` is True. Defaults to 1000.
        Returns:
            torch.Tensor: A tensor containing the negative log-likelihood values for each output head. Shape is (K,).
        Notes:
            - The method processes the input `X` using the `process` method before computing the NLL.
            - The NLL is computed for each output head in the model, and the results are stacked into a single tensor.
            - If `mc` is True, Monte Carlo sampling is used to approximate the NLL, but this functionality is not implemented in the provided code.
        """
        
        X_processed = self.process(X)
        nlls = []
        for head, y in zip(self.heads, y.T):
            nll = head(X_processed).val_loss_fn(y.long())
            nlls.append(nll)
        return torch.stack(nlls)
    
    def get_confidences(self, preds):
        """
        Compute the confidence scores for the given predictions.
        Args:
            preds (torch.Tensor): A tensor containing the predictions. 
                The last dimension is expected to represent the class probabilities or scores.
        Returns:
            torch.Tensor: A tensor containing the maximum confidence score for each prediction 
                along the last dimension.
        Notes:
            - This method assumes that the input `preds` is a tensor where higher values 
              indicate higher confidence for a particular class.
            - The confidence score is computed as the maximum value along the last dimension.
        """
        return torch.max(preds, dim=-1)[0]


""" ## Softmax VBLL CC model """
class SoftmaxVBLLCC(LLModelCC, SoftmaxVBLL):
    """
    SoftmaxVBLLCC model for multi-label classification using VBLL (Variational Bayesian Last Layer) with a chain structure.
    """
    def __init__(self, p, K, beta, vbll_cfg, num_classes_lst=None, intercept=False, backbone=None, chain_order=None, chain_type="probability"):
        """
        Initialize an instance of the SoftmaxVBLLCC class.
        Args:
            p (int): Dimensionality of the input features after processing by the backbone network.
            K (int): Number of outputs (attributes).
            beta (float): Regularization parameter.
            vbll_cfg (dict): Configuration dictionary for the VBLL model.
            num_classes_lst (list of int, optional): List specifying the number of classes for each output. Defaults to None.
            intercept (bool, optional): Whether to include an intercept term in the model. Defaults to False.
            backbone (torch.nn.Module, optional): Backbone network to transform input features. Defaults to None.
            chain_order (list of int, optional): Order of the chain. Defaults to None (sequential order from 0 to K-1).
            chain_type (str, optional): Type of the chain. Must be one of ["logit", "probability", "prediction", "true"]. Defaults to "probability".
        Notes:
            - This class combines functionality from both `LLModelCC` and `SoftmaxVBLL`.
            - The `chain_order` parameter determines the sequence in which the outputs are processed.
            - The `chain_type` parameter specifies how the chain is interpreted or used in the model.
            - The `heads` attribute is a list of output layers, one for each output, reordered according to `chain_order`.
            - The `params` attribute aggregates the parameters of the backbone and the output layers.
        """
        
        LLModelCC.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone, chain_order=chain_order, chain_type=chain_type)
        SoftmaxVBLL.__init__(self, p, K, beta=beta, intercept=intercept, backbone=backbone, num_classes_lst=num_classes_lst, vbll_cfg=vbll_cfg)
        print(f"[SoftmaxVBLLCC]")

        self.heads = nn.ModuleList([self.make_output_layer(num_hidden=self.p+sum(self.num_classes_lst[:k]), num_classes=self.num_classes_lst[k]) for k in range(self.K)])
        self.heads = nn.ModuleList([self.heads[(self.chain_order == i_k).nonzero().item()] for i_k, val_k in enumerate(self.chain_order)])

        self.params = nn.ParameterList()
        if self.backbone is not None:
            self.params += list(self.backbone.parameters())
        for head in nn.ModuleList(self.heads):
            self.params += list(head.parameters())
    
    def train_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the training loss for a batch of data.
        Args:
            X_batch (torch.Tensor): Input batch of features with shape (batch_size, num_features).
            y_batch (torch.Tensor): Corresponding batch of target labels with shape (batch_size, num_outputs).
            data_size (int, optional): Total size of the dataset. Defaults to the size of `X_batch`.
            verbose (bool, optional): If True, prints the loss for each head during computation. Defaults to False.
        Returns:
            torch.Tensor: The total training loss computed across all heads.
        Notes:
            - The method processes the input features using the `process` method before computing the loss.
            - The model consists of multiple heads, and the loss is computed sequentially for each head based on the chain order.
            - For the first head, the loss is computed directly from the processed input features.
              For subsequent heads, the loss is computed using both the processed input features and the logits from previous heads.
            - The `chain_order` attribute determines the sequence in which the heads are processed.
            - If `verbose` is True, the loss for each head is printed during computation.
        """
        data_size = data_size or X_batch.shape[0]
        X_processed = self.process(X_batch)

        logits = []
        loss = 0.
        for i_k, val_k in enumerate(self.chain_order):
            relevant_head = val_k
            if i_k == 0:
                out = self.heads[relevant_head](X_processed)
                loss1 = out.train_loss_fn(y_batch.T[relevant_head].long())
            else:
                prev_logits = torch.cat(logits, dim=1)
                out = self.heads[relevant_head](torch.cat((X_processed, prev_logits), dim=1))
                loss1 = out.train_loss_fn(y_batch.T[relevant_head].long())
            logits.append(out.predictive.probs)
            loss += loss1
            if verbose:
                print(f"head={i_k} loss={loss1:.2f}")
        return loss

    @torch.no_grad()
    def test_loss(self, X_batch, y_batch, data_size=None, verbose=False):
        """
        Compute the test loss for a batch of data.
        Args:
            X_batch (torch.Tensor): Input batch of data with shape (batch_size, num_features).
            y_batch (torch.Tensor): Corresponding labels for the input batch with shape (batch_size, num_outputs).
            data_size (int, optional): Total size of the dataset. Defaults to the size of `X_batch`.
            verbose (bool, optional): If True, prints the loss for each head during computation. Defaults to False.
        Returns:
            torch.Tensor: The total loss computed across all heads in the chain.
        Notes:
            - The method processes the input data using the `process` method before computing the loss.
            - The computation iterates over the chain order (`self.chain_order`), where each head processes the data sequentially.
            - For the first head, only the processed input is used. For subsequent heads, the predictions from previous heads are concatenated with the processed input.
            - The loss for each head is computed using the `val_loss_fn` method of the respective head.
            - If `verbose` is True, the loss for each head is printed during the computation.
        """
        data_size = data_size or X_batch.shape[0]
        X_processed = self.process(X_batch)

        logits = []
        loss = 0.
        for i_k, val_k in enumerate(self.chain_order):
            relevant_head = val_k
            if i_k == 0:
                out = self.heads[relevant_head](X_processed)
                loss1 = self.heads[relevant_head](X_processed).val_loss_fn(y_batch.T[relevant_head].long())
            else:
                prev_logits = torch.cat(logits, dim=1)
                out = self.heads[relevant_head](torch.cat((X_processed, prev_logits), dim=1))
                loss1 = self.heads[relevant_head](torch.cat((X_processed, prev_logits), dim=1)).val_loss_fn(y_batch.T[relevant_head].long())
            logits.append(out.predictive.probs)
            loss += loss1
            if verbose:
                print(f"head={i_k} loss={loss1:.2f}")
            return loss
    
    def forward(self, X):
        """
        Perform a forward pass through the model.
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, input_dim), where `input_dim` 
                      corresponds to the dimensionality of the input features.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, K, num_classes), where `K` is the 
                  number of outputs (attributes) and `num_classes` is the number of 
                  classes for each output.
        Notes:
            - The method processes the input `X` using a backbone network and computes the logits 
              for each output in the chain order specified by `self.chain_order`.
            - For the first output in the chain, the logits are computed directly from the processed 
              input. For subsequent outputs, the logits are computed using both the processed input 
              and the concatenated logits of the previous outputs.
            - The logits for each output are stacked along a new dimension to form the final output tensor.
        """
        X_processed = self.process(X)
        logits = []
        for i_k, val_k in enumerate(self.chain_order):
            if i_k == 0:
                logit = self.heads[val_k](X_processed).predictive.probs
            else:
                logit = self.heads[val_k](torch.cat((X_processed, torch.cat(logits, dim=1)), dim=1)).predictive.probs
            logits.append(logit)
        return torch.stack(logits, dim=1)
    
    def compute_negative_log_likelihood(self, X, y, mc=False, n_samples=1000):
        """
        Compute the negative log-likelihood (NLL) for the given input data and labels.
        Args:
            X (torch.Tensor): Input features of shape (batch_size, num_features).
            y (torch.Tensor): Ground truth labels of shape (batch_size, num_outputs).
            mc (bool, optional): Whether to use Monte Carlo sampling. Defaults to False.
            n_samples (int, optional): Number of Monte Carlo samples to draw if `mc` is True. Defaults to 1000.
        Returns:
            torch.Tensor: A tensor containing the negative log-likelihood values for each output in the chain.
        Notes:
            - The method processes the input features using the `process` method before computing the NLL.
            - The computation follows a chain structure defined by `self.chain_order`, where each output depends on the logits of the previous outputs.
            - The `val_loss_fn` method of each head is used to compute the NLL for the corresponding output.
            - The logits for each output are stored and concatenated to form the input for subsequent outputs in the chain.
        """
        X_processed = self.process(X)
        nlls = []
        logits = []
        for i_k, val_k in enumerate(self.chain_order):
            relevant_head = val_k
            head = self.heads[relevant_head]
            y_k = y[:, relevant_head]
            if i_k == 0:
                out = head(X_processed)
            else:
                out = head(torch.cat((X_processed, torch.cat(logits, dim=1)), dim=1))
            nll = out.val_loss_fn(y_k.long())
            nlls.append(nll)
            logit = out.predictive.probs
            logits.append(logit)
        return torch.stack(nlls)
