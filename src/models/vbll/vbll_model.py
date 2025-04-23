import sys
import os
import torch
import torch.nn as nn
from ..generic import LLModel

"""## Load VBLL"""
def load_vbll(vbll_path):
    # sys.path.append(os.path.abspath(vbll_path)) # currently VBLL v0.4.0.2 after 0fcea86800d137a3d9f49853c2570e38462a1a4b
    try:
        sys.path.append(os.path.abspath(vbll_path))# currently VBLL v0.4.0.2 after 0fcea86800d137a3d9f49853c2570e38462a1a4b
        import vbll
        print("vbll found")
    except:
        print("vbll not found")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vbll"])
        import vbll
    return vbll


class VBLLModel(LLModel):
    """
    VBLLModel for multi-label classification using Variational Bayesian Last Layer (VBLL).
    """
    NUM_PER_OUTPUT = 2

    def __init__(
        self,
        p,
        K,
        backbone=None,
        chain_order=None,
        chain_type=None,
        nums_per_output=NUM_PER_OUTPUT,
        vbll_cfg=None,
        device=None,
    ):
        super().__init__(
            p=p,
            K=K,
            backbone=backbone,
            chain_order=chain_order,
            chain_type=chain_type,
            nums_per_output=nums_per_output,
        )
        self.device = device or torch.device("cpu")
        self.vbll_cfg = vbll_cfg or dict()
        vbll_package = load_vbll(getattr(self.vbll_cfg, "PATH", "."))
        self.num_classes_lst = [nums_per_output] * self.K
        self.head_initialization(vbll_package, self.vbll_cfg, self.num_classes_lst)

    def _make_disc_vbll_layer(self, vbll_package, num_hidden, num_classes, cfg):
        """ VBLL Discriminative classification head. """
        return vbll_package.DiscClassification( num_hidden,
                                        num_classes,
                                        self.beta,
                                        softmax_bound=cfg.SOFTMAX_BOUND,
                                        return_empirical=cfg.RETURN_EMPIRICAL,
                                        softmax_bound_empirical=cfg.SOFTMAX_BOUND_EMPIRICAL,
                                        parameterization = cfg.PARAMETRIZATION,
                                        return_ood=cfg.RETURN_OOD,
                                        prior_scale=cfg.PRIOR_SCALE,
                                        noise_label=cfg.NOISE_LABEL,
                                        wishart_scale=cfg.WISHART_SCALE,
                                        dof=cfg.DOF,
                                        cov_rank=cfg.COV_RANK,
                                       )

    def _make_gen_vbll_layer(self, vbll_package, num_hidden, num_classes, cfg):
        """ VBLL Generative classification head. """
        return vbll_package.GenClassification(  num_hidden,
                                        num_classes,
                                        self.beta,
                                        softmax_bound=cfg.SOFTMAX_BOUND,
                                        return_empirical=cfg.RETURN_EMPIRICAL,
                                        softmax_bound_empirical=cfg.SOFTMAX_BOUND_EMPIRICAL,
                                        parameterization = cfg.PARAMETRIZATION,
                                        return_ood=cfg.RETURN_OOD,
                                        prior_scale=cfg.PRIOR_SCALE,
                                        noise_label=cfg.NOISE_LABEL)
        
    def head_initialization(self, vbll_package, cfg, num_classes_lst):
        """
        Initialize the heads of the VBLL model.
        """
        vbll_type = getattr(cfg, "type", "disc")
        assert vbll_type in ["disc", "gen"], f"VBLL type {vbll_type} not supported. Use 'disc' or 'gen'."
        vbll_constr = self._make_gen_vbll_layer if vbll_type == "gen" else self._make_disc_vbll_layer
        heads = []
        for i_k, num_classes in enumerate(num_classes_lst):
            heads.append(vbll_constr(vbll_package, num_hidden=self.p, num_classes=num_classes, cfg=cfg))
        self.heads = nn.ModuleList(heads)
        

    def forward(self, X_batch, y_batch=None):
        """
        Forward pass: returns predicted probabilities for each output.
        """
        X_processed = self.process(X_batch)
        logits_list = []
        if self.chain:
            prev_list = []
            for i_k, head in enumerate(self.heads):
                X = self.chain.process_chain(X_processed, prev_list, i_k)
                y = y_batch[:, i_k] if y_batch is not None else None
                out = head(X)
                probs = out.predictive.probs
                logits = out.predictive.logits
                # with torch.no_grad():
                #     probs = torch.softmax(logits, dim=-1)
                logits_list.append(logits)
                prev_list = self.chain.update_chain(prev_list, logits, probs, y)         
        else:
            for i, head in enumerate(self.heads):
                out = head(X_processed)
                logits = out.predictive.logits
                logits_list.append(logits)
        logits = torch.stack(logits_list, dim=1)
        return logits
    
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
              the raw predictions (e.g., probabilities) for each class.
        Notes:
            - The method computes predictions for each of the K outputs by selecting the 
              class with the highest score (argmax) along the last dimension.
            - The `threshold` parameter is included for potential future use but is not 
              currently utilized in the method.
        """

        logits = self.forward(X_batch)
        all_preds = []
        for i_k in range(self.K):
            max_class = torch.argmax(logits[:, i_k, :], dim=-1)
            all_preds.append(max_class)
        return torch.stack(all_preds, dim=1), torch.softmax(logits, dim=-1)

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
        if self.chain:
            prev_list = []
            logits_list = []
            probs_list = []
            for i_k, head in enumerate(self.heads):
                X = self.chain.process_chain(X_processed, prev_list, i_k)
                y = y_batch[:, i_k] if y_batch is not None else None
                out = head(X)
                probs = out.predictive.probs
                logits = out.predictive.logits
                probs_list.append(probs)
                logits_list.append(logits)
                loss1 = out.train_loss_fn(y.long())
                loss += loss1
                if verbose:
                    print(f"head={i_k} loss={loss1:.2f}")
                prev_list = self.chain.update_chain(prev_list, logits, probs, y)
        else: 
            for i_k, head in enumerate(self.heads):
                y = y_batch[:, i_k] if y_batch is not None else None
                loss1 = head(X_processed).val_loss_fn(y.long())
                loss += loss1
                if verbose:
                    print(f"head={i_k} loss={loss1:.2f}")
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

        return self.train_loss(X_batch, y_batch, data_size=data_size, verbose=verbose)

    @torch.no_grad()
    def compute_negative_log_likelihood(self, X_batch, y_batch, mc = False, n_samples = 1000):
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
        
        X_processed = self.process(X_batch)
        nlls = []
        if self.chain:
            prev_list = []
            for i_k, head in enumerate(self.heads):
                X = self.chain.process_chain(X_processed, prev_list, i_k)
                y = y_batch[:, i_k] if y_batch is not None else None
                out = head(X)
                probs = out.predictive.probs
                logits = out.predictive.logits
                nll = out.val_loss_fn(y.long())
                nlls.append(nll)
                prev_list = self.chain.update_chain(prev_list, logits, probs, y)
        else:
            for head, y in zip(self.heads, y_batch.T):
                nll = head(X_processed).val_loss_fn(y.long())
                nlls.append(nll)
        return torch.stack(nlls)

    @torch.no_grad()
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
    