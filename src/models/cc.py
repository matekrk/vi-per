import torch

class ChainOfClassifiers:
    """
    Utility class to handle chain-of-classifier logic for models with structured outputs.
    """

    valid_chain_types = ["logit", "probability", "prediction", "ground_truth"]

    def __init__(self, K, chain_order=None, chain_type="logit", nums_per_output=1):
        """
        Initialize the ChainOfClassifiers.

        Args:
            K (int): Number of outputs (attributes).
            chain_order (list of int or list of list of int or torch.Tensor, optional): Order of the chain. 
                If list of int, it represents a permuted lower triangular matrix without diagonal filled with 1s.
                If list of list of int or torch.Tensor, it explicitly indicates dependencies.
                Defaults to sequential order [0, 1, ..., K-1].
            chain_type (str, optional): Type of the chain. Must be one of ["logit", "probability", "prediction", "ground_truth"]. 
                Defaults to "logit".
        """
        self.K = K
        self.nums_per_output = nums_per_output
        self.extra_dim = K * nums_per_output
        self.chain_type = chain_type
        assert chain_type in self.valid_chain_types, \
            f"Invalid chain_type: {chain_type}. Must be one of ['logit', 'probability', 'prediction', 'ground_truth']."

        if chain_order is None:
            self.chain_order = self._generate_full_dependency_matrix_from_order(list(range(K)))
        elif isinstance(chain_order, list) and all(isinstance(x, int) for x in chain_order):
            self.chain_order = self._generate_full_dependency_matrix_from_order(chain_order)
        elif isinstance(chain_order, (list, torch.Tensor)):
            self.chain_order = torch.tensor(chain_order, dtype=torch.int)
            assert self.chain_order.shape == (K, K), \
                f"Explicit chain_order must be of shape ({K}, {K})."
            assert torch.all(self.chain_order.sum(dim=1) <= torch.arange(self.K)), \
                "Each row must have at most as many dependencies as its index in the chain order."
            assert torch.all(torch.matrix_power(self.chain_order, self.K) == 0), \
                "Dependency matrix must not have cycles."
        else:
            raise ValueError("Invalid chain_order format. Must be a list of int, list of list of int, or torch.Tensor.")

    def _generate_full_dependency_matrix_from_order(self, order):
        """
        Generate a dependency matrix from a list of integers representing the chain order.

        Args:
            order (list of int): Permuted order of the chain.

        Returns:
            torch.Tensor: Dependency matrix of shape (K, K).
        """
        dependency_matrix = torch.zeros((self.K, self.K), dtype=torch.int)
        for i, idx in enumerate(order):
            dependency_matrix[idx, :i] = 1
        return dependency_matrix

    def process_chain(self, X_processed, prev_list, i_k):
        """
        Process the chain by concatenating previous outputs with the input features.

        Args:
            X_processed (torch.Tensor): Processed input features. Shape (batch_size, input_dim).
            prev_list (list of torch.Tensor): List of previous outputs in the chain.
            i_k (int): Index of the current output in the chain.

        Returns:
            torch.Tensor: Concatenated input features for the current output.
        """
        batch_size, p = X_processed.shape
        if i_k == 0:
            return torch.cat((X_processed, torch.zeros(batch_size, self.K * self.nums_per_output, device=X_processed.device)), dim=1)
        
        dependencies = self.chain_order[i_k]
        prev_cat = torch.zeros(batch_size, self.extra_dim, device=X_processed.device)
        
        for j in range(self.K):
            if dependencies[j] == 1:
                prev_cat[:, j*self.nums_per_output: (j+1)*self.nums_per_output] = prev_list[j].view(batch_size, self.nums_per_output)
        
        return torch.cat((X_processed, prev_cat), dim=1)

    def update_chain(self, prev_list, logits, probs, true_label, threshold=0.5):
        """
        Update the chain with the current output based on the chain type.

        Args:
            prev_list (list of torch.Tensor): List of previous outputs in the chain.
            logits (torch.Tensor): Logits for the current output. Shape (batch_size,).
            probs (torch.Tensor): Probabilities for the current output. Shape (batch_size,).

        Returns:
            list of torch.Tensor: Updated list of previous outputs in the chain.
        """
        if self.chain_type == "logit":
            prev_list.append(logits.unsqueeze(1))
        elif self.chain_type == "probability":
            prev_list.append(probs.unsqueeze(1))
        elif self.chain_type == "prediction":
            prev_list.append((probs > threshold).float().unsqueeze(1))
        elif self.chain_type == "ground_truth":
            prev_list.append(true_label.unsqueeze(1))
        return prev_list
