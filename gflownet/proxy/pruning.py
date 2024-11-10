from typing import Optional

import numpy as np
import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.envs.pruning import Pruning


class PruningProxy(Proxy):
    """
    A proxy for evaluating neural network pruning that balances:
    1. Model accuracy on a benchmark (e.g. CIFAR-10)
    2. Amount of pruning achieved (sparsity)
    3. Computational efficiency (FLOPs reduction)
    """

    def __init__(
        self,
        accuracy_weight: float = 1.0,
        sparsity_weight: float = 0.3,
        flops_weight: float = 0.2,
        min_accuracy: float = 0.6,
        **kwargs
    ):
        """
        Parameters
        ----------
        accuracy_weight : float
            Weight factor for the accuracy term in the energy computation.
            Higher values prioritize maintaining accuracy.
        
        sparsity_weight : float
            Weight for the sparsity (pruning amount) term.
            Higher values encourage more aggressive pruning.
            
        flops_weight : float
            Weight for the FLOPs reduction term.
            Higher values prioritize computational efficiency.
            
        min_accuracy : float
            Minimum acceptable accuracy threshold.
            Solutions below this threshold get heavily penalized.
        """
        super().__init__(**kwargs)
        
        self.accuracy_weight = accuracy_weight
        self.sparsity_weight = sparsity_weight
        self.flops_weight = flops_weight
        self.min_accuracy = min_accuracy
        
        self.model = None
        self.test_loader = None

    def setup(self, env: Optional[Pruning] = None):
        """Initialize with model and test data from the environment."""
        if env:
            self.model = env.model
            self.test_loader = env.test_loader

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        """
        Compute energy values for a batch of pruning states.
        
        The energy combines:
        1. Negative accuracy on test set (weighted)
        2. Negative sparsity ratio (weighted)
        3. Negative FLOPs reduction (weighted)
        
        Lower energy = better pruning (higher accuracy, more pruning, fewer FLOPs)
        
        Parameters
        ----------
        states : tensor
            Batch of pruning states to evaluate
            
        Returns
        -------
        tensor
            Energy values for each pruning state
        """
        energies = []

        for state in states:
            # Apply pruning mask to model
            self.model.apply_pruning_mask(state)
            
            # Calculate accuracy on test set
            accuracy = self._evaluate_accuracy()
            
            # Calculate sparsity (% of pruned weights)
            sparsity = self._calculate_sparsity(state)
            
            # Calculate FLOPs reduction
            flops_reduction = self._calculate_flops_reduction(state)
            
            # Accuracy penalty if below minimum threshold
            if accuracy < self.min_accuracy:
                accuracy_term = -self.accuracy_weight * (accuracy - self.min_accuracy)**2
            else:
                accuracy_term = -self.accuracy_weight * (1.0 - accuracy)
            
            # Combine terms into final energy
            # Note: We negate beneficial terms since we want to minimize energy
            energy = (accuracy_term 
                     - self.sparsity_weight * sparsity 
                     - self.flops_weight * flops_reduction)
            
            energies.append(energy)

        return torch.tensor(energies, device=self.device, dtype=self.float)

    def _evaluate_accuracy(self) -> float:
        """Evaluate model accuracy on test set."""
        correct = 0
        total = 0
        self.model.eval()
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        
        return correct / total

    def _calculate_sparsity(self, state: TensorType["state_dim"]) -> float:
        """Calculate fraction of weights that are pruned."""
        total_params = sum(mask.numel() for mask in state.values())
        pruned_params = sum((mask == 0).sum().item() for mask in state.values())
        return pruned_params / total_params

    def _calculate_flops_reduction(self, state: TensorType["state_dim"]) -> float:
        """
        Calculate reduction in FLOPs from pruning.
        This is an approximation based on the sparsity of each layer.
        """
        original_flops = 0
        pruned_flops = 0
        
        for name, mask in state.items():
            layer = self.model.get_layer(name)
            if hasattr(layer, 'weight'):
                layer_flops = np.prod(layer.weight.shape)
                original_flops += layer_flops
                pruned_flops += layer_flops * (mask == 0).sum().item() / mask.numel()
        
        return pruned_flops / original_flops if original_flops > 0 else 0.0
