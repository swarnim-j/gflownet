from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Any

import torch.nn as nn
import torch
from torch import TensorType

from gflownet.envs.base import GFlowNetEnv


LAYER_TYPE_TO_NN = {
    "linear": nn.Linear,
    "conv2d": nn.Conv2d,
    "batch_norm": nn.BatchNorm2d,
    "relu": nn.ReLU,
    "dropout": nn.Dropout,
}


@dataclass
class ModelArchitecture:
    layer_params: List[Dict[str, Any]]


class Model:
    def __init__(self, architecture: ModelArchitecture, pretrained_weights: str = None):
        self.architecture = architecture
        self.model = self._build_model()
        
        # Only count neurons in Linear layers for pruning
        self.total_neurons = sum(
            layer.out_features 
            for layer in self.model.modules() 
            if isinstance(layer, nn.Linear)
        )
        self.active_neurons = self.total_neurons
        
        if pretrained_weights is not None:
            self.model.load_state_dict(torch.load(pretrained_weights))

        # Initialize masks only for Linear layers
        self.state = {}
        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear):
                self.state[layer_name] = torch.ones(layer.out_features)

    def _build_model(self) -> nn.Module:
        layers = []
        
        for params in self.architecture.layer_params:
            layer_type = params['type']
            layer_class = LAYER_TYPE_TO_NN[layer_type]

            # Remove 'type' from params before passing to layer constructor
            layer_params = {k: v for k, v in params.items() if k != 'type'}
            # Add layer directly to Sequential without ModuleDict
            layers.append(layer_class(**layer_params))
        
        return nn.Sequential(*layers)

    def update_state(self, layer_name: str, neuron_idx: int):
        mask = self.state[layer_name]
        mask[neuron_idx] = 0
        self.state[layer_name] = mask
        delta_neurons = (
            mask - self.state[layer_name]
        ).sum()
        self.active_neurons += delta_neurons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep track of the current output
        current_output = x
        
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Sequential):
                continue
            if name in self.state:  # Only Linear layers will have masks
                current_output = layer(current_output) * self.state[name].unsqueeze(0)
            else:
                current_output = layer(current_output)
        
        return current_output


class Pruning(GFlowNetEnv):
    """
    GFlowNet environment for pruning neural networks.
    """

    def __init__(
        self,
        architecture: ModelArchitecture = None,
        pretrained_weights: str = None,
        test_dataset: str = "cifar10",
        batch_size: int = 128,
        num_workers: int = 4,
        **kwargs,
    ):
        """
        Parameters
        ----------
        architecture : ModelArchitecture
            Architecture specification for the model to be pruned
        pretrained_weights : str
            Path to pretrained model weights
        test_dataset : str
            Name of dataset to use for evaluation ('cifar10', 'cifar100', etc.)
        batch_size : int
            Batch size for test data loader
        num_workers : int
            Number of workers for test data loader
        """
        if architecture is None:
            # implement default architecture
            pass

        self.model = Model(architecture, pretrained_weights)
        self.test_loader = self._setup_test_loader(
            test_dataset, batch_size, num_workers
        )
        self.source = torch.ones(
            (self.model.total_neurons), dtype=torch.float, device='cpu'
        )
        super().__init__(**kwargs)

    def _setup_test_loader(self, dataset_name: str, batch_size: int, num_workers: int):
        """
        Set up test data loader for model evaluation.

        Parameters
        ----------
        dataset_name : str
            Name of dataset to use
        batch_size : int
            Batch size for data loader
        num_workers : int
            Number of worker processes

        Returns
        -------
        torch.utils.data.DataLoader
            Test data loader
        """
        import torchvision
        import torchvision.transforms as transforms

        # Standard normalization for CIFAR
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )

        # Basic test transform
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

        # Setup dataset
        if dataset_name.lower() == "cifar10":
            test_dataset = torchvision.datasets.CIFAR10(
                root="./data", train=False, download=True, transform=test_transform
            )
        elif dataset_name.lower() == "cifar100":
            test_dataset = torchvision.datasets.CIFAR100(
                root="./data", train=False, download=True, transform=test_transform
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Create data loader
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return test_loader

    def get_mask_invalid_actions_forward(
        self,
        state: Dict[str, torch.Tensor] = None,
    ) -> List[bool]:
        return [False for _ in range(self.action_space_dim)]

    def get_mask_invalid_actions_backward(
        self,
        state: Dict[str, torch.Tensor] = None,
    ) -> List[bool]:
        return [False for _ in range(self.action_space_dim)]

    def get_parents(
        self,
        state: List[Dict[str, torch.Tensor]] | None = None,
        done: List[bool] | None = None,
        action: List[Tuple[int, int]] | None = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Tuple[int, int]]]:
        """
        Get parent states and the actions that lead to the current state.
        For pruning, parents are states where one of the currently pruned neurons is active.

        Returns:
            Tuple containing:
            - List of parent states
            - List of actions that transform each parent to the current state
        """
        if state is None or len(state) == 0:
            return [], []

        parents = []
        actions = []

        # For each state in the batch
        for s in state:
            state_parents = []
            state_actions = []

            # For each layer in the state
            for layer_idx, (layer_name, mask) in enumerate(s.items()):
                # Find pruned neurons (where mask is 0)
                pruned_neurons = torch.where(mask == 0)[0]

                # For each pruned neuron, create a parent state where that neuron is active
                for neuron_idx in pruned_neurons:
                    # Create a copy of the current state
                    parent_state = {k: v.clone() for k, v in s.items()}
                    # Set the neuron to active in the parent state
                    parent_state[layer_name][neuron_idx] = 1

                    state_parents.append(parent_state)
                    state_actions.append((layer_idx, neuron_idx.item()))

            parents.append(state_parents)
            actions.append(state_actions)

        return parents, actions

    def get_action_space(self) -> List[Tuple[int, int]]:
        """
        Actions are a tuple containing:
            1) layer index of Linear layer,
            2) neuron index.
        Skip first (input) and last (output) Linear layers.
        """
        linear_layers = [
            (idx, layer) 
            for idx, (name, layer) in enumerate(self.model.model.named_modules())
            if isinstance(layer, nn.Linear)
        ]
        
        # Skip first and last Linear layers
        prunable_layers = linear_layers[1:-1]
        
        return [
            (layer_idx, neuron_idx)
            for layer_idx, layer in prunable_layers
            for neuron_idx in range(
                layer.out_features
            )
        ]

    def step(
        self, action: Tuple[int, int]
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[int, int], bool]:
        layer_idx, neuron_idx = action
        # Prevent pruning of input and output layers
        if (
            layer_idx <= 0
            or layer_idx >= len(self.model.architecture.layer_sizes) - 1
            or neuron_idx < 0
            or neuron_idx >= self.model.architecture.layer_sizes[layer_idx]
        ):
            return self.model.state, action, False

        self.model.update_state(layer_idx, neuron_idx)
        return self.model.state, action, True

    def states2proxy(
        self, states: Union[List[Dict], torch.Tensor]
    ) -> torch.Tensor:
        """
        Converts a batch of states (dictionaries of masks) into a single tensor for the proxy.
        """
        if isinstance(states[0], dict):
            # Convert list of dicts into a single tensor
            return torch.stack(
                [
                    torch.cat(
                        [
                            state[layer_name]
                            for layer_name in self.model.architecture.layer_names
                        ]
                    )
                    for state in states
                ]
            ).to(self.device, self.float)
        return states.to(self.device, self.float)

    def state2readable(self, state: Dict[str, torch.Tensor] = None) -> str:
        """
        Converts a state (dictionary of masks) into a human-readable string.
        Format: "layer1: active_neurons/total_neurons, layer2: active_neurons/total_neurons, ..."
        """
        if state is None:
            state = self.state

        readable = []
        for layer_name, mask in state.items():
            active = torch.sum(mask).item()
            total = len(mask)
            readable.append(f"{layer_name}: {active}/{total}")

        return ", ".join(readable)
