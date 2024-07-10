from typing import List

from omegaconf import OmegaConf
from torch import nn

from gflownet.policy.base import Policy


class MLPPolicy(Policy):
    def __init__(self, n_layers: int = 2, n_hid: int = 128, tail: List = [], **kwargs):
        """
        MLP Policy class for a :class:`GFlowNetAgent`.

        Parameters
        ----------
        n_layers : int
            The number of layers in the MLP architecture.
        n_hid : int
            The number of hidden units per layer.
        tail : list
            A list of layers to conform the top (tail) of the MLP architecture.
        """
        # MLP features: number of layers, number of hidden units, tail, etc.
        self.n_layers = n_layers
        self.n_hid = n_hid
        self.tail = tail
        # Base init
        super().__init__(**kwargs)

    def make_model(self, activation: nn.Module = nn.LeakyReLU()):
        """
        Instantiates an MLP with no top layer activation as the policy model.

        If self.shared_weights is True, the base model with which weights are to be
        shared must be provided.

        Parameters
        ----------
        activation : nn.Module
            Activation function of the MLP layers

        Returns
        -------
        model : torch.tensor or torch.nn.Module
            A torch model containing the MLP.
        is_model : bool
            True because an MLP is a model.
        """
        activation.to(self.device)

        if self.shared_weights == True and self.base is not None:
            mlp = nn.Sequential(
                self.base.model[:-1],
                nn.Linear(
                    self.base.model[-1].in_features, self.base.model[-1].out_features
                ),
            )
            return mlp, True
        elif self.shared_weights == False:
            layers_dim = (
                [self.state_dim] + [self.n_hid] * self.n_layers + [(self.output_dim)]
            )
            mlp = nn.Sequential(
                *(
                    sum(
                        [
                            [nn.Linear(idim, odim)]
                            + ([activation] if n < len(layers_dim) - 2 else [])
                            for n, (idim, odim) in enumerate(
                                zip(layers_dim, layers_dim[1:])
                            )
                        ],
                        [],
                    )
                    + self.tail
                )
            )
            return mlp, True
        else:
            raise ValueError(
                "Base Model must be provided when shared_weights is set to True"
            )
