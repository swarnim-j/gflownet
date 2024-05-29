"""
Branin objective function, relying on the botorch implementation.

This code is based on the implementation by Nikita Saxena (nikita-0209) in 
https://github.com/alexhernandezgarcia/activelearning

The implementation assumes by default that the inputs will be on [-1, 1] x [-1, 1] and
will be mapped to the standard domain of the Branin function (see X1_DOMAIN and
X2_DOMAIN). Setting do_domain_map to False will prevent the mapping.

Branin function is typically used as a minimization problem, with the minima around
zero but positive. This is the convention followed by default, therefore the user
should carefully select the reward function.
"""

import torch
from botorch.test_functions.multi_fidelity import AugmentedBranin
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat

X1_DOMAIN = [-5, 10]
X1_LENGTH = X1_DOMAIN[1] - X1_DOMAIN[0]
X2_DOMAIN = [0, 15]
X2_LENGTH = X2_DOMAIN[1] - X2_DOMAIN[0]
UPPER_BOUND_IN_DOMAIN = 309


class Branin(Proxy):
    def __init__(
        self,
        fidelity=1.0,
        do_domain_map=True,
        shift_to_negative=False,
        reward_function="product",
        rewareward_function_kwargs={"beta": -1},
        **kwargs
    ):
        """
        Parameters
        ----------
        fidelity : float
            Fidelity of the Branin oracle. 1.0 corresponds to the original Branin.
            Smaller values (up to 0.0) reduce the fidelity of the oracle.
        do_domain_map : bool
            If True, the states are assumed to be in [0, 1] x [0, 1] and are re-mapped
            to the standard domain before calling the botorch method. If False, the
            botorch method is called directly on the states values.

        See: https://botorch.org/api/test_functions.html
        """
        super().__init__(**kwargs)
        self.fidelity = fidelity
        self.do_domain_map = do_domain_map
        self.shift_to_negative = shift_to_negative
        self.function_mf_botorch = AugmentedBranin(negate=False)
        # Constants
        self.domain_left = tfloat(
            [[X1_DOMAIN[0], X2_DOMAIN[0]]], float_type=self.float, device=self.device
        )
        self.domain_length = tfloat(
            [[X1_LENGTH, X2_LENGTH]], float_type=self.float, device=self.device
        )
        # Modes and extremum compatible with 100x100 grid
        self.modes = [
            [12.4, 81.833],
            [54.266, 15.16],
            [94.98, 16.5],
        ]
        self.extremum = 0.397887

    def __call__(self, states: TensorType["batch", "2"]) -> TensorType["batch"]:
        if states.shape[1] != 2:
            raise ValueError(
                """
            Inputs to the Branin function must be 2-dimensional, but inputs with
            {states.shape[1]} dimensions were passed.
            """
            )
        if self.do_domain_map:
            states = self.map_to_standard_domain(states)
        # Append fidelity as a new dimension of states
        states = torch.cat(
            [
                states,
                self.fidelity
                * torch.ones(
                    states.shape[0], device=self.device, dtype=self.float
                ).unsqueeze(-1),
            ],
            dim=1,
        )
        if self.shift_to_negative:
            return Branin.map_to_negative_range(self.function_mf_botorch(states))
        else:
            return self.function_mf_botorch(states)

    @property
    def min(self):
        if not hasattr(self, "_min"):
            self._min = torch.tensor(
                -UPPER_BOUND_IN_DOMAIN, device=self.device, dtype=self.float
            )
        return self._min

    def map_to_standard_domain(
        self,
        states: TensorType["batch", "2"],
    ) -> TensorType["batch", "2"]:
        """
        Maps a batch of input states onto the domain typically used to evaluate the
        Branin function. See X1_DOMAIN and X2_DOMAIN. It assumes that the inputs are on
        [-1, 1] x [-1, 1].
        """
        return self.domain_left + ((states + 1.0) * self.domain_length) / 2.0

    @staticmethod
    def map_to_negative_range(values: TensorType["batch"]) -> TensorType["batch"]:
        """
        Maps a batch of function values onto a negative range by substracting an upper
        bound of the Branin function in the standard domain (UPPER_BOUND_IN_DOMAIN).
        """
        return values - UPPER_BOUND_IN_DOMAIN

    @staticmethod
    def map_to_standard_range(values: TensorType["batch"]) -> TensorType["batch"]:
        """
        Maps a batch of function values in a negative range back onto the standard
        range by adding an upper bound of the Branin function in the standard domain
        (UPPER_BOUND_IN_DOMAIN).
        """
        return values + UPPER_BOUND_IN_DOMAIN
