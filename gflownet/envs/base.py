"""
Base class of GFlowNet environments
"""
from abc import abstractmethod
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch.distributions import Categorical
from torchtyping import TensorType

from gflownet.utils.common import set_device, set_float_precision


class GFlowNetEnv:
    """
    Base class of GFlowNet environments
    """

    def __init__(
        self,
        device="cpu",
        float_precision=32,
        env_id=None,
        reward_min=1e-8,
        reward_beta=1.0,
        reward_norm=1.0,
        reward_norm_std_mult=0.0,
        reward_func="identity",
        energies_stats=None,
        denorm_proxy=False,
        proxy=None,
        oracle=None,
        proxy_state_format="oracle",
        **kwargs,
    ):
        # Call reset() to set initial state, done, n_actions
        self.reset()
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)
        # Reward settings
        self.min_reward = reward_min
        assert self.min_reward > 0
        self.reward_beta = reward_beta
        assert self.reward_beta > 0
        self.reward_norm = reward_norm
        assert self.reward_norm > 0
        self.reward_norm_std_mult = reward_norm_std_mult
        self.reward_func = reward_func
        self.energies_stats = energies_stats
        self.denorm_proxy = denorm_proxy
        # Proxy and oracle
        self.proxy = proxy
        if oracle is None:
            self.oracle = self.proxy
        else:
            self.oracle = oracle
        if self.oracle is None or self.oracle.higher_is_better:
            self.proxy_factor = 1.0
        else:
            self.proxy_factor = -1.0
        self.proxy_state_format = proxy_state_format
        # Log SoftMax function
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        # Action space
        self.action_space = self.get_actions_space()
        # Policy outputs
        self.fixed_policy_output = self.get_fixed_policy_output()
        self.random_policy_output = self.get_fixed_policy_output()
        self.policy_output_dim = len(self.fixed_policy_output)
        self.policy_input_dim = len(self.state2policy())
        # To be probably removed:
        self._true_density = None
        self._z = None

    def copy(self):
        # return self.__class__(**self.__dict__)
        return deepcopy(self)

    def set_energies_stats(self, energies_stats):
        self.energies_stats = energies_stats

    def set_reward_norm(self, reward_norm):
        self.reward_norm = reward_norm

    @abstractmethod
    def get_actions_space(self):
        """
        Constructs list with all possible actions (excluding end of sequence)
        """
        pass

    def get_fixed_policy_output(self):
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy. As a baseline, the fixed policy is uniform over the
        dimensionality of the action space.
        """
        return np.ones(len(self.action_space))

    def get_max_traj_len(
        self,
    ):
        return 1e3

    def state2proxy(self, state: List = None):
        """
        Prepares a state in "GFlowNet format" for the proxy.

        Args
        ----
        state : list
            A state
        """
        if state is None:
            state = self.state.copy()
        return self.statebatch2proxy([state])

    def statebatch2proxy(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Prepares a batch of states in "GFlowNet format" for the proxy.

        Args
        ----
        state : list
            A state
        """
        return np.array(states)

    def statetorch2proxy(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Prepares a batch of states in torch "GFlowNet format" for the proxy.
        """
        return states

    def state2oracle(self, state: List = None):
        """
        Prepares a list of states in "GFlowNet format" for the oracle

        Args
        ----
        state : list
            A state
        """
        if state is None:
            state = self.state.copy()
        return state

    def statebatch2oracle(self, states: List[List]):
        """
        Prepares a batch of states in "GFlowNet format" for the oracles
        """
        return states

    def reward(self, state=None, done=None):
        """
        Computes the reward of a state
        """
        if done is None:
            done = self.done
        if done:
            return np.array(0.0)
        if state is None:
            state = self.state.copy()
        return self.proxy2reward(self.proxy(self.state2proxy(state)))

    def reward_batch(self, states: List[List], done=None):
        """
        Computes the rewards of a batch of states, given a list of states and 'dones'
        """
        if done is None:
            done = np.ones(len(states), dtype=bool)
        states_proxy = self.statebatch2proxy(states)[list(done), :]
        rewards = np.zeros(len(done))
        if states_proxy.shape[0] > 0:
            rewards[list(done)] = self.proxy2reward(self.proxy(states_proxy)).tolist()
        return rewards

    def reward_torchbatch(
        self, states: TensorType["batch", "state_dim"], done: TensorType["batch"] = None
    ):
        """
        Computes the rewards of a batch of states in "GFlownet format"
        """
        if done is None:
            done = torch.ones(states.shape[0], dtype=torch.bool, device=self.device)
        states_proxy = self.statetorch2proxy(states[done, :])
        reward = torch.zeros(done.shape[0], dtype=self.float, device=self.device)
        if states[done, :].shape[0] > 0:
            reward[done] = self.proxy2reward(self.proxy(states_proxy))
        return reward

    def proxy2reward(self, proxy_vals):
        """
        Prepares the output of an oracle for GFlowNet: the inputs proxy_vals is
        expected to be a negative value (energy), unless self.denorm_proxy is True. If
        the latter, the proxy values are first de-normalized according to the mean and
        standard deviation in self.energies_stats. The output of the function is a
        strictly positive reward - provided self.reward_norm and self.reward_beta are
        positive - and larger than self.min_reward.
        """
        if self.denorm_proxy:
            # TODO: do with torch
            proxy_vals = proxy_vals * self.energies_stats[3] + self.energies_stats[2]
        if self.reward_func == "power":
            return torch.clamp(
                (self.proxy_factor * proxy_vals / self.reward_norm) ** self.reward_beta,
                min=self.min_reward,
                max=None,
            )
        elif self.reward_func == "boltzmann":
            return torch.clamp(
                torch.exp(self.proxy_factor * self.reward_beta * proxy_vals),
                min=self.min_reward,
                max=None,
            )
        elif self.reward_func == "identity":
            return torch.clamp(
                self.proxy_factor * proxy_vals,
                min=self.min_reward,
                max=None,
            )
        else:
            raise NotImplemented

    def reward2proxy(self, reward):
        """
        Converts a "GFlowNet reward" into a (negative) energy or values as returned by
        an oracle.
        """
        if self.reward_func == "power":
            return self.proxy_factor * torch.exp(
                (torch.log(reward) + self.reward_beta * torch.log(self.reward_norm))
                / self.reward_beta
            )
        elif self.reward_func == "boltzmann":
            return self.proxy_factor * torch.log(reward) / self.reward_beta
        elif self.reward_func == "identity":
            return self.proxy_factor * reward
        else:
            raise NotImplemented

    def statetorch2policy(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in torch "GFlowNet format" for the policy
        """
        return states

    def state2policy(self, state=None):
        """
        Converts a state into a format suitable for a machine learning model, such as a
        one-hot encoding.
        """
        if state is None:
            state = self.state
        return state

    def statebatch2policy(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Converts a batch of states into a format suitable for a machine learning model,
        such as a one-hot encoding. Returns a numpy array.
        """
        return np.array(states)

    def policy2state(self, state_policy: List) -> List:
        """
        Converts the model (e.g. one-hot encoding) version of a state given as
        argument into a state.
        """
        return state_policy

    def state2readable(self, state=None):
        """
        Converts a state into human-readable representation.
        """
        if state is None:
            state = self.state
        return str(state)

    def readable2state(self, readable):
        """
        Converts a human-readable representation of a state into the standard format.
        """
        return readable

    def traj2readable(self, traj=None):
        """
        Converts a trajectory into a human-readable string.
        """
        return str(traj).replace("(", "[").replace(")", "]").replace(",", "")

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.state = self.source.copy()
        self.n_actions = 0
        self.done = False
        self.id = env_id
        return self

    def get_parents(self, state=None, done=None, action=None):
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : tuple
            Last action performed

        Returns
        -------
        parents : list
            List of parents in state format

        actions : list
            List of actions that lead to state for each parent in parents
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos]
        else:
            parents = []
            actions = []
        return parents, actions

    def sample_actions(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        sampling_method: str = "policy",
        mask_invalid_actions: TensorType["n_states", "policy_output_dim"] = None,
        temperature_logits: float = 1.0,
        loginf: float = 1000,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs. This implementation
        is generally valid for all discrete environments.
        """
        device = policy_outputs.device
        ns_range = torch.arange(policy_outputs.shape[0]).to(device)
        if sampling_method == "uniform":
            logits = torch.ones(policy_outputs.shape).to(device)
        elif sampling_method == "policy":
            logits = policy_outputs
            logits /= temperature_logits
        if mask_invalid_actions is not None:
            logits[mask_invalid_actions] = -loginf
        action_indices = Categorical(logits=logits).sample()
        logprobs = self.logsoftmax(logits)[ns_range, action_indices]
        # Build actions
        actions = [self.action_space[idx] for idx in action_indices]
        return actions, logprobs

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        is_forward: bool,
        actions: TensorType["n_states", 2],
        states_target: TensorType["n_states", "policy_input_dim"],
        mask_invalid_actions: TensorType["batch_size", "policy_output_dim"] = None,
        loginf: float = 1000,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions. This
        implementation is generally valid for all discrete environments.
        """
        device = policy_outputs.device
        ns_range = torch.arange(policy_outputs.shape[0]).to(device)
        logits = policy_outputs
        if mask_invalid_actions is not None:
            logits[mask_invalid_actions] = -loginf
        # TODO: fix need to convert to tuple: implement as in continuous
        action_indices = (
            torch.tensor(
                [self.action_space.index(tuple(action.tolist())) for action in actions]
            )
            .to(int)
            .to(device)
        )
        logprobs = self.logsoftmax(logits)[ns_range, action_indices]
        return logprobs

    def get_trajectories(
        self, traj_list, traj_actions_list, current_traj, current_actions
    ):
        """
        Determines all trajectories leading to each state in traj_list, recursively.

        Args
        ----
        traj_list : list
            List of trajectories (lists)

        traj_actions_list : list
            List of actions within each trajectory

        current_traj : list
            Current trajectory

        current_actions : list
            Actions of current trajectory

        Returns
        -------
        traj_list : list
            List of trajectories (lists)

        traj_actions_list : list
            List of actions within each trajectory
        """
        parents, parents_actions = self.get_parents(current_traj[-1], False)
        if parents == []:
            traj_list.append(current_traj)
            traj_actions_list.append(current_actions)
            return traj_list, traj_actions_list
        for idx, (p, a) in enumerate(zip(parents, parents_actions)):
            traj_list, traj_actions_list = self.get_trajectories(
                traj_list, traj_actions_list, current_traj + [p], current_actions + [a]
            )
        return traj_list, traj_actions_list

    def step(self, action_idx):
        """
        Executes step given an action.

        Args
        ----
        action_idx : int
            Index of action in the action space. a == eos indicates "stop action"

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action_idx : int
            Action index

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        if action < self.eos:
            self.done = False
            valid = True
        else:
            self.done = True
            valid = True
            self.n_actions += 1
        return self.state, action, valid

    def no_eos_mask(self, state=None):
        """
        Returns True if no eos action is allowed given state
        """
        if state is None:
            state = self.state
        return False

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        """
        Returns a vector of length the action space + 1: True if forward action is
        invalid given the current state, False otherwise.
        """
        mask = [False for _ in range(len(self.action_space))]
        return mask

    def get_mask_invalid_actions_backward(self, state=None, done=None, parents_a=None):
        """
        Returns a vector with the length of the discrete part of the action space + 1:
        True if action is invalid going backward given the current state, False
        otherwise.
        """
        if parents_a is None:
            _, parents_a = self.get_parents()
        mask = [True for _ in range(len(self.action_space))]
        for pa in parents_a:
            mask[self.action_space.index(pa)] = False
        return mask

    def set_state(self, state, done):
        """
        Sets the state and done of an environment.
        """
        self.state = state
        self.done = done
        return self

    def true_density(self):
        """
        Computes the reward density (reward / sum(rewards)) of the whole space

        Returns
        -------
        Tuple:
          - normalized reward for each state
          - un-normalized reward
          - states
        """
        return (None, None, None)

    @staticmethod
    def np2df(*args):
        """
        Args
        ----
        """
        return None

    def setup_proxy(self):
        if self.proxy:
            proxy.setup(env)
