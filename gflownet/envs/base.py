"""
Base class of GFlowNet environments
"""
from abc import abstractmethod
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from torch.distributions import Categorical
from torchtyping import TensorType

from gflownet.utils.common import set_device, set_float_precision

CMAP = mpl.colormaps["cividis"]


class GFlowNetEnv:
    """
    Base class of GFlowNet environments
    """

    def __init__(
        self,
        device: str = "cpu",
        float_precision: int = 32,
        env_id: Union[int, str] = "env",
        reward_min: float = 1e-8,
        reward_beta: float = 1.0,
        reward_norm: float = 1.0,
        reward_norm_std_mult: float = 0.0,
        reward_func: str = "identity",
        energies_stats: List[int] = None,
        denorm_proxy: bool = False,
        proxy=None,
        oracle=None,
        proxy_state_format: str = "oracle",
        fixed_distribution: Optional[dict] = None,
        random_distribution: Optional[dict] = None,
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
        self.setup_proxy()
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
        self.action_space = self.get_action_space()
        self.action_space_torch = torch.tensor(
            self.action_space, device=self.device, dtype=self.float
        )
        self.action_space_dim = len(self.action_space)
        # Max trajectory length
        self.max_traj_length = self.get_max_traj_length()
        # Policy outputs
        self.fixed_policy_output = self.get_policy_output(fixed_distribution)
        self.random_policy_output = self.get_policy_output(random_distribution)
        self.policy_output_dim = len(self.fixed_policy_output)
        self.policy_input_dim = len(self.state2policy())

    @abstractmethod
    def get_action_space(self):
        """
        Constructs list with all possible actions (excluding end of sequence)
        """
        pass

    def actions2indices(
        self, actions: TensorType["batch_size", "action_dim"]
    ) -> TensorType["batch_size"]:
        """
        Returns the corresponding indices in the action space of the actions in a batch.
        """
        # Expand the action_space tensor: [batch_size, d_actions_space, action_dim]
        action_space = torch.unsqueeze(self.action_space_torch, 0).expand(
            actions.shape[0], -1, -1
        )
        # Expand the actions tensor: [batch_size, d_actions_space, action_dim]
        actions = torch.unsqueeze(actions, 1).expand(-1, self.action_space_dim, -1)
        # Take the indices at the d_actions_space dimension where all the elements in
        # the action_dim dimension are True
        return torch.where(torch.all(actions == action_space, dim=2))[1]

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.
        For continuous or hybrid environments, this mask corresponds to the discrete
        part of the action space.
        """
        return [False for _ in range(self.action_space_dim)]

    def get_mask_invalid_actions_backward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        parents_a: Optional[List] = None,
    ) -> List:
        """
        Returns a list of length the action space with values:
            - True if the backward action is invalid from the current state.
            - False otherwise.
        For continuous or hybrid environments, this mask corresponds to the discrete
        part of the action space.

        The base implementation below should be common to all discrete spaces as it
        relies on get_parents, which is environment-specific and must be implemented.
        Continuous environments will probably need to implement its specific version of
        this method.
        """
        if parents_a is None:
            _, parents_a = self.get_parents()
        mask = [True for _ in range(self.action_space_dim)]
        for pa in parents_a:
            mask[self.action_space.index(pa)] = False
        return mask

    def get_parents(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        In continuous environments, get_parents() should return only the parent from
        which action leads to state.

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
            return [state], [(self.eos,)]
        parents = []
        actions = []
        return parents, actions

    def step(self, action: Tuple[int]) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action from the action space.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action index

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        # If env is done, return invalid
        if self.done:
            return self.state, action, False
        # If action not found in action space raise an error
        if action not in self.action_space:
            raise ValueError(
                f"Tried to execute action {action} not present in action space."
            )
        action_idx = self.action_space.index(action)
        # If action is in invalid mask, exit immediately
        if self.get_mask_invalid_actions_forward()[action_idx]:
            return self.state, action, False
        return None, None, None

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
        is generally valid for all discrete environments but continuous environments
        will likely have to implement its own.
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
        actions: TensorType["n_states", "actions_dim"],
        states_target: TensorType["n_states", "policy_input_dim"],
        mask_invalid_actions: TensorType["batch_size", "policy_output_dim"] = None,
        loginf: float = 1000,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions. This
        implementation is generally valid for all discrete environments but continuous
        environments will likely have to implement its own.
        """
        device = policy_outputs.device
        ns_range = torch.arange(policy_outputs.shape[0]).to(device)
        logits = policy_outputs
        if mask_invalid_actions is not None:
            logits[mask_invalid_actions] = -loginf
        action_indices = (
            torch.tensor(
                [self.action_space.index(tuple(action.tolist())) for action in actions]
            )
            .to(int)
            .to(device)
        )
        logprobs = self.logsoftmax(logits)[ns_range, action_indices]
        return logprobs

    def get_policy_output(self, params: Optional[dict] = None):
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy. As a baseline, the policy is uniform over the dimensionality of
        the action space.

        Continuous environments will generally have to overwrite this method.
        """
        return np.ones(self.action_space_dim)

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
        self, states: TensorType["batch_size", "state_dim"]
    ) -> TensorType["batch_size", "state_proxy_dim"]:
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

    def statetorch2policy(
        self, states: TensorType["batch_size", "state_dim"]
    ) -> TensorType["batch_size", "policy_input_dim"]:
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
        self,
        states: TensorType["batch_size", "state_dim"],
        done: TensorType["batch_size"] = None,
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
            raise NotImplementedError

    def reward2proxy(self, reward):
        """
        Converts a "GFlowNet reward" into a (negative) energy or values as returned by
        an oracle.
        """
        if self.reward_func == "power":
            return self.proxy_factor * torch.exp(
                (
                    torch.log(reward)
                    + self.reward_beta * torch.log(torch.as_tensor(self.reward_norm))
                )
                / self.reward_beta
            )
        elif self.reward_func == "boltzmann":
            return self.proxy_factor * torch.log(reward) / self.reward_beta
        elif self.reward_func == "identity":
            return self.proxy_factor * reward
        else:
            raise NotImplementedError

    def reset(self, env_id: Union[int, str] = None):
        """
        Resets the environment.
        """
        if torch.is_tensor(self.source):
            self.state = self.source.clone().detach()
        else:
            self.state = self.source.copy()
        self.n_actions = 0
        self.done = False
        self.id = env_id
        return self

    def set_state(self, state: List, done: Optional[bool] = False):
        """
        Sets the state and done of an environment.
        """
        self.state = state
        self.done = done
        return self

    def copy(self):
        # return self.__class__(**self.__dict__)
        return deepcopy(self)

    def set_energies_stats(self, energies_stats):
        self.energies_stats = energies_stats

    def set_reward_norm(self, reward_norm):
        self.reward_norm = reward_norm

    def get_max_traj_length(self):
        return 1e3

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

    def setup_proxy(self):
        if self.proxy:
            self.proxy.setup(self)

    @torch.no_grad()
    def compute_train_energy_proxy_and_rewards(self):
        """
        Gather batched proxy data:

        * The ground-truth energy of the train set
        * The predicted proxy energy over the train set
        * The reward version of those energies (with env.proxy2reward)

        Returns
        -------
        gt_energy : torch.Tensor
            The ground-truth energies in the proxy's train set

        proxy_energy : torch.Tensor
            The proxy's predicted energies over its train set

        gt_reward : torch.Tensor
            The reward version of the ground-truth energies

        proxy_reward : torch.Tensor
            The reward version of the proxy's predicted energies
        """
        gt_energy, proxy_energy = self.proxy.infer_on_train_set()
        gt_reward = self.proxy2reward(gt_energy)
        proxy_reward = self.proxy2reward(proxy_energy)

        return gt_energy, proxy_energy, gt_reward, proxy_reward

    @torch.no_grad()
    def top_k_metrics_and_plots(
        self,
        states,
        top_k,
        name,
        energy=None,
        reward=None,
        step=None,
        **kwargs,
    ):
        """
        Compute top_k metrics and plots for the given states.

        In particular, if no states, energy, or reward are passed, then the name
        *must* be "train", and the energy and reward will be computed from the
        proxy using `env.compute_train_energy_proxy_and_rewards()`. In this case,
        `top_k_metrics_and_plots` will be called a second time to compute the
        metrics and plots of the proxy distribution in addition to the ground-truth
        distribution.
        Train mode should only be called once at the begining of training as
        distributions do not change over time.

        If `states` are passed, then the energy and reward will be computed from the
        proxy for those states. They are typically sampled from the current GFN.

        Otherwise, energy and reward should be passed directly.

        *Plots and metrics*:
        - mean+std of energy and reward
        - mean+std of top_k energy and reward
        - histogram of energy and reward
        - histogram of top_k energy and reward


        Args
        ----
        states: list
            List of states to compute metrics and plots for.

        top_k: int
            Number of top k states to compute metrics and plots for.
            "top" means lowest energy/highest reward.

        name: str
            Name of the distribution to compute metrics and plots for.
            Typically "gflownet", "random" or "train". Will be used in
            metrics names like `f"Mean {name} energy"`.

        energy: torch.Tensor, optional
            Batch of pre-computed energies

        reward: torch.Tensor, optional
            Batch of pre-computed rewards

        step: int, optional
            Step number to use for the plot title.

        Returns
        -------
        metrics: dict
            Dictionary of metrics: str->float

        figs: list
            List of matplotlib figures

        figs_names: list
            List of figure names for `figs`
        """

        if states is None and energy is None and reward is None:
            assert name == "train"
            (
                energy,
                proxy,
                energy_reward,
                proxy_reward,
            ) = self.compute_train_energy_proxy_and_rewards()
            name = "train ground truth"
            reward = energy_reward
        elif energy is None and reward is None:
            x = torch.stack([self.state2proxy(s) for s in states])
            energy = self.proxy(x.to(self.device)).cpu()
            reward = self.proxy2reward(energy)

        assert energy is not None and reward is not None

        # select top k best energies and rewards
        top_k_e = torch.topk(energy, top_k, largest=False, dim=0).values.numpy()
        top_k_r = torch.topk(reward, top_k, largest=True, dim=0).values.numpy()

        # find best energy and reward
        best_e = torch.min(energy).item()
        best_r = torch.max(reward).item()

        # to numpy to plot
        energy = energy.numpy()
        reward = reward.numpy()

        # compute stats
        mean_e = np.mean(energy)
        mean_r = np.mean(reward)

        std_e = np.std(energy)
        std_r = np.std(reward)

        mean_top_k_e = np.mean(top_k_e)
        mean_top_k_r = np.mean(top_k_r)

        std_top_k_e = np.std(top_k_e)
        std_top_k_r = np.std(top_k_r)

        # automatic color scale
        # currently: cividis colour map
        colors = ["full", "top_k"]
        normalizer = mpl.colors.Normalize(vmin=0, vmax=len(colors) - 0.5)
        colors = {k: CMAP(normalizer(i)) for i, k in enumerate(colors[::-1])}

        # two sublopts: left is energy, right is reward
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # energy full distribution and stats lines
        ax[0].hist(
            energy,
            bins=100,
            alpha=0.35,
            label=f"All = {len(energy)}",
            color=colors["full"],
            density=True,
        )
        ax[0].axvline(
            mean_e,
            color=colors["full"],
            linestyle=(0, (5, 10)),
            label=f"Mean = {mean_e:.3f}",
        )
        ax[0].axvline(
            mean_e + std_e,
            color=colors["full"],
            linestyle=(0, (1, 10)),
            label=f"Std = {std_e:.3f}",
        )
        ax[0].axvline(
            mean_e - std_e,
            color=colors["full"],
            linestyle=(0, (1, 10)),
        )

        # energy top k distribution and stats lines
        ax[0].hist(
            top_k_e,
            bins=100,
            alpha=0.7,
            label=f"Top k = {top_k}",
            color=colors["top_k"],
            density=True,
        )
        ax[0].axvline(
            mean_top_k_e,
            color=colors["top_k"],
            linestyle=(0, (5, 10)),
            label=f"Mean = {mean_top_k_e:.3f}",
        )
        ax[0].axvline(
            mean_top_k_e + std_top_k_e,
            color=colors["top_k"],
            linestyle=(0, (1, 10)),
            label=f"Std = {std_top_k_e:.3f}",
        )
        ax[0].axvline(
            mean_top_k_e - std_top_k_e,
            color=colors["top_k"],
            linestyle=(0, (1, 10)),
        )
        # energy title & legend
        ax[0].set_title(
            f"Energy distribution for {top_k} vs {len(energy)}"
            + f" samples\nBest: {best_e:.3f}",
            y=0,
            pad=-20,
            verticalalignment="top",
            size=12,
        )
        ax[0].legend()

        # reward full distribution and stats lines
        ax[1].hist(
            reward,
            bins=100,
            alpha=0.35,
            label=f"All = {len(reward)}",
            color=colors["full"],
            density=True,
        )
        ax[1].axvline(
            mean_r,
            color=colors["full"],
            linestyle=(0, (5, 10)),
            label=f"Mean = {mean_r:.3f}",
        )
        ax[1].axvline(
            mean_r + std_r,
            color=colors["full"],
            linestyle=(0, (1, 10)),
            label=f"Std = {std_r:.3f}",
        )
        ax[1].axvline(
            mean_r - std_r,
            color=colors["full"],
            linestyle=(0, (1, 10)),
        )

        # reward top k distribution and stats lines
        ax[1].hist(
            top_k_r,
            bins=100,
            alpha=0.7,
            label=f"Top k = {top_k}",
            color=colors["top_k"],
            density=True,
        )
        ax[1].axvline(
            mean_top_k_r,
            color=colors["top_k"],
            linestyle=(0, (5, 10)),
            label=f"Mean = {mean_top_k_r:.3f}",
        )
        ax[1].axvline(
            mean_top_k_r + std_top_k_r,
            color=colors["top_k"],
            linestyle=(0, (1, 10)),
            label=f"Std = {std_top_k_r:.3f}",
        )
        ax[1].axvline(
            mean_top_k_r - std_top_k_r,
            color=colors["top_k"],
            linestyle=(0, (1, 10)),
        )

        # reward title & legend
        ax[1].set_title(
            f"Reward distribution for {top_k} vs {len(reward)}"
            + f" samples\nBest: {best_r:.3f}",
            y=0,
            pad=-20,
            verticalalignment="top",
            size=12,
        )
        ax[1].legend()

        # Finalize figure
        title = f"{name.capitalize()} energy and reward distributions"
        if step is not None:
            title += f" (step {step})"
        fig.suptitle(title, y=0.95)
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        # store metrics
        metrics = {
            f"Mean {name} energy": mean_e,
            f"Std {name} energy": std_e,
            f"Mean {name} reward": mean_r,
            f"Std {name} reward": std_r,
            f"Mean {name} top k energy": mean_top_k_e,
            f"Std {name} top k energy": std_top_k_e,
            f"Mean {name} top k reward": mean_top_k_r,
            f"Std {name} top k reward": std_top_k_r,
            f"Best (min) {name} energy": best_e,
            f"Best (max) {name} reward": best_r,
        }
        figs = [fig]
        fig_names = [title]

        if name.lower() == "train ground truth":
            # train stats mode: the ground truth data has meen plotted
            # and computed, let's do it again for the proxy data.
            # This can be used to visualize potential distribution mismatch
            # between the proxy and the ground truth data.
            proxy_metrics, proxy_figs, proxy_fig_names = self.top_k_metrics_and_plots(
                None,
                top_k,
                "train proxy",
                energy=proxy,
                reward=proxy_reward,
                step=None,
                **kwargs,
            )
            # aggregate metrics and figures
            metrics.update(proxy_metrics)
            figs += proxy_figs
            fig_names += proxy_fig_names

        return metrics, figs, fig_names
