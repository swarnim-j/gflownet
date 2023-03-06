"""
Classes to represent a hyper-grid environments
"""
from typing import List, Tuple
import itertools
import numpy as np
import numpy.typing as npt
import torch
from torchtyping import TensorType
from gflownet.envs.base import GFlowNetEnv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Grid(GFlowNetEnv):
    """
    Hyper-grid environment

    Attributes
    ----------
    ndim : int
        Dimensionality of the grid

    length : int
        Size of the grid (cells per dimension)

    cell_min : float
        Lower bound of the cells range

    cell_max : float
        Upper bound of the cells range
    """

    def __init__(
        self,
        n_dim=2,
        length=3,
        min_step_len=1,
        max_step_len=1,
        cell_min=-1,
        cell_max=1,
        rescale=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_dim = n_dim
        self.eos = self.n_dim
        self.source = [0 for _ in range(self.n_dim)]
        self.length = length
        self.min_step_len = min_step_len
        self.max_step_len = max_step_len
        self.cells = np.linspace(cell_min, cell_max, length)
        self.reset()
        self.action_space = self.get_actions_space()
        self.fixed_policy_output = self.get_fixed_policy_output()
        self.random_policy_output = self.get_fixed_policy_output()
        self.policy_output_dim = len(self.fixed_policy_output)
        self.policy_input_dim = len(self.state2policy())
        if self.proxy_state_format == "ohe":
            self.statebatch2proxy = self.statebatch2policy
        elif self.proxy_state_format == "oracle":
            self.statebatch2proxy = self.statebatch2oracle
            self.statetorch2proxy = self.statetorch2oracle
        elif self.proxy_state_format == "state":
            self.statebatch2proxy = self.statebatch2state
            self.statetorch2proxy = self.statetorch2state
            # Assumes that the oracle is always Branin
            self.statebatch2oracle = self.statebatch2state
            self.statetorch2oracle = self.statetorch2state
        else:
            raise NotImplementedError(
                f"Proxy state format {self.proxy_state_format} not implemented"
            )
        if self.oracle is not None and hasattr(self.oracle, "n_dim"):
            self.oracle.n_dim = self.n_dim
            self.oracle.setup()
        self.rescale = rescale

    def statebatch2state(self, state_batch):
        """
        Converts a batch of states to AugmentedBranin oracle format
        """
        if isinstance(state_batch, torch.Tensor) == False:
            state_batch = torch.tensor(state_batch)
        return self.statetorch2state(state_batch)

    def statetorch2state(self, state_torch):
        """
        Converts a batch of states to AugmentedBranin oracle format
        """
        state_torch = state_torch / self.rescale
        return state_torch.to(self.float).to(self.device)

    def get_actions_space(self):
        """
        Constructs list with all possible actions, including eos.
        """
        valid_steplens = np.arange(self.min_step_len, self.max_step_len + 1)
        dims = [a for a in range(self.n_dim)]
        actions = []
        for r in valid_steplens:
            actions_r = [el for el in itertools.product(dims, repeat=r)]
            actions += actions_r
        actions += [(self.eos,)]
        return actions

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        """
        Returns a vector of length the action space + 1: True if forward action is
        invalid given the current state, False otherwise.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(self.policy_output_dim)]
        mask = [False for _ in range(self.policy_output_dim)]
        for idx, a in enumerate(self.action_space[:-1]):
            for d in a:
                if state[d] + 1 >= self.length:
                    mask[idx] = True
                    break
        return mask

    def true_density(self):
        # Return pre-computed true density if already stored
        if self._true_density is not None:
            return self._true_density
        # Calculate true density
        all_states = np.int32(
            list(itertools.product(*[list(range(self.length))] * self.n_dim))
        )
        state_mask = np.array(
            [len(self.get_parents(s, False)[0]) > 0 or sum(s) == 0 for s in all_states]
        )
        all_oracle = self.state2oracle(all_states)
        rewards = self.oracle(all_oracle)[state_mask]
        self._true_density = (
            rewards / rewards.sum(),
            rewards,
            list(map(tuple, all_states[state_mask])),
        )
        return self._true_density

    def state2oracle(self, state: List = None):
        """
        Prepares a state in "GFlowNet format" for the oracles: a list of length
        n_dim with values in the range [cell_min, cell_max] for each state.

        Args
        ----
        state : list
            State
        """
        if state is None:
            state = self.state.copy()
        return (
            self.state2policy(state).reshape((self.n_dim, self.length))
            * self.cells[None, :]
        ).sum(axis=1)

    def statebatch2oracle(
        self, states: List[List]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in "GFlowNet format" for the oracles: a list of
        length n_dim with values in the range [cell_min, cell_max] for each state.

        Args
        ----
        state : list
            State
        """
        return self.statetorch2oracle(
            torch.tensor(states, device=self.device, dtype=self.float)
        )

    def statetorch2oracle(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in torch "GFlowNet format" for the oracle.
        """
        return (
            self.statetorch2policy(states).reshape(
                (len(states), self.n_dim, self.length)
            )
            * torch.tensor(self.cells[None, :]).to(states.device).to(self.float)
        ).sum(axis=2)

    def state2policy(self, state: List = None) -> List:
        """
        Transforms the state given as argument (or self.state if None) into a
        one-hot encoding. The output is a list of len length * n_dim,
        where each n-th successive block of length elements is a one-hot encoding of
        the position in the n-th dimension.

        Example:
          - State, state: [0, 3, 1] (n_dim = 3)
          - state2policy(state): [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0] (length = 4)
                              |     0    |      3    |      1    |
        """
        if state is None:
            state = self.state.copy()
        state_policy = np.zeros(self.length * self.n_dim, dtype=np.float32)
        state_policy[(np.arange(len(state)) * self.length + state)] = 1
        return state_policy.tolist()

    def statebatch2policy(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Transforms a batch of states into a one-hot encoding. The output is a numpy
        array of shape [n_states, length * n_dim].

        See state2policy().
        """
        cols = np.array(states) + np.arange(self.n_dim) * self.length
        rows = np.repeat(np.arange(len(states)), self.n_dim)
        state_policy = np.zeros(
            (len(states), self.length * self.n_dim), dtype=np.float32
        )
        state_policy[rows, cols.flatten()] = 1.0
        return state_policy

    def statetorch2policy(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "policy_output_dim"]:
        """
        Transforms a batch of states into a one-hot encoding. The output is a numpy
        array of shape [n_states, length * n_dim].

        See state2policy().
        """
        device = states.device
        cols = (states + torch.arange(self.n_dim).to(device) * self.length).to(int)
        rows = torch.repeat_interleave(
            torch.arange(states.shape[0]).to(device), self.n_dim
        )
        state_policy = torch.zeros(
            (states.shape[0], self.length * self.n_dim), dtype=states.dtype
        ).to(device)
        state_policy[rows, cols.flatten()] = 1.0
        return state_policy

    def policy2state(self, state_policy: List) -> List:
        """
        Transforms the one-hot encoding version of a state given as argument
        into a state (list of the position at each dimension).

        Example:
          - state_policy: [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0] (length = 4, n_dim = 3)
                          |     0    |      3    |      1    |
          - policy2state(state_policy): [0, 3, 1]
        """
        return np.where(np.reshape(state_policy, (self.n_dim, self.length)))[1].tolist()

    def readable2state(self, readable, alphabet={}):
        """
        Converts a human-readable string representing a state into a state as a list of
        positions.
        """
        return [int(el) for el in readable.strip("[]").split(" ") if el != ""]

    def state2readable(self, state, alphabet={}):
        """
        Converts a state (a list of positions) into a human-readable string
        representing a state.
        """
        return str(state).replace("(", "[").replace(")", "]").replace(",", "")

    def statetorch2readable(self, state, alphabet={}):
        """
        Dataset Handler in activelearning deals only in tensors. This function converts the tesnor to readble format to save the train dataset
        """
        assert torch.eq(state.to(torch.long), state).all()
        state = state.to(torch.long)
        state = state.detach().cpu().numpy()
        return str(state).replace("(", "[").replace(")", "]").replace(",", "")

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
            Representation of a state, as a list of length length where each element is
            the position at each dimension.

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : None
            Ignored

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
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space[:-1]):
                state_aux = state.copy()
                for a_sub in a:
                    if state_aux[a_sub] > 0:
                        state_aux[a_sub] -= 1
                    else:
                        break
                else:
                    parents.append(state_aux)
                    actions.append(a)
        return parents, actions

    def step(self, action: Tuple[int]) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple int values indicating the
            dimensions to increment by 1.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        if self.done:
            return self.state, action, False
        # If only possible action is eos, then force eos
        # All dimensions are at the maximum length
        if all([s == self.length - 1 for s in self.state]):
            self.done = True
            self.n_actions += 1
            return self.state, (self.eos,), True
        # If action is not eos, then perform action
        elif action[0] != self.eos:
            state_next = self.state.copy()
            for a in action:
                state_next[a] += 1
            if any([s >= self.length for s in state_next]):
                valid = False
            else:
                self.state = state_next
                valid = True
                self.n_actions += 1
            return self.state, action, valid
        # If action is eos, then perform eos
        else:
            self.done = True
            self.n_actions += 1
            return self.state, (self.eos,), True

    def get_all_terminating_states(self) -> List[List]:
        all_x = np.int32(
            list(itertools.product(*[list(range(self.length))] * self.n_dim))
        )
        return all_x.tolist()

    def get_uniform_terminating_states(self, n_states: int) -> List[List]:
        states = np.random.randint(low=0, high=self.length, size=(n_states, self.n_dim))
        return states.tolist()

    def plot_samples_frequency(self, samples, ax=None, title=None, rescale=1):
        """
        Plot 2D histogram of samples.
        """
        if self.n_dim > 2:
            return None
        if ax is None:
            fig, ax = plt.subplots()
            standalone = True
        else:
            standalone = False
        # assuming the first time this function would be called when the dataset is created
        if self.rescale == None:
            self.rescale = rescale
        # make a list of integers from 0 to n_dim
        if self.rescale != 1:
            step = int(self.length / self.rescale)
        else:
            step = 1
        ax.set_xticks(np.arange(start=0, stop=self.length, step=step))
        ax.set_yticks(np.arange(start=0, stop=self.length, step=step))
        # check if samples is on GPU
        if torch.is_tensor(samples) and samples.is_cuda:
            samples = samples.detach().cpu()
        states = np.array(samples).astype(int)
        grid = np.zeros((self.length, self.length))
        if title == None:
            ax.set_title("Frequency of Coordinates Sampled")
        else:
            ax.set_title(title)
        # TODO: optimize
        for state in states:
            grid[state[0], state[1]] += 1
        im = ax.imshow(grid)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
        if standalone == True:
            plt.tight_layout()
            plt.close()
        return ax

    def get_pairwise_distance(self, sample_set1, sample_set2=None):
        """
        Calculates the pairwise distance between two set of states.
        """
        if sample_set2 == None:
            diversity_in_set_cal = True
            sample_set2 = sample_set1
        else:
            diversity_in_set_cal = False
        sample_states1 = torch.tensor(sample_set1, device=self.device, dtype=self.float)
        sample_states2 = torch.tensor(sample_set2, device=self.device, dtype=self.float)
        dist_matrix = torch.cdist(sample_states1, sample_states2, p=2)
        if diversity_in_set_cal == True:
            dist_upper_triangle = torch.triu(dist_matrix, diagonal=1)
            dist_vector = dist_upper_triangle[dist_upper_triangle != 0]
            return dist_vector
        else:
            dist_vector = torch.min(dist_matrix, dim=1)[0]
            return dist_vector

    # def plot_reward_samples(self, states, scores, figure_title):
    #     # make compatible with n_dim > 2
    #     fig, ax = plt.subplots()
    #     grid_scores = np.ones((self.length, self.length)) * (-0.0001)
    #     index = states.long().detach().cpu().numpy()
    #     grid_scores[index[:, 0], index[:, 1]] = scores
    #     im = ax.imshow(grid_scores)
    #     divider = make_axes_locatable(ax)
    #     ax.set_title(figure_title)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     plt.colorbar(im, cax=cax)
    #     plt.show()
    #     plt.close()
    #     return fig
