"""
Base class to stack multiple environments.
"""
import json
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional, OrderedDict, Tuple, Union

import torch
from torch import Tensor
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tbool, tfloat, tlong


class Stack(GFlowNetEnv):
    """
    Base class to create new environments by stacking multiple environments.

    This class imposes the order specified in the creation, such that the actions
    corresponding to a sub-environment are not valid until the preceding
    sub-environment in the stack reaches the final ("done") state.

    This class enables the incorporation of constraints across sub-environments via the
    apply_constraints() method, which must be implemented by the children classes.

    For example, a new environment can be created by stacking the (continuous) Cube and
    the Tetris.
    """

    def __init__(
        self,
        subenvs: Tuple[GFlowNetEnv],
        **kwargs,
    ):
        """
        Args
        ----

        subenvs : tuple
            A tuple containing the ordered list of the sub-environments to be stacked.
        """
        self.subenvs = OrderedDict({idx: subenv for idx, subenv in enumerate(subenvs)})
        self.n_subenvs = len(self.subenvs)

        # States are represented as a list of subenv's states, front-padded by the
        # index of the current subenv (stage). The source state is the list of source
        # states, starting with stage 0.
        self.source = [0] + [subenv.source for subenv in self.subenvs.values()]

        # Get action dimensionality by computing the maximum action length among all
        # sub-environments, and adding 1 to indicate the sub-environment.
        self.action_dim = max([len(subenv.eos) for subenv in self.subenvs.values()]) + 1

        # EOS is EOS of the last stage
        self.eos = self._pad_action(
            self.subenvs[self.n_subenvs - 1].eos, stage=self.n_subenvs - 1
        )

        # Get the mask dimensionality by computing the maximum length among all
        # sub-environments, plus n_subenvs indicate the sub-environment as a one-hot
        # encoding (to keep it boolean).
        self.mask_dim = (
            max([subenv.mask_dim for subenv in self.subenvs.values()]) + self.n_subenvs
        )

        # The stack is continuous if any subenv is continuous
        self.continuous = any([subenv.continuous for subenv in self.subenvs.values()])

        # Base class init
        super().__init__(
            fixed_distr_params=[
                subenv.fixed_distr_params for subenv in self.subenvs.values()
            ],
            random_distr_params=[
                subenv.random_distr_params for subenv in self.subenvs.values()
            ],
            **kwargs,
        )

    def get_action_space(self) -> List[Tuple]:
        """
        Constructs list with all possible actions, including eos.

        The action space of a stack environment is the concatenation of the actions of
        all the sub-environments.

        In order to make all actions the same length (required to construct batches of
        actions as a tensor), the actions are zero-padded from the back.

        In order to make all actions unique, the stage index is added as the first
        element of the action.

        See: _pad_action(), _depad_action()
        """
        action_space = []
        for stage, subenv in self.subenvs.items():
            action_space.extend(
                [self._pad_action(action, stage) for action in subenv.action_space]
            )
        return action_space

    def _pad_action(self, action: Tuple, stage: int) -> Tuple:
        """
        Pads an action by adding the stage index as the first element and zeros as
        padding.

        See: get_action_space()
        """
        return (stage,) + action + (0,) * (self.action_dim - len(action) - 1)

    def _depad_action(self, action: Tuple, stage: int) -> Tuple:
        """
        Reverses padding operation, such that the resulting action can be passed to the
        underlying environment.

        See: _pad_action()
        """
        return action[1 : 1 + len(self.subenvs[stage].eos)]

    def get_max_traj_length(self) -> int:
        return sum([subenv.get_max_traj_length() for subenv in self.subenvs.values()])

    def get_policy_output(self, params: list[dict]) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model.

        The policy output is the concatenation of the policy outputs of all the
        sub-environments.
        """
        return torch.cat(
            [
                subenv.get_policy_output(params_subenv)
                for subenv, params_subenv in zip(self.subenvs.values(), params)
            ]
        )

    def _get_policy_outputs_of_subenv(
        self, policy_outputs: TensorType["n_states", "policy_output_dim"], stage: int
    ):
        """
        Returns the columns of the policy outputs that correspond to the
        sub-environment indicated by stage.

        Args
        ----
        policy_outputs : tensor
            A tensor containing a batch of policy outputs. It is assumed that all the
            rows in the this tensor correspond to the same stage.

        stage : int
            Index of the sub-environment of which the corresponding columns of the
            policy outputs are to be extracted.
        """
        init_col = 0
        for stg, subenv in self.subenvs.items():
            end_col = init_col + subenv.policy_output_dim
            if stg == stage:
                return policy_outputs[:, init_col:end_col]
            init_col = end_col

    def reset(self, env_id: Union[int, str] = None):
        # TODO: if the properties of the environment change due to constraints, then we
        # may have to store the original subenvs separately and use them here.
        for subenv in self.subenvs.values():
            subenv.reset()
        super().reset(env_id=env_id)
        return self

    # TODO: do we need a method for this?
    def _get_stage(self, state: Optional[List] = None) -> int:
        """
        Returns the stage of the current environment from self.state[0] or from the
        state passed as an argument.
        """
        if state is None:
            state = self.state
        return state[0]

    # TODO: do we need a method for this?
    def _get_state_of_subenv(self, state: List, stage: Optional[int] = None):
        """
        Returns the part of the state corresponding to the subenv indicated by stage.

        Args
        ----
        state : list
            A state of the parent stack environment.

        stage : int
            Index of the sub-environment of which the corresponding part of the
            state is to be extracted. If None, the stage of the state is used.
        """
        if stage is None:
            stage = self._get_stage(state)
        return state[stage + 1]

    def get_mask_invalid_actions_forward(
        self, state: Optional[List[int]] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Computes the forward actions mask of the state.

        The mask of the stack environment is the mask of the current sub-environment,
        preceded by a one-hot encoding of the index of the subenv and padded with False
        up to mask_dim. Including only the relevant mask saves memory and computation.
        """
        state = self._get_state(state)
        stage = self._get_stage(state)
        done = self._get_done(done)

        subenv = self.subenvs[stage]
        stage_onehot = [False] * self.n_subenvs
        stage_onehot[stage] = True
        padding = [False] * (self.mask_dim - (subenv.mask_dim + 3))
        return (
            stage_onehot
            + subenv.get_mask_invalid_actions_forward(
                self._get_state_of_subenv(state, stage), done
            )
            + padding
        )

    def get_mask_invalid_actions_backward(
        self, state: Optional[List[int]] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Computes the backward actions mask of the state.

        The mask of the stack environment is the mask of the relevant sub-environment,
        preceded by a one-hot encoding of the index of the subenv and padded with False
        up to mask_dim. Including only the relevant mask saves memory and computation.

        The relevant sub-environment regarding the backward mask is always the current
        sub-environment except if the state of the sub-environment is the source, in
        which case the mask must be the one of the preceding sub-environment, so as to
        sample its EOS action.

        Exceptions to the above are:
            - if done is True, in which case the current sub-environment is the last
              stage and the EOS action must come from itself, not the preceding subenv.
            - if the current stage is the first sub-environment, in which case there is
              no preceding stage.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        stage = self._get_stage(state)

        subenv = self.subenvs[stage]
        if (
            stage > 0
            and not done
            and self.equal(self._get_state_of_subenv(state, stage), subenv.source)
        ):
            stage -= 1
            subenv = self.subenvs[stage]
            done = True

        stage_onehot = [False] * self.n_subenvs
        stage_onehot[stage] = True
        padding = [False] * (self.mask_dim - (subenv.mask_dim + 3))
        return (
            stage_onehot
            + subenv.get_mask_invalid_actions_backward(
                self._get_state_of_subenv(state, stage), done
            )
            + padding
        )

    # TODO: do we need a method for this?
    def _update_state(self, stage: int):
        """
        Updates the global state based on the states of the sub-environments and the
        stage passed as an argument.
        """
        return [stage] + [subenv.state for subenv in self.subenvs.values()]

    def step(
        self, action: Tuple, skip_mask_check: bool = False
    ) -> Tuple[List, Tuple, bool]:
        """
        Executes forward step given an action.

        The action is performed by the corresponding sub-environment and then the
        global state is updated accordingly. If the action is the EOS of the
        sub-environment, the stage is advanced and constraints are set on the
        subsequent sub-environment.

        Args
        ----
        action : tuple
            Action to be executed. The input action is global, that is padded.

        Returns
        -------
        self.state : list
            The state after executing the action.

        action : int
            Action executed.

        valid : bool
            False, if the action is not allowed for the current state. True otherwise.
        """
        # If done, exit immediately
        if self.done:
            return self.state, action, False

        # Get stage, subenv, and action of subenv
        stage = self._get_stage(self.state)
        subenv = self.subenvs[stage]
        action_subenv = self._depad_action(action, stage)

        # Perform pre-step from subenv - if it was done from the stack env there could
        # be a mismatch between mask and action space due to continuous subenvs.
        action_to_check = subenv.action2representative(action_subenv)
        # Skip mask check if stage is lattice parameters (continuous actions)
        if subenv.continuous:
            skip_mask_check = True
        do_step, _, _ = subenv._pre_step(
            action_to_check,
            skip_mask_check=(skip_mask_check or self.skip_mask_check),
        )
        if not do_step:
            return self.state, action, False

        # Call step of current subenvironment
        _, action_subenv, valid = subenv.step(action_subenv)

        # If action is invalid, exit immediately. Otherwise increment actions and go on
        if not valid:
            return self.state, action, False
        self.n_actions += 1

        # If action is EOS of subenv, check if global EOS, advance stage and set
        # constraints
        if action_subenv == subenv.eos:
            # Check if global EOS
            if action == self.eos:
                self.done = True
            else:
                stage += 1
                self._apply_constraints()

        # Update gloabl state and return
        self.state = self._update_state(stage)
        return self.state, action, valid

    def step_backwards(
        self, action: Tuple, skip_mask_check: bool = False
    ) -> Tuple[List, Tuple, bool]:
        """
        Executes backward step given an action.

        The action is performed by the corresponding sub-environment and then the
        global state is updated accordingly. If the updated state of the
        sub-environment becomes its source, the stage is decreased.

        Args
        ----
        action : tuple
            Action to be executed. The input action is global, that is padded.

        Returns
        -------
        self.state : list
            The state after executing the action.

        action : int
            Action executed.

        valid : bool
            False, if the action is not allowed for the current state. True otherwise.
        """
        # Get stage from action (not from state), subenv and action of subenv
        stage = action[0]
        subenv = self.subenvs[stage]
        action_subenv = self._depad_action(action, stage)

        # If stage of action and state are different, action must be eos of subenv
        if stage != self._get_stage(self.state):
            assert action_subenv == subenv.eos

        # Perform pre-step from subenv - if it was done from the "superenv" there could
        # be a mismatch between mask and action space due to continuous subenvs.
        action_to_check = subenv.action2representative(action_subenv)
        # Skip mask check if stage is lattice parameters (continuous actions)
        if subenv.continuous:
            skip_mask_check = True
        do_step, _, _ = subenv._pre_step(
            action_to_check,
            backward=True,
            skip_mask_check=(skip_mask_check or self.skip_mask_check),
        )
        if not do_step:
            return self.state, action, False

        # Call step of current subenvironment
        state_next, _, valid = subenv.step_backwards(action_subenv)

        # If action is invalid, exit immediately. Otherwise continue,
        if not valid:
            return self.state, action, False
        self.n_actions += 1

        # If action from done, set done False
        if self.done:
            assert action == self.eos
            self.done = False

        self.state = self._update_state(stage)
        return self.state, action, valid

    def _apply_constraints(self):
        """
        Applies constraints across sub-environments. No constraints are applied by
        default, but this method may be overriden by children classes to incorporate
        specific constraints.

        This method is used in step() and set_state().
        """
        pass

    def set_state(self, state: List, done: Optional[bool] = False):
        """
        Sets a state and done.

        The correct state and done of each sub-environment are set too.
        """
        super().set_state(state, done)

        # Set state and done of each sub-environment
        n_done = self._get_stage(state) + int(done)
        dones = (True,) * n_done + (False,) * (self.n_subenvs - n_done)
        for (stage, subenv), done_subenv in zip(self.subenvs.items(), dones):
            subenv.set_state(self._get_state_of_subenv(self.state, stage), done_subenv)

        # Apply constraints
        self._apply_constraints()

        return self

    def sample_actions_batch(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "policy_output_dim"]] = None,
        states_from: List = None,
        is_backward: Optional[bool] = False,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 10,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs.

        This method calls the sample_actions_batch() method of the sub-environment
        corresponding to each state in the batch.

        Note that in order to call sample_actions_batch() of the sub-environments, we
        need to first extract the part of the policy outputs, the masks and the states
        that correspond to the sub-environment.
        """
        # Get the relevant stage of each mask from the one-hot prefix
        stages = torch.where(mask[:, : self.n_subenvs])[1]
        stages_int = stages.tolist()
        states_dict = {stage: [] for stage in self.subenvs.keys()}
        """
        A dictionary with keys equal to the stage indices and the values are the list
        of states in the stage of the key. The states are only the part corresponding
        to the sub-environment.
        """
        for state, stage in zip(states_from, stages_int):
            states_dict[stage].append(self._get_state_of_subenv(state, stage))

        # Sample actions from each sub-environment
        actions_logprobs_dict = {}
        for stage, subenv in self.subenvs.items():
            stage_mask = stages == stage
            if not torch.any(stage_mask):
                continue
            actions_logprobs_dict[stage] = subenv.sample_actions_batch(
                self._get_policy_outputs_of_subenv(policy_outputs[stage_mask], stage),
                mask[stage_mask, self.n_subenvs : self.n_subenvs + subenv.mask_dim],
                states_dict[stage],
                is_backward,
                sampling_method,
                temperature_logits,
                max_sampling_attempts,
            )

        # Stitch all actions in the right order, with the right padding
        actions = []
        for stage in stages_int:
            actions.append(
                self._pad_action(actions_logprobs_dict[stage][0].pop(0), stage)
            )
        return actions, None

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: TensorType["n_states", "actions_dim"],
        mask: TensorType["n_states", "mask_dim"],
        states_from: List,
        is_backward: bool,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.

        Args
        ----
        policy_outputs : tensor
            The output of the GFlowNet policy model.

        mask : tensor
            The mask containing information about invalid actions and special cases.

        actions : tensor
            The actions (global) from each state in the batch for which to compute the
            log probability.

        states_from : tensor
            The states originating the actions, in environment format.

        is_backward : bool
            True if the actions are backward, False if the actions are forward
            (default).
        """
        n_states = policy_outputs.shape[0]
        # Get the relevant stage of each mask from the one-hot prefix
        stages = torch.where(mask[:, : self.n_subenvs])[1]
        stages_int = stages.tolist()
        states_dict = {stage: [] for stage in self.subenvs.keys()}
        """
        A dictionary with keys equal to Stage and the values are the list of states in
        the stage of the key. The states are only the part corresponding to the
        sub-environment.
        """
        for state, stage in zip(states_from, stages_int):
            states_dict[stage].append(self._get_state_of_subenv(state, stage))

        # Compute logprobs from each sub-environment
        logprobs = torch.empty(n_states, dtype=self.float, device=self.device)
        for stage, subenv in self.subenvs.items():
            stage_mask = stages == stage
            if not torch.any(stage_mask):
                continue
            logprobs[stage_mask] = subenv.get_logprobs(
                self._get_policy_outputs_of_subenv(policy_outputs[stage_mask], stage),
                actions[stage_mask, 1 : 1 + len(subenv.eos)],
                mask[stage_mask, self.n_subenvs : self.n_subenvs + subenv.mask_dim],
                states_dict[stage],
                is_backward,
            )
        return logprobs

    def states2policy(
        self, states: List[List]
    ) -> TensorType["batch", "state_policy_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: simply
        a concatenation of the policy-format states of the sub-environments.

        Args
        ----
        states : list
            A batch of states in environment format.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        return torch.cat(
            [
                subenv.states2policy([state[stage + 1] for state in states])
                for stage, subenv in self.subenvs.items()
            ],
            dim=1,
        )

    def states2proxy(
        self, states: List[List]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in "environment format" for a proxy: simply a
        concatenation of the proxy-format states of the sub-environments.

        Args
        ----
        states : list
            A batch of states in environment format.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        return torch.cat(
            [
                subenv.states2proxy([state[stage + 1] for state in states])
                for stage, subenv in self.subenvs.items()
            ],
            dim=1,
        )

    @staticmethod
    def equal(state_x, state_y):
        """
        Overwrites equal() to account for the composite nature of the states.
        """
        return all([GFlowNetEnv.equal(sx, sy) for sx, sy in zip(state_x, state_y)])
