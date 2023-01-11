"""
Classes to represent aptamers environments
"""
from typing import List
import itertools
import numpy as np
import pandas as pd
from gflownet.envs.base import GFlowNetEnv
import time
from sklearn.model_selection import GroupKFold, train_test_split
from clamp_common_eval.defaults import get_default_data_splits


class AMP(GFlowNetEnv):
    """
    Anti-microbial peptide  sequence environment

    Attributes
    ----------
    max_seq_length : int
        Maximum length of the sequences

    min_seq_length : int
        Minimum length of the sequences

    nalphabet : int
        Number of letters in the alphabet

    state : list
        Representation of a sequence (state), as a list of length max_seq_length where
        each element is the index of a letter in the alphabet, from 0 to (nalphabet -
        1).

    done : bool
        True if the sequence has reached a terminal state (maximum length, or stop
        action executed.

    func : str
        Name of the reward function

    n_actions : int
        Number of actions applied to the sequence

    proxy : lambda
        Proxy model
    """

    def __init__(
        self,
        max_seq_length=50,
        min_seq_length=1,
        nalphabet=20,
        min_word_len=1,
        max_word_len=1,
        proxy=None,
        oracle=None,
        reward_beta=1,
        env_id=None,
        energies_stats=None,
        reward_norm=1.0,
        reward_norm_std_mult=0.0,
        reward_func="power",
        denorm_proxy=False,
        **kwargs,
    ):
        super(AMP, self).__init__(
            env_id,
            reward_beta,
            reward_norm,
            reward_norm_std_mult,
            reward_func,
            energies_stats,
            denorm_proxy,
            proxy,
            oracle,
            **kwargs,
        )
        self.state = []
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.nalphabet = nalphabet
        self.obs_dim = self.nalphabet * self.max_seq_length
        self.min_word_len = min_word_len
        self.max_word_len = max_word_len
        self.action_space = self.get_actions_space()
        self.eos = len(self.action_space)
        self.max_traj_len = self.get_max_traj_len()
        self.vocab = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]
        if self.proxy_state_format == "ohe":
            self.state2proxy = self.state2obs
        elif self.proxy_state_format == "oracle":
            self.state2proxy = self.state2oracle

    def get_alphabet(self, vocab=None, index_int=True):
        if vocab is None:
            vocab = self.vocab
        if index_int:
            alphabet = dict((i, a) for i, a in enumerate(vocab))
        else:
            alphabet = dict((a, i) for i, a in enumerate(vocab))
        return alphabet

    def get_actions_space(self):
        """
        Constructs list with all possible actions
        """
        assert self.max_word_len >= self.min_word_len
        valid_wordlens = np.arange(self.min_word_len, self.max_word_len + 1)
        alphabet = [a for a in range(self.nalphabet)]
        actions = []
        for r in valid_wordlens:
            actions_r = [el for el in itertools.product(alphabet, repeat=r)]
            actions += actions_r
        return actions

    def get_max_traj_len(
        self,
    ):
        return self.max_seq_length / self.min_word_len + 1

    def reward_arbitrary_i(self, state):
        if len(state) > 0:
            return (state[-1] + 1) * len(state)
        else:
            return 0

    def state2oracle(self, state_list):
        # TODO
        """
        Prepares a sequence in "GFlowNet format" for the oracles.

        Args
        ----
        state_list : list of lists
            List of sequences.
        """
        queries = ["".join(self.state2readable(state)) for state in state_list]
        # queries = [s + [-1] * (self.max_seq_length - len(s)) for s in state_list]
        # queries = np.array(queries, dtype=int)
        # if queries.ndim == 1:
        #     queries = queries[np.newaxis, ...]
        # queries += 1
        # if queries.shape[1] == 1:
        #     import ipdb

        #     ipdb.set_trace()
        #     queries = np.column_stack((queries, np.zeros(queries.shape[0])))
        return queries

    def state2obs(self, state=None):
        """
        Transforms the sequence (state) given as argument (or self.state if None) into a
        one-hot encoding. The output is a list of length nalphabet * max_seq_length,
        where each n-th successive block of nalphabet elements is a one-hot encoding of
        the letter in the n-th position.

        Example:
          - Sequence: AATGC
          - state: [0, 1, 3, 2]
                    A, T, G, C
          - state2obs(state): [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                              |     A    |      T    |      G    |      C    |

        If max_seq_length > len(s), the last (max_seq_length - len(s)) blocks are all
        0s.
        """
        if state is None:
            state = self.state.copy()

        z = np.zeros(self.obs_dim, dtype=np.float32)

        if len(state) > 0:
            if hasattr(
                state[0], "device"
            ):  # if it has a device at all, it will be cuda (CPU numpy array has no dev
                state = [subseq.cpu().detach().numpy() for subseq in state]

            z[(np.arange(len(state)) * self.nalphabet + state)] = 1
        return z

    def obs2state(self, obs: List) -> List:
        """
        Transforms the one-hot encoding version of a sequence (state) given as argument
        into a a sequence of letter indices.

        Example:
          - Sequence: AATGC
          - obs: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                 |     A    |      A    |      T    |      G    |      C    |
          - state: [0, 0, 1, 3, 2]
                    A, A, T, G, C
        """
        obs_mat = np.reshape(obs, (self.max_seq_length, self.nalphabet))
        state = np.where(obs_mat)[1].tolist()
        return state

    def state2readable(self, state, alphabet=None):
        """
        Transforms a sequence given as a list of indices into a sequence of letters
        according to an alphabet.
        """
        if alphabet == None:
            alphabet = self.get_alphabet(index_int=True)
        return [alphabet[el] for el in state]

    def readable2state(self, letters, alphabet=None):
        # TODO
        """
        Transforms a sequence given as a list of letters into a sequence of indices
        according to an alphabet.
        """
        if alphabet == None:
            alphabet = self.get_alphabet(index_int=False)
        alphabet = {v: k for k, v in alphabet.items()}
        return [alphabet[el] for el in letters]

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.state = []
        self.n_actions = 0
        self.done = False
        self.id = env_id
        return self

    def get_parents(self, state=None, done=None):
        """
        Determines all parents and actions that lead to sequence state

        Args
        ----
        state : list
            Representation of a sequence (state), as a list of length max_seq_length
            where each element is the index of a letter in the alphabet, from 0 to
            (nalphabet - 1).

        action : int
            Last action performed, only to determine if it was eos.

        Returns
        -------
        parents : list
            List of parents as state2obs(state)

        actions : list
            List of actions that lead to state for each parent in parents
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [self.state2obs(state)], [self.eos]
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                is_parent = state[-len(a) :] == list(a)
                if not isinstance(is_parent, bool):
                    is_parent = all(is_parent)
                if is_parent:
                    parents.append(self.state2obs(state[: -len(a)]))
                    actions.append(idx)
        return parents, actions

    def get_parents_debug(self, state=None, done=None):
        """
        Like get_parents(), but returns state format
        """
        obs, actions = self.get_parents(state, done)
        parents = [self.obs2state(el) for el in obs]
        return parents, actions

    def step(self, action_idx):
        """
        Executes step given an action index

        If action_idx is smaller than eos (no stop), add action to next
        position.

        See: step_daug()
        See: step_chain()

        Args
        ----
        action_idx : int
            Index of action in the action space. a == eos indicates "stop action"

        Returns
        -------
        self.state : list
            The sequence after executing the action

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        # If only possible action is eos, then force eos
        if len(self.state) == self.max_seq_length:
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # If action is not eos, then perform action
        if action_idx != self.eos:
            action = self.action_space[action_idx]
            state_next = self.state + list(action)
            if len(state_next) > self.max_seq_length:
                valid = False
            else:
                self.state = state_next
                valid = True
                self.n_actions += 1
            return self.state, action_idx, valid
        # If action is eos, then perform eos
        else:
            if len(self.state) < self.min_seq_length:
                valid = False
            else:
                self.done = True
                valid = True
                self.n_actions += 1
            return self.state, self.eos, valid

    def get_mask_invalid_actions(self, state=None, done=None):
        """
        Returns a vector of length the action space + 1: True if action is invalid
        given the current state, False otherwise.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(len(self.action_space) + 1)]
        mask = [False for _ in range(len(self.action_space) + 1)]
        seq_length = len(state)
        if seq_length < self.min_seq_length:
            mask[self.eos] = True
        for idx, a in enumerate(self.action_space):
            if seq_length + len(a) > self.max_seq_length:
                mask[idx] = True
        return mask

    def no_eos_mask(self, state=None):
        """
        Returns True if no eos action is allowed given state
        """
        if state is None:
            state = self.state.copy()
        return len(state) < self.min_seq_length

    def true_density(self, max_states=1e6):
        """
        Computes the reward density (reward / sum(rewards)) of the whole space, if the
        dimensionality is smaller than specified in the arguments.

        Returns
        -------
        Tuple:
          - normalized reward for each state
          - states
          - (un-normalized) reward)
        """
        if self._true_density is not None:
            return self._true_density
        if self.nalphabet**self.max_seq_length > max_states:
            return (None, None, None)
        state_all = np.int32(
            list(
                itertools.product(*[list(range(self.nalphabet))] * self.max_seq_length)
            )
        )
        traj_rewards, state_end = zip(
            *[
                (self.proxy(state), state)
                for state in state_all
                if len(self.get_parents(state, False)[0]) > 0 or sum(state) == 0
            ]
        )
        traj_rewards = np.array(traj_rewards)
        self._true_density = (
            traj_rewards / traj_rewards.sum(),
            list(map(tuple, state_end)),
            traj_rewards,
        )
        return self._true_density

    def make_train_set(
        self,
        ntrain=1000,
        seed=20,
        oracle=None,
        output_csv=None,
        split="D1",
        nfold=5,
        load_precomputed_scores=False,
    ):
        """
        Constructs a randomly sampled train set by calling the oracle

        Args
        ----
        ntrain : int
            Number of train samples.

        seed : int
            Random seed.

        output_csv: str
            Optional path to store the test set as CSV.
        """
        source = get_default_data_splits(setting="Target")
        rng = np.random.RandomState(142857)
        self.data = source.sample(
            split, -1
        )  # dictionary with two keys 'AMP' and 'nonAMP' and values as lists
        self.nfold = nfold  # 5
        if split == "D1":
            groups = np.array(source.d1_pos.group)
        if split == "D2":
            groups = np.array(source.d2_pos.group)
        if split == "D":
            groups = np.concatenate(
                (np.array(source.d1_pos.group), np.array(source.d2_pos.group))
            )

        n_pos, n_neg = len(self.data["AMP"]), len(self.data["nonAMP"])
        pos_train, pos_valid = next(
            GroupKFold(nfold).split(np.arange(n_pos), groups=groups)
        )  # makes k folds in such a way that each group will appear exactly once in the test set across all folds
        neg_train, neg_valid = next(
            GroupKFold(nfold).split(
                np.arange(n_neg), groups=rng.randint(0, nfold, n_neg)
            )
        )  # we get indices above

        pos_train = [
            self.data["AMP"][i] for i in pos_train
        ]  # here we make the dataset using the indices
        neg_train = [self.data["nonAMP"][i] for i in neg_train]
        pos_valid = [self.data["AMP"][i] for i in pos_valid]
        neg_valid = [
            self.data["nonAMP"][i] for i in neg_valid
        ]  # all these are lists of strings, just the sequences, no energies
        self.train = pos_train + neg_train
        self.valid = pos_valid + neg_valid

        if oracle is None:
            oracle = self.oracle

        if load_precomputed_scores:
            loaded = self._load_precomputed_scores(split)
        else:
            self.train_scores = oracle(self.train)  # train is a list of strings
            self.valid_scores = oracle(self.valid)
            df_train = pd.DataFrame(
                {
                    "samples": self.readable2state(self.train),
                    "sequences": self.train,
                    "energies": self.train_scores,
                }
            )
            df_valid = pd.DataFrame(
                {
                    "samples": self.readable2state(self.valid),
                    "sequences": self.valid,
                    "energies": self.valid_scores,
                }
            )

        if output_csv:
            df_train.to_csv(output_csv)
        return df_train, df_valid

    # TODO: improve approximation of uniform
    def make_test_set(
        self,
        path_base_dataset,
        ntest,
        min_length=0,
        max_length=np.inf,
        seed=167,
        output_csv=None,
    ):
        """
        Constructs an approximately uniformly distributed (on the score) set, by
        selecting samples from a larger base set.

        Args
        ----
        path_base_dataset : str
            Path to a CSV file containing the base data set.

        ntest : int
            Number of test samples.

        seed : int
            Random seed.

        dask : bool
            If True, use dask to efficiently read a large base file.

        output_csv: str
            Optional path to store the test set as CSV.
        """
        if path_base_dataset is None:
            return None, None
        times = {
            "all": 0.0,
            "indices": 0.0,
        }
        t0_all = time.time()
        if seed:
            np.random.seed(seed)
        df_base = pd.read_csv(path_base_dataset, index_col=0)
        df_base = df_base.loc[
            (df_base["samples"].map(len) >= min_length)
            & (df_base["samples"].map(len) <= max_length)
        ]
        energies_base = df_base["energies"].values
        min_base = energies_base.min()
        max_base = energies_base.max()
        distr_unif = np.random.uniform(low=min_base, high=max_base, size=ntest)
        # Get minimum distance samples without duplicates
        t0_indices = time.time()
        idx_samples = []
        for idx in tqdm(range(ntest)):
            dist = np.abs(energies_base - distr_unif[idx])
            idx_min = np.argmin(dist)
            if idx_min in idx_samples:
                idx_sort = np.argsort(dist)
                for idx_next in idx_sort:
                    if idx_next not in idx_samples:
                        idx_samples.append(idx_next)
                        break
            else:
                idx_samples.append(idx_min)
        t1_indices = time.time()
        times["indices"] += t1_indices - t0_indices
        # Make test set
        df_test = df_base.iloc[idx_samples]
        if output_csv:
            df_test.to_csv(output_csv)
        t1_all = time.time()
        times["all"] += t1_all - t0_all
        return df_test, times

    @staticmethod
    def np2df(test_path, al_init_length, al_queries_per_iter, pct_test, data_seed):
        data_dict = np.load(test_path, allow_pickle=True).item()
        letters = numbers2letters(data_dict["samples"])
        df = pd.DataFrame(
            {
                "samples": letters,
                "energies": data_dict["energies"],
                "train": [False] * len(letters),
                "test": [False] * len(letters),
            }
        )
        # Split train and test section of init data set
        rng = np.random.default_rng(data_seed)
        indices = rng.permutation(al_init_length)
        n_tt = int(pct_test * len(indices))
        indices_tt = indices[:n_tt]
        indices_tr = indices[n_tt:]
        df.loc[indices_tt, "test"] = True
        df.loc[indices_tr, "train"] = True
        # Split train and test the section of each iteration to preserve splits
        idx = al_init_length
        iters_remaining = (len(df) - al_init_length) // al_queries_per_iter
        indices = rng.permutation(al_queries_per_iter)
        n_tt = int(pct_test * len(indices))
        for it in range(iters_remaining):
            indices_tt = indices[:n_tt] + idx
            indices_tr = indices[n_tt:] + idx
            df.loc[indices_tt, "test"] = True
            df.loc[indices_tr, "train"] = True
            idx += al_queries_per_iter
        return df
