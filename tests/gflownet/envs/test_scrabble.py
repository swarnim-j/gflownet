import common
import pytest
import torch

from gflownet.envs.seqs.scrabble import Scrabble
from gflownet.utils.common import tlong


@pytest.fixture
def env():
    return Scrabble(max_length=7, device="cpu")


def test__environment_initializes_properly():
    env = Scrabble(max_length=7, device="device")
    import ipdb; ipdb.set_trace()
    assert True
# 
# 
# @pytest.mark.parametrize(
#     "action_space",
#     [
#         [(1,), (2,), (3,), (4,), (5,), (-1,)],
#     ],
# )
# def test__get_action_space__returns_expected(env, action_space):
#     assert set(action_space) == set(env.action_space)
# 
# 
# @pytest.mark.parametrize(
#     "state, parents_expected, parents_a_expected",
#     [
#         (
#             [0, 0, 0, 0, 0],
#             [],
#             [],
#         ),
#         (
#             [1, 0, 0, 0, 0],
#             [[0, 0, 0, 0, 0]],
#             [(1,)],
#         ),
#         (
#             [1, 3, 2, 4, 0],
#             [[1, 3, 2, 0, 0]],
#             [(4,)],
#         ),
#     ],
# )
# def test__get_parents__returns_expected(
#     env, state, parents_expected, parents_a_expected
# ):
#     state = tlong(state, device=env.device)
#     parents_expected = [tlong(parent, device=env.device) for parent in parents_expected]
#     parents, parents_a = env.get_parents(state)
#     for p, p_e in zip(parents, parents_expected):
#         assert torch.equal(p, p_e)
#     for p_a, p_a_e in zip(parents_a, parents_a_expected):
#         assert p_a == p_a_e
# 
# 
# @pytest.mark.parametrize(
#     "state, action, next_state",
#     [
#         (
#             [0, 0, 0, 0, 0],
#             (1,),
#             [1, 0, 0, 0, 0],
#         ),
#         (
#             [1, 0, 0, 0, 0],
#             (3,),
#             [1, 3, 0, 0, 0],
#         ),
#         (
#             [1, 3, 0, 0, 0],
#             (2,),
#             [1, 3, 2, 0, 0],
#         ),
#         (
#             [1, 3, 2, 0, 0],
#             (1,),
#             [1, 3, 2, 1, 0],
#         ),
#         (
#             [1, 3, 2, 1, 0],
#             (4,),
#             [1, 3, 2, 1, 4],
#         ),
#         (
#             [1, 3, 2, 1, 4],
#             (-1,),
#             [1, 3, 2, 1, 4],
#         ),
#         (
#             [1, 3, 2, 0, 0],
#             (-1,),
#             [1, 3, 2, 0, 0],
#         ),
#     ],
# )
# def test__step__returns_expected(env, state, action, next_state):
#     env.set_state(tlong(state, device=env.device))
#     env.step(action)
#     assert torch.equal(env.state, tlong(next_state, device=env.device))
# 
# 
# @pytest.mark.parametrize(
#     "states, states_policy",
#     [
#         (
#             [[0, 0, 0, 0, 0], [1, 3, 2, 1, 4]],
#             [
#                 # fmt: off
#                 [
#                     1, 0, 0, 0, 0, 0, 
#                     1, 0, 0, 0, 0, 0, 
#                     1, 0, 0, 0, 0, 0, 
#                     1, 0, 0, 0, 0, 0, 
#                     1, 0, 0, 0, 0, 0, 
#                 ],
#                 [
#                     0, 1, 0, 0, 0, 0, 
#                     0, 0, 0, 1, 0, 0, 
#                     0, 0, 1, 0, 0, 0, 
#                     0, 1, 0, 0, 0, 0, 
#                     0, 0, 0, 0, 1, 0, 
#                 ],
#                 # fmt: on
#             ],
#         ),
#         (
#             [[1, 3, 2, 1, 4]],
#             [
#                 # fmt: off
#                 [
#                     0, 1, 0, 0, 0, 0, 
#                     0, 0, 0, 1, 0, 0, 
#                     0, 0, 1, 0, 0, 0, 
#                     0, 1, 0, 0, 0, 0, 
#                     0, 0, 0, 0, 1, 0, 
#                 ],
#                 # fmt: on
#             ],
#         ),
#     ],
# )
# def test__states2policy__returns_expected(env, states, states_policy):
#     states = tlong(states, device=env.device)
#     states_policy = tlong(states_policy, device=env.device)
#     assert torch.equal(states_policy, env.states2policy(states))
# 
# 
# @pytest.mark.parametrize(
#     "states, states_proxy",
#     [
#         (
#             [[0, 0, 0, 0, 0], [1, 3, 2, 1, 4]],
#             [[10, 10, 10, 10, 10], [-2, 0, -1, -2, 1]],
#         ),
#         (
#             [[1, 3, 2, 1, 4]],
#             [[-2, 0, -1, -2, 1]],
#         ),
#     ],
# )
# def test__states2proxy__returns_expected(env, states, states_proxy):
#     states = tlong(states, device=env.device)
#     assert env.states2proxy(states) == states_proxy
# 
# 
# @pytest.mark.parametrize(
#     "state, readable",
#     [
#         ([0, 0, 0, 0, 0], "10 10 10 10 10"),
#         ([1, 3, 2, 1, 4], "-2 0 -1 -2 1"),
#     ],
# )
# def test__state2readable__returns_expected(env, state, readable):
#     state = tlong(state, device=env.device)
#     assert env.state2readable(state) == readable
# 
# 
# @pytest.mark.parametrize(
#     "state, readable",
#     [
#         ([0, 0, 0, 0, 0], "10 10 10 10 10"),
#         ([1, 3, 2, 1, 4], "-2 0 -1 -2 1"),
#     ],
# )
# def test__readable2state__returns_expected(env, state, readable):
#     state = tlong(state, device=env.device)
#     assert torch.equal(env.readable2state(readable), state)
# 
# 
# def test__all_env_common__standard(env):
#     print(
#         f"\n\nCommon tests for Sequence with tokens {env.tokens} and "
#         f"max_length {env.max_length}\n"
#     )
#     return common.test__all_env_common(env)
