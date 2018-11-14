# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#
from pysc2.agents.random_agent import RandomAgent
from pysc2.tests import utils
from absl.testing import absltest
from environment.parallel_env import ParallelEnvs
from algorithm.script.move_to_beacon import MoveToBeacon
from algorithm.script.parallel_agent import ParallelAgent
from random import randint


class TestParallelEnv(utils.TestCase):
    # tests two sc2env parallel
    def test_parallel_env(self):
        env_arg = {
            'map_name': 'MoveToBeacon',
            'step_mul': 8
        }
        env_num = 2
        parallel_envs = ParallelEnvs.new(env_num=env_num, env_args=env_arg)
        agent_cls = [MoveToBeacon, RandomAgent]
        agents = [agent_cls[randint(0, 1)]() for _ in range(env_num)]

        #  !!! it is not necessary when env_num value is small
        #  !!! this is just an example
        parallel_agents = ParallelAgent(agents)
        parallel_agents.setup(
            parallel_envs.observation_spec(),
            parallel_envs.action_spec())

        obs = parallel_envs.reset()
        for _ in range(100):
            actions = parallel_agents.step(obs)
            acts = [(act,) for act in actions]
            obs = parallel_envs.step(acts)
        parallel_envs.close()
        parallel_agents.close()


if __name__ == "__main__":
    absltest.main()
