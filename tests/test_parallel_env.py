# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#
from pysc2.agents.random_agent import RandomAgent
from pysc2.tests import utils
from absl.testing import absltest
from environment.parallel_env import ParallelEnvs
from algorithm.script.move_to_beacon import MoveToBeacon
from algorithm.script.parallel_agent import ParallelAgent


class TestParallelEnv(utils.TestCase):
    # tests two sc2env parallel
    def test_parallel_env(self):
        env_arg = {
            'map_name': 'MoveToBeacon',
            'step_mul': 8
        }
        env_num = 2
        parallel_envs = ParallelEnvs(env_num=env_num, env_args=env_arg)
        agents = [MoveToBeacon(), RandomAgent()]

        #  !!! it is not necessary when env_num value is small
        #  !!! this is just an example
        parallel_agents = ParallelAgent(agents)
        parallel_agents.setup(
            list(spec[0] for spec in parallel_envs.observation_spec()),
            list(spec[0] for spec in parallel_envs.action_spec())
        )

        obs = parallel_envs.reset()
        for _ in range(100):
            actions = parallel_agents.step(list(o[0] for o in obs))
            obs = parallel_envs.step([(actions[0],), (actions[1],)])
        parallel_envs.close()
        parallel_agents.close()


if __name__ == "__main__":
    absltest.main()
