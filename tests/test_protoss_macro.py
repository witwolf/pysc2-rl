# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/1.
#

import sys

sys.path.append('.')
from pysc2.tests import utils
from absl.testing import absltest
from pysc2.env import sc2_env
from pysc2.lib import features
from lib.protoss_macro import U
from lib.protoss_macro import PROTOSS_MACROS
from environment.macro_env import default_macro_env_maker
from tests.agent_test.protoss_stalker_agent import ProtossStalkerAgent


class TestProtossMacro(utils.TestCase):
    def test_build_pylon(self):
        with self._create_macro_env() as env:
            obs = env.reset()[0]
            while True:
                act = PROTOSS_MACROS.No_op()
                if U.can_build_pylon(obs):
                    act = PROTOSS_MACROS.Build_Pylon()
                obs = env.step((act,))[0]

    def test_build_gateway(self):
        pass

    # def test_stalker_agent(self):
    #     agent = ProtossStalkerAgent()
    #     with self._create_sc2env() as env:
    #         agent.setup(env.observation_spec(), env.action_spec())
    #         obs = env.reset()
    #         agent.reset()
    #         while True:
    #             step_actions = [agent.step(obs[0])]
    #             if obs[0].last():
    #                 break
    #             obs = env.step(step_actions)
    #
    # def test_zealot_agent(self):
    #     pass

    def _create_macro_env(self):
        env = default_macro_env_maker(dict(map_name='Simple64'))
        return env

    def _create_sc2env(self):
        env = sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.protoss),
                     sc2_env.Bot(sc2_env.Race.terran,
                                 sc2_env.Difficulty.very_easy)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True, use_raw_units=True, use_unit_counts=True),
            step_mul=4,
            game_steps_per_episode=0,
            visualize=True)
        return env


if __name__ == "__main__":
    absltest.main()
