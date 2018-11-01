# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/1.
#

from pysc2.tests import utils
from absl.testing import absltest
from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.agents import base_agent
from pysc2.lib import units
from pysc2.lib.actions import FUNCTIONS
from lib.protoss_macro import PROTOSS_MACROS


class ProtossAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ProtossAgent, self).__init__()
        self.actions = []
        self.aid = 0
        self.frame = 0

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def get_all_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type]

    def get_all_complete_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and int(unit.build_progress) == 100]

    def get_unit_counts(self, obs, unit_type):
        for unit in obs.observation.unit_counts:
            if unit[0] == unit_type:
                return unit[1]
        return 0

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def future_food(self, obs):
        nexus = len(self.get_all_units_by_type(
            obs, units.Protoss.Nexus))
        pylons = len(self.get_all_units_by_type(
            obs, units.Protoss.Pylon))
        return 15 * (nexus) + 8 * (pylons)

    def step(self, obs):
        super(ProtossAgent, self).step(obs)

        self.frame += 1
        if self.frame % 4 != 1:
            return FUNCTIONS.no_op()
        # find macro to execute
        if len(self.actions) == 0:
            # if enough food, train probe
            mineral = obs.observation.player.minerals
            food = self.future_food(obs) - obs.observation.player.food_used
            if mineral > 50 and food > 1:
                self.actions = PROTOSS_MACROS.Train_Probe()
            # build a pylon if not enough food
            elif mineral > 100:
                self.actions = PROTOSS_MACROS.Build_Pylon()

            return FUNCTIONS.no_op()

        action, arg_func = self.actions[0]
        if self.can_do(obs, action.id):
            self.actions = self.actions[1:]
            action_args = arg_func(obs)
            return action(*action_args)

        return FUNCTIONS.no_op()


class TestProtossMacro(utils.TestCase):

    # def test_protoss_macro(self):
    #     attack_enemy = PROTOSS_MACROS.Attack_Enemy()
    #     assert len(attack_enemy) == 3
    #     for func, arg in attack_enemy:
    #         pass
    #     for i in range(3):
    #         pass

    def test_protoss_agent(self):
        agent = ProtossAgent()
        with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.protoss),
                         sc2_env.Bot(sc2_env.Race.terran,
                                     sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True, use_raw_units=True, use_unit_counts=True),
                step_mul=4,
                game_steps_per_episode=0,
                visualize=True) as env:
            agent.setup(env.observation_spec(), env.action_spec())
            timesteps = env.reset()
            agent.reset()
            while True:
                step_actions = [agent.step(timesteps[0])]
                if timesteps[0].last():
                    break
                timesteps = env.step(step_actions)


if __name__ == "__main__":
    absltest.main()
