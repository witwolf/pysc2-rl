# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/1.
#
from pysc2.agents import base_agent
from pysc2.lib import units
from lib.protoss_macro import PROTOSS_MACROS

class ProtossBaseAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ProtossBaseAgent, self).__init__()
        self.actions = []
        self.frame = 0
        self.frame_skip = 4
        self.check_idle_frame_skip = 10 * self.frame_skip
        self.last_build_frame = 0
        self.build_frame_gap = 3 * (3 * self.frame_skip)

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

    def is_building_macro(self, macro_action):
        return macro_action in [PROTOSS_MACROS.Build_Pylon.id,
                          PROTOSS_MACROS.Build_Gateway.id,
                          PROTOSS_MACROS.Build_Assimilator.id,
                          PROTOSS_MACROS.Build_CyberneticsCore.id]

    def can_build(self):
        return self.frame - self.last_build_frame > self.build_frame_gap

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def future_food(self, obs):
        nexus = len(self.get_all_units_by_type(
            obs, units.Protoss.Nexus))
        pylons = len(self.get_all_units_by_type(
            obs, units.Protoss.Pylon))
        return 15 * (nexus) + 8 * (pylons)

    def step(self, obs):
        super(ProtossBaseAgent, self).step(obs)
        self.frame += 1

