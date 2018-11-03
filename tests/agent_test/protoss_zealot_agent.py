# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/1.
#

import sys
sys.path.append('.')
from pysc2.lib import units
from pysc2.lib.actions import FUNCTIONS
from lib.protoss_macro import PROTOSS_MACROS
from tests.agent_test.protoss_base_agent import ProtossBaseAgent

class ProtossZealotAgent(ProtossBaseAgent):

    def step(self, obs):
        super(ProtossZealotAgent, self).step(obs)

        if self.frame % self.frame_skip != 1:
            return FUNCTIONS.no_op()
        # find macro to execute
        if len(self.actions) == 0:
            # if enough food, train probe
            mineral = obs.observation.player.minerals
            food = self.future_food(obs) - obs.observation.player.food_used
            idle_workers = obs.observation.player.idle_worker_count
            # if can build a building
            if idle_workers > 0:
                self.actions = PROTOSS_MACROS.Collect_Mineral()
            elif mineral > 50 and food > 1 and self.get_unit_counts(obs, units.Protoss.Probe) < 16:
                self.actions = PROTOSS_MACROS.Train_Probe()
            elif mineral > 100 and food > 2 and len(self.get_all_complete_units_by_type(obs, units.Protoss.Gateway)) > 0:
                self.actions = PROTOSS_MACROS.Train_Zealot()
            elif self.can_build():
                if food < 4 and mineral > 100:
                    self.actions = PROTOSS_MACROS.Build_Pylon()
                    self.last_build_frame = self.frame
                elif mineral > 150 and self.get_unit_counts(obs, units.Protoss.Gateway) < 8:
                    self.actions = PROTOSS_MACROS.Build_Gateway()
                    self.last_build_frame = self.frame

            return FUNCTIONS.no_op()

        action, arg_func = self.actions[0]
        self.actions = self.actions[1:]
        if self.can_do(obs, action.id):
            action_args = arg_func(obs)
            return action(*action_args)

        return FUNCTIONS.no_op()

