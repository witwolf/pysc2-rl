# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/1.
#

import sys
sys.path.append('.')
from pysc2.lib import units
from pysc2.lib.actions import FUNCTIONS
from lib.protoss_macro import PROTOSS_MACROS
from tests.agent_test.protoss_base_agent import ProtossBaseAgent

class ProtossStalkerAgent(ProtossBaseAgent):

    def step(self, obs):
        super(ProtossStalkerAgent, self).step(obs)

        if self.frame % self.frame_skip != 1:
            return FUNCTIONS.no_op()
        # find macro to execute
        if len(self.actions) == 0:
            assigned_gas_harvesters = [unit.assigned_harvesters for unit in obs.observation.feature_units if unit.unit_type == units.Protoss.Assimilator]
            gasHarvestersFull = True
            gasHarvestersOverload = False
            for assigned_gas_harvester in assigned_gas_harvesters:
                if assigned_gas_harvester < 3:
                    gasHarvestersFull = False
                if assigned_gas_harvester > 3:
                    gasHarvestersOverload = True
            # if enough food, train probe
            mineral = obs.observation.player.minerals
            gas = obs.observation.player.vespene
            food = self.future_food(obs) - obs.observation.player.food_used
            idle_workers = obs.observation.player.idle_worker_count
            # if can build a building
            if idle_workers > 0:
                self.actions = PROTOSS_MACROS.Callback_Idle_Workers()
            elif gasHarvestersOverload:
                self.actions = PROTOSS_MACROS.Collect_Mineral()
            elif mineral > 50 and food > 1 and self.get_unit_counts(obs, units.Protoss.Probe) < 16:
                self.actions = PROTOSS_MACROS.Train_Probe()
            elif mineral > 125 and gas > 50 and food > 2 and len(self.get_all_complete_units_by_type(obs, units.Protoss.Gateway)) and len(self.get_all_complete_units_by_type(obs, units.Protoss.CyberneticsCore)) > 0:
                self.actions = PROTOSS_MACROS.Train_Stalker()
            elif len(self.get_all_complete_units_by_type(obs, units.Protoss.Assimilator)) > 0 and not gasHarvestersFull:
                self.actions = PROTOSS_MACROS.Collect_Gas()
            elif self.can_build():
                if food < 4 and mineral > 100:
                    self.actions = PROTOSS_MACROS.Build_Pylon()
                    self.last_build_frame = self.frame
                elif self.get_unit_counts(obs, units.Protoss.Probe) > 16 and len(self.get_all_complete_units_by_type(obs, units.Protoss.Assimilator)) < 2 and mineral > 75:
                    self.actions = PROTOSS_MACROS.Build_Assimilator()
                    self.last_build_frame = self.frame
                elif len(self.get_all_complete_units_by_type(obs, units.Protoss.Gateway)) > 0 and mineral > 150:
                    self.actions = PROTOSS_MACROS.Build_CyberneticsCore()
                    self.last_build_frame = self.frame
                elif mineral > 150 and self.get_unit_counts(obs, units.Protoss.Gateway) < 4:
                    self.actions = PROTOSS_MACROS.Build_Gateway()
                    self.last_build_frame = self.frame

            return FUNCTIONS.no_op()

        action, arg_func = self.actions[0]
        self.actions = self.actions[1:]
        if self.can_do(obs, action.id):
            action_args = arg_func(obs)
            return action(*action_args)

        return FUNCTIONS.no_op()

