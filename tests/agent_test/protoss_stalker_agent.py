# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/1.
#

import sys

sys.path.append('.')
from pysc2.lib import units
from pysc2.lib.actions import FUNCTIONS
from lib.protoss_macro import PROTOSS_MACROS
from tests.agent_test.protoss_base_agent import ProtossBaseAgent
from lib.protoss_adapter import ProtossRewardAdapter as RewardAdapter
from lib.protoss_timestep import ProtossTimeStepFactory as TimeStepFactory
from lib.config import Config
import time


class ProtossStalkerAgent(ProtossBaseAgent):
    def __init__(self):
        super(ProtossStalkerAgent, self).__init__()
        self.time = time.time()
        self.probe_count = 12
        self.probe_frame = 0
        self.timestep_factory = TimeStepFactory(None, True, 1)

    def step(self, obs):
        super(ProtossStalkerAgent, self).step(obs)

        obs = self.timestep_factory.process(obs)
        # print(obs.to_feature())
        # print(obs.observation.last_actions)
        # pylon_progress = [unit.build_progress for unit in obs.observation.raw_units
        #                   if unit.unit_type == units.Protoss.Stalker]
        # if len(pylon_progress) > 0:
        #     print(pylon_progress)

        # probe_count = len([unit for unit in obs.observation.raw_units if unit.unit_type == units.Protoss.Zealot])
        # if probe_count != self.probe_count:
        #     print("probe frame", self.frame - self.probe_frame)
        #     self.probe_frame = self.frame
        #     self.probe_count = probe_count

        if self.frame % self.frame_skip != 1:
            return FUNCTIONS.no_op()
        # find macro to execute
        if len(self.actions) == 0:
            config = Config()
            adapter = RewardAdapter(config, 1)

            assigned_gas_harvesters = [unit.assigned_harvesters for unit in obs.observation.feature_units
                                       if unit.unit_type == units.Protoss.Assimilator]
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
            elif mineral > 50 and food > 1 and self.get_unit_counts(obs, units.Protoss.Probe) < 16:
                # print(PROTOSS_MACROS.Train_Probe, adapter.transform([obs],
                #                                                     [PROTOSS_MACROS.Train_Probe]))
                self.actions = PROTOSS_MACROS.Train_Probe()
            elif mineral > 125 and gas > 50 and food > 2 \
                    and len(self.get_all_complete_units_by_type(obs, units.Protoss.Gateway)) \
                    and len(self.get_all_complete_units_by_type(obs, units.Protoss.CyberneticsCore)) > 0:
                # print(PROTOSS_MACROS.Train_Stalker, adapter.transform([obs],
                #                                                       [PROTOSS_MACROS.Train_Stalker]))
                self.actions = PROTOSS_MACROS.Train_Stalker()
            elif len(self.get_all_complete_units_by_type(obs, units.Protoss.Assimilator)) > 0 and not gasHarvestersFull:
                # print(PROTOSS_MACROS.Collect_Gas, adapter.transform([obs],
                #                                                     [PROTOSS_MACROS.Collect_Gas]))
                self.actions = PROTOSS_MACROS.Collect_Gas()
            elif self.can_build():
                if mineral > 40 * obs.observation.player.food_cap:
                    # print(PROTOSS_MACROS.Build_Pylon, adapter.transform([obs],
                    #                                                     [PROTOSS_MACROS.Build_Pylon]))
                    self.actions = PROTOSS_MACROS.Build_Pylon()
                    self.last_build_frame = self.frame
                elif self.get_unit_counts(obs, units.Protoss.Probe) > 16 and mineral > 75 \
                        and len(self.get_all_complete_units_by_type(obs, units.Protoss.Assimilator)) < 1:
                    # print(PROTOSS_MACROS.Build_Assimilator, adapter.transform([obs],
                    #                                                           [PROTOSS_MACROS.Build_Assimilator]))
                    self.actions = PROTOSS_MACROS.Build_Assimilator()
                    self.last_build_frame = self.frame
                elif len(self.get_all_complete_units_by_type(obs, units.Protoss.Gateway)) > 0 and mineral > 150:
                    # print(PROTOSS_MACROS.Build_CyberneticsCore, adapter.transform([obs],
                    #                                                               [PROTOSS_MACROS.Build_CyberneticsCore]))
                    self.actions = PROTOSS_MACROS.Build_CyberneticsCore()
                    self.last_build_frame = self.frame
                elif self.get_unit_counts(obs, units.Protoss.Probe) > 16 and mineral > 75 \
                        and len(self.get_all_complete_units_by_type(obs, units.Protoss.Assimilator)) < 2:
                    # print(PROTOSS_MACROS.Build_Assimilator, adapter.transform([obs],
                    #                                                           [PROTOSS_MACROS.Build_Assimilator]))
                    self.actions = PROTOSS_MACROS.Build_Assimilator()
                    self.last_build_frame = self.frame
                elif mineral > 150 and self.get_unit_counts(obs, units.Protoss.Gateway) < 1:
                    # print(PROTOSS_MACROS.Build_Gateway, adapter.transform([obs],
                    #                                                       [PROTOSS_MACROS.Build_Gateway]))
                    self.actions = PROTOSS_MACROS.Build_Gateway()
                    self.last_build_frame = self.frame

            return FUNCTIONS.no_op()

        action, arg_func = self.actions[0]
        self.actions = self.actions[1:]
        if self.can_do(obs, action.id):
            action_args = arg_func(obs)
            if not action_args:
                self.actions = []
                return FUNCTIONS.no_op()
            for action_arg in action_args:
                if not action_arg:
                    self.actions = []
                    return FUNCTIONS.no_op()
            return action(*action_args)

        return FUNCTIONS.no_op()
