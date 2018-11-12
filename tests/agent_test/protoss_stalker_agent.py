# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/1.
#

import sys
import math
import numpy as np
sys.path.append('.')
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FUNCTIONS
from lib.protoss_macro import PROTOSS_MACROS
from tests.agent_test.protoss_base_agent import ProtossBaseAgent
from lib.protoss_adapter import ProtossRewardAdapter as RewardAdapter
from lib.protoss_adapter import ProtossInformationAdapter as InformationAdapter
from lib.config import Config
import time

_PRO=units.Protoss
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_POWER_TYPE=features.SCREEN_FEATURES.power.index
_WORKER=_PRO.Probe
_CENTER=_PRO.Nexus
_GATE=_PRO.Gateway
_PYLON=_PRO.Pylon
_POWER="PylonPower"
_ASSIM=_PRO.Assimilator
_TMP_BUILDS=[_CENTER,_GATE,_ASSIM,_PYLON]

class ProtossStalkerAgent(ProtossBaseAgent):
    def __init__(self):
        super(ProtossStalkerAgent, self).__init__()
        self.information = InformationAdapter(Config())
        self.time = time.time()
        self.probe_count = 12
        self.probe_frame = 0
        self.allunits={}
        self._ALL_UNIT=[i for i in _PRO]
        self._ALL_UNIT.append(_POWER)
    
    def getTypespots(self,obs,unitType):
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
        power_type=obs.observation['feature_screen'][_POWER_TYPE]
        if unitType!=_POWER:
            ys,xs=(unit_type == unitType).nonzero()
        else:
            ys,xs=(power_type == 1).nonzero()
        if len(list(ys))>0 and (unitType not in self.allunits.keys()):
            self.allunits[unitType]=[(list(xs),list(ys)),self._getPoint((xs,ys))]
            buildTmp=self.allunits[unitType]
            distances=np.array([np.linalg.norm(np.array(i)-np.array(buildTmp[1])) 
                for i in zip(list(xs),list(ys))])
            self.allunits[unitType].append(math.ceil(distances.max()))
        return list(ys),list(xs)
    def _getPoint(self,points):
        x=(points[0].max()+points[0].min())/2
        y=(points[1].max()+points[1].min())/2
        return (x,y)

    def step(self, obs):

        class TimestepWrapper:
            def __init__(self, timestep, success):
                self._timestep = timestep
                self.macro_success = success

            def __getattr__(self, item):
                if item == 'macro_success':
                    return self.macro_success
                return getattr(self._timestep, item)

        super(ProtossStalkerAgent, self).step(obs)

        self.information.update([obs])
        #print(self.information.transform([obs]))

        for tmpUnit in self._ALL_UNIT:
            self.getTypespots(obs,tmpUnit)

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
            if len(self.allunits)>0:
                for tmp in self.allunits.keys():
                    print(tmp," Center:",self.allunits[tmp][1]," Radius:",self.allunits[tmp][2])
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
            armyCount=obs.observation.player.army_count
            food = self.future_food(obs) - obs.observation.player.food_used
            idle_workers = obs.observation.player.idle_worker_count
            # if can build a building
            if armyCount>5:
                self.actions=PROTOSS_MACROS.Attack_Enemy()
                print(PROTOSS_MACROS.Attack_Enemy, adapter.transform([TimestepWrapper(obs, True)],
                                                                    [PROTOSS_MACROS.Attack_Enemy]))
            elif idle_workers > 0:
                self.actions = PROTOSS_MACROS.Callback_Idle_Workers()
            elif mineral > 50 and food > 1 and self.get_unit_counts(obs, units.Protoss.Probe) < 16:
                print(PROTOSS_MACROS.Train_Probe, adapter.transform([TimestepWrapper(obs, True)],
                                                                    [PROTOSS_MACROS.Train_Probe]))
                self.actions = PROTOSS_MACROS.Train_Probe()
            elif mineral > 125 and gas > 50 and food > 2 \
                    and len(self.get_all_complete_units_by_type(obs, units.Protoss.Gateway)) \
                    and len(self.get_all_complete_units_by_type(obs, units.Protoss.CyberneticsCore)) > 0:
                print(PROTOSS_MACROS.Train_Stalker, adapter.transform([TimestepWrapper(obs, True)],
                                                                      [PROTOSS_MACROS.Train_Stalker]))
                self.actions = PROTOSS_MACROS.Train_Zealot()
            elif len(self.get_all_complete_units_by_type(obs, units.Protoss.Assimilator)) > 0 and not gasHarvestersFull:
                print(PROTOSS_MACROS.Collect_Gas, adapter.transform([TimestepWrapper(obs, True)],
                                                                    [PROTOSS_MACROS.Collect_Gas]))
                self.actions = PROTOSS_MACROS.Collect_Gas()
            elif self.can_build():
                if food < 4 and mineral > 100:
                    print(PROTOSS_MACROS.Build_Pylon, adapter.transform([TimestepWrapper(obs, True)],
                                                                        [PROTOSS_MACROS.Build_Pylon]))
                    self.actions = PROTOSS_MACROS.Build_Pylon()
                    self.last_build_frame = self.frame
                elif self.get_unit_counts(obs, units.Protoss.Probe) > 16 and mineral > 75 \
                        and len(self.get_all_complete_units_by_type(obs, units.Protoss.Assimilator)) < 1:
                    print(PROTOSS_MACROS.Build_Assimilator, adapter.transform([TimestepWrapper(obs, True)],
                                                                              [PROTOSS_MACROS.Build_Assimilator]))
                    self.actions = PROTOSS_MACROS.Build_Assimilator()
                    self.last_build_frame = self.frame
                elif len(self.get_all_complete_units_by_type(obs, units.Protoss.Gateway)) > 0 and mineral > 150:
                    print(PROTOSS_MACROS.Build_CyberneticsCore, adapter.transform([TimestepWrapper(obs, True)],
                                                                                  [
                                                                                      PROTOSS_MACROS.Build_CyberneticsCore]))
                    self.actions = PROTOSS_MACROS.Build_CyberneticsCore()
                    self.last_build_frame = self.frame
                elif self.get_unit_counts(obs, units.Protoss.Probe) > 16 and mineral > 75 \
                        and len(self.get_all_complete_units_by_type(obs, units.Protoss.Assimilator)) < 2:
                    print(PROTOSS_MACROS.Build_Assimilator, adapter.transform([TimestepWrapper(obs, True)],
                                                                              [PROTOSS_MACROS.Build_Assimilator]))
                elif mineral > 150 and self.get_unit_counts(obs, units.Protoss.Gateway) < 1:
                    print(PROTOSS_MACROS.Build_Gateway, adapter.transform([TimestepWrapper(obs, True)],
                                                                          [PROTOSS_MACROS.Build_Gateway]))
                    self.actions = PROTOSS_MACROS.Build_Gateway()
                    self.last_build_frame = self.frame

            return FUNCTIONS.no_op()

        action, arg_func = self.actions[0]
        self.actions = self.actions[1:]
        if self.can_do(obs, action.id):
            action_args = arg_func(obs)
            return action(*action_args)

        return FUNCTIONS.no_op()
