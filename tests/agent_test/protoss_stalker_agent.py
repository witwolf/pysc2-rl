# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/1.
#

import sys
import math
import numpy as np
sys.path.append('.')
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FUNCTIONS
from lib.protoss_macro import PROTOSS_MACROS,U
from tests.agent_test.protoss_base_agent import ProtossBaseAgent
from lib.protoss_adapter import ProtossRewardAdapter as RewardAdapter
from lib.protoss_timestep import ProtossTimeStepFactory as TimeStepFactory
from lib.config import Config
import time

_PRO=units.Protoss
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_POWER_TYPE=features.SCREEN_FEATURES.power.index
_CAMERA = features.MINIMAP_FEATURES.camera.index
_WORKER=_PRO.Probe
_CENTER=_PRO.Nexus
_GATE=_PRO.Gateway
_PYLON=_PRO.Pylon
_POWER="PylonPower"
_ASSIM=_PRO.Assimilator
_TMP_BUILDS=[_CENTER,_GATE,_ASSIM,_PYLON]
_MINERAL=units.Neutral.MineralField

class ProtossStalkerAgent(ProtossBaseAgent):
    def __init__(self):
        super(ProtossStalkerAgent, self).__init__()
        self.time = time.time()
        self.probe_count = 12
        self.probe_frame = 0
        self.allunits={}
        self._ALL_UNIT=[i for i in _PRO]
        self._ALL_UNIT.append(_POWER)
        self.timestep_factory = TimeStepFactory(1)
    
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

        self.timestep_factory = TimeStepFactory(1)

    def step(self, obs):

        super(ProtossStalkerAgent, self).step(obs)

        #self.information.update([obs])
        #print(self.information.transform([obs]))

        for tmpUnit in self._ALL_UNIT:
            self.getTypespots(obs,tmpUnit)

        #print(obs.to_feature())

        obs = self.timestep_factory.update(obs)
        #print(obs.timestep_information)
        #print(len(obs.power_list))
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
            if len(self.allunits)>0:
                for tmp in self.allunits.keys():
                    print(tmp," Center:",self.allunits[tmp][1]," Radius:",self.allunits[tmp][2])
            config = Config()
            adapter = RewardAdapter(config, 1)
            minerals = U.get_other_mineral_location(obs)
            #all_unit=obs.observation.raw_units
            #minerals_location=[(tmp_unit.x,tmp_unit.y) for tmp_unit in all_unit 
            #        if tmp_unit.unit_type == _MINERAL]
            print("mineral locations:",minerals)

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

            power_type=obs.observation['feature_screen'][_POWER_TYPE]
            power_map=(power_type == 1).nonzero()

            training_probe = obs.training_queues[0]
            probe = obs._unit_counts.get(units.Protoss.Probe, 0)

            # if can build a building
            if armyCount>5:
                self.actions=PROTOSS_MACROS.Attack_Enemy()
                print(PROTOSS_MACROS.Attack_Enemy, adapter.transform([obs],
                                                                    [PROTOSS_MACROS.Attack_Enemy]))
            elif idle_workers > 0:
                self.actions = PROTOSS_MACROS.Callback_Idle_Workers()
            elif mineral > 50 and food > 1 and training_probe + probe < 16:
                self.actions = PROTOSS_MACROS.Train_Probe()
            elif mineral > 100 and food > 2 \
                    and len(self.get_all_complete_units_by_type(obs, units.Protoss.Gateway)):
                self.actions = PROTOSS_MACROS.Train_Zealot()
            elif mineral > 125 and gas > 50 and food > 2 \
                     and len(self.get_all_complete_units_by_type(obs, units.Protoss.Gateway)) \
                     and len(self.get_all_complete_units_by_type(obs, units.Protoss.CyberneticsCore)) > 0:
                 self.actions = PROTOSS_MACROS.Train_Stalker()
            elif len(self.get_all_complete_units_by_type(obs, units.Protoss.Assimilator)) > 0 and not gasHarvestersFull:
                 self.actions = PROTOSS_MACROS.Collect_Gas()
            elif self.can_build():
                if mineral > 40 * obs.observation.player.food_cap:
                    self.actions = PROTOSS_MACROS.Build_Pylon()
                    self.last_build_frame = self.frame
                elif self.get_unit_counts(obs, units.Protoss.Probe) > 15 and mineral > 75 \
                         and len(self.get_all_complete_units_by_type(obs, units.Protoss.Assimilator)) < 1:
                             self.actions = PROTOSS_MACROS.Build_Assimilator()
                             self.last_build_frame = self.frame
                elif len(self.get_all_complete_units_by_type(obs, units.Protoss.Gateway)) > \
                         0 and mineral > 150:
                             self.actions = PROTOSS_MACROS.Build_CyberneticsCore()
                             self.last_build_frame = self.frame
                elif self.get_unit_counts(obs, units.Protoss.Probe) > 15 and mineral > 75 \
                         and len(self.get_all_complete_units_by_type(obs, units.Protoss.Assimilator)) < 2:
                             self.actions = PROTOSS_MACROS.Build_Assimilator()
                             self.last_build_frame = self.frame
                elif mineral > 150 and self.get_unit_counts(obs, units.Protoss.Gateway) < 1:
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
            if action == FUNCTIONS.move_camera:
                print("Move Camera")
                print(action_args)
            return action(*action_args)
        return FUNCTIONS.no_op()
