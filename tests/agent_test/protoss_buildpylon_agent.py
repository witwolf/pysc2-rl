import sys
import math
import numpy as np
sys.path.append('.')
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FUNCTIONS
from lib.protoss_macro import PROTOSS_MACROS
from tests.agent_test.protoss_base_agent import ProtossBaseAgent
from lib.protoss_adapter import ProtossRewardAdapter as RewardAdapter
from lib.protoss_timestep import ProtossTimeStepFactory as TimeStepFactory
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



class ProtossBuildPylonAgent(ProtossBaseAgent):
    def __init__(self):
        super(ProtossBuildPylonAgent, self).__init__()
        self.time = time.time()
        self.probe_count = 12
        self.probe_frame = 0
        self.allunits={}
        self._ALL_UNIT=[i for i in _PRO]
        self._ALL_UNIT.append(_POWER)
        self.timestep_factory = TimeStepFactory(None, True, 1)
    
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
        super(ProtossBuildPylonAgent, self).step(obs)
        for tmpUnit in self._ALL_UNIT:
            self.getTypespots(obs,tmpUnit)

        obs = self.timestep_factory.process(obs)
        #cprint(obs.to_feature())
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
            power_type=obs.observation['feature_screen'][_POWER_TYPE]
            power_map=(power_type == 1).nonzero()

            if idle_workers > 0:
                self.actions = PROTOSS_MACROS.Callback_Idle_Workers()
            elif mineral > 50 and food > 1 and self.get_unit_counts(obs, units.Protoss.Probe) < 16:
                print(PROTOSS_MACROS.Train_Probe, adapter.transform([obs],
                                                                    [PROTOSS_MACROS.Train_Probe]))
                self.actions = PROTOSS_MACROS.Train_Probe()
            elif self.can_build():
                print(PROTOSS_MACROS.Build_Pylon, adapter.transform([obs],
                                                                        [PROTOSS_MACROS.Build_Pylon]))
                self.actions = PROTOSS_MACROS.Build_Pylon()
                self.last_build_frame = self.frame
            return FUNCTIONS.no_op()
        action, arg_func = self.actions[0]
        self.actions = self.actions[1:]
        if self.can_do(obs, action.id):
            action_args = arg_func(obs)
            if not action_args:
                return FUNCTIONS.no_op()
            for action_arg in action_args:
                if not action_arg:
                    return FUNCTIONS.no_op()
            return action(*action_args)

        return FUNCTIONS.no_op()
