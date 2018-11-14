import sys
import math
import numpy as np
sys.path.append('.')
import random
import numpy as np
from absl import app
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import enum
import math
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
_POWER=features.SCREEN_FEATURES.power.index
_WORKER=_PRO.Probe
_CENTER=_PRO.Nexus
_GATE=_PRO.Gateway
_PYLON=_PRO.Pylon
_ASSIM=_PRO.Assimilator
_TMP_BUILDS=[_CENTER,_GATE,_ASSIM,_PYLON]

class UnitSize(enum.IntEnum):
    '''units' radius'''
    Nexus=10
    Pylon=4
    Assimilator=6
    Gateway=7
#Nexus:10,Pylon:4,Assimilator:6,Gateway:7

class TestAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TestAgent,self).__init__()
        self.pylonlocate=(40,41)
        self.attack_coordinates = None
        self.action_args = []
        self.actions = []
        self.overlord_eggs = 0
        self.buildings={}
        self.selected=False
    def getTypespots(self,obs,unitType):
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
        ys,xs=(unit_type == unitType).nonzero()
        if len(list(ys))>0:
            self.buildings[unitType]=[list(zip(list(xs),list(ys))),self._getPoint((xs,ys))]
            buildTmp=self.buildings[unitType]
            distances=np.array([np.linalg.norm(np.array(i)-np.array(buildTmp[1])) 
                    for i in zip(list(xs),list(ys))])
            self.buildings[unitType].append(math.ceil(distances.max()))
        return list(ys),list(xs)
    def _getPoint(self,points):
        x=(points[0].max()+points[0].min())/2
        y=(points[1].max()+points[1].min())/2
        return (x,y)
    def getUnitsByType(self,obs,unitType):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unitType]
    def getPowerSize(self,obs):
        power_feature=obs.observation['feature_screen'][_POWER]
        ys,xs=(power_feature == 1).nonzero()
        unitType="PylonPower"
        if len(ys)>0:
            self.buildings[unitType]=[list(zip(list(xs),list(ys))),self._getPoint((xs,ys))]
            buildTmp=self.buildings[unitType]
            distances=np.array([np.linalg.norm(np.array(i)-np.array(buildTmp[1]))
                    for i in zip(list(xs),list(ys))])
            self.buildings[unitType].append(math.ceil(distances.max()))
        return ys,xs
    def step(self,obs):
        super(TestAgent, self).step(obs)
        if len(obs.observation.last_actions)>0:
            print("Last action:")
            print(obs.observation.last_actions)
        pl_ys,pl_xs=self.getTypespots(obs,_PYLON)
        pw_ys,pw_xs=self.getPowerSize(obs)
        '''
        if len(pl_ys)<=0:
            print("Pylon haven't been build")
        else:
            if len(pw_ys)>0:
                print("Pylon have been build")
                print("Power_X:",list(pw_xs));print("Power_y:",list(pw_ys))
            else:
                print("Pylon is been build")
        '''
        #yTmp=list(ys);xTmp=list(xs)
        #for i in range(len(ys)):
        #    print(power_feature[yTmp[i]+1][xTmp[i]+1])
        for tmpType in _TMP_BUILDS:
            y,x=self.getTypespots(obs,tmpType)
            if tmpType==_PYLON and len(y)>0:
                node=np.array(list(zip(x,y))[0]);centerpot=np.array(self.buildings[_PYLON][1])
                pylonnode=np.array(self.pylonlocate)
                #print("LeftTop:");print(node)
                #print("Center:");print(centerpot)
                #print("Setting Location:");print(pylonnode)
                '''
                if np.linalg.norm(node-pylonnode)>np.linalg.norm(centerpot-pylonnode):
                    print("Build on Center")
                else:
                    print("Build on Lefttop")
                '''
        #for tmp in self.buildings.keys():
        #    print(tmp,"Center:",self.buildings[tmp][1],"Radius:",self.buildings[tmp][2])
        if not self.selected:
            workers=self.getUnitsByType(obs,_WORKER)
            if len(workers)>0:
                worker=random.choice(workers)
            else:
                print("no workers")
                return actions.FUNCTIONS.no_op()
            action=actions.FUNCTIONS.select_point
            self.selected=True
            print("select worker")
            return action("select",(worker.x,worker.y))
        else:
            action,arg_func=PROTOSS_MACROS.Attack_Enemy()[0]
            action_args = arg_func(obs)
            return action(*action_args)
def main(unused_argv):
  agent =TestAgent()
  try:
    while True:
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
  except KeyboardInterrupt:
    pass

if __name__ == "__main__":
    app.run(main)
