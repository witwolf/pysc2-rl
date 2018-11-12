import random
import numpy as np
from absl import app
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import enum
import math

_PRO=units.Protoss
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_WORKER=_PRO.Probe
_CENTER=_PRO.Nexus
_GATE=_PRO.Gateway
_PYLON=_PRO.Pylon
_ASSIM=_PRO.Assimilator
_TMP_BUILDS=[_CENTER,_GATE,_ASSIM,_PYLON]

class unitSize(enum.IntEnum):
    Nexus=5
    Pylon=2
    Assimilator=3

class TestAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TestAgent,self).__init__()
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
            self.buildings[unitType]=[(list(xs),list(ys)),self._getPoint((xs,ys))]
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
    def _judgeLocat(self,base,justy):
        if base>justy:
            return 2
        else:
            return base+random.randint(12,justy-base-12)
    def getLocation(self,obs):
        centers=self.getUnitsByType(obs,_CENTER)
        if len(centers)>0:
            centerTmp=random.choice(centers)
            x=self._judgeLocat(centerTmp.x,self.W/2)
            y=self._judgeLocat(centerTmp.y,self.H/2)
            return (x,y)
        else:
            return (None,None)
    def step(self,obs):
        self.H=obs.observation.feature_screen.shape[-2]
        self.W=obs.observation.feature_screen.shape[-1]
        super(TestAgent, self).step(obs)
        for tmpType in _TMP_BUILDS:
            y,x=self.getTypespots(obs,tmpType)
        print(self.buildings)
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
            if len(self.getUnitsByType(obs,_PYLON))>0:
                print("Have Pylon")
                action=actions.FUNCTIONS.Build_Gateway_screen
                (x,y)=tuple(np.array(self.buildings[_PYLON][1])+np.array([3,0]))
            else:
                print("lack of pylon")
                action=actions.FUNCTIONS.Build_Pylon_screen
                (x,y)=tuple(np.array(self.buildings[_CENTER][1])+np.array([11,0]))
            #(x,y)=self.getLocation(obs)
            if x:
                self.selected=False
                return action("now",(int(x),int(y)))
            else:
                print("Get location faild!")
                return actions.FUNCTIONS.no_op()


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
    '''
    if len(y)>0:
        if tmpType==_PRO.Pylon:
            print("_PRO.Pylon:")
            print("X:")
            print(x)
            print("Y:")
            print(y)
    '''

