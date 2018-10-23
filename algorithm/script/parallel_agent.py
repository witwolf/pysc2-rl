# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/19.
#

from pysc2.lib.run_parallel import RunParallel
from algorithm.script.collect_mineral_shards import CollectMineralShards
from algorithm.script.move_to_beacon import MoveToBeacon


def script_agent_maker(map_name):
    script_cls = {
        'MoveToBeacon': MoveToBeacon,
        'CollectMineralShards': CollectMineralShards
    }

    def agent_maker(*args):
        return script_cls[map_name]

    return agent_maker


class ParallelAgent(object):
    def __init__(self, agents=None,
                 agent_num=None,
                 agent_args=None,
                 agent_makers=None):
        super().__init__()
        self._parallel = RunParallel()
        self._agents = agents
        if agents is not None:
            self._agents = agents
            self._agent_num = len(agents)
        else:
            self._agent_num = agent_num
            if not isinstance(agent_makers, list):
                agent_makers = [agent_makers] * agent_num
            if not isinstance(agent_args, list):
                agent_args = [agent_args] * agent_num
            self._agents = self._parallel.run(
                (func, args) for func, args in zip(agent_makers, agent_args))

    def setup(self, obs_specs, action_specs):
        for agent, obs_spec, action_spec in zip(
                self._agents, obs_specs, action_specs):
            agent.setup(obs_spec, action_spec)

    def reset(self):
        for agent in self._agents:
            agent.reset()

    def step(self, obs):
        return self._parallel.run(
            (a.step, o) for a, o in zip(self._agents, obs))

    def close(self):
        self._parallel.shutdown()
