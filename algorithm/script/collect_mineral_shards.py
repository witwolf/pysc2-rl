# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#

import numpy as np
from pysc2.lib import units
from pysc2.agents import base_agent
from pysc2.lib.actions import FUNCTIONS

from . import get_units_by_type, select_unit, get_distances


class CollectMineralShards(base_agent.BaseAgent):
    def __init__(self,*args):
        super().__init__()
        self.minerals = None
        self.pairs = None
        self.reset()

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)

    def reset(self):
        super().reset()
        self.minerals = set()
        self.pairs = dict()

    def step(self, obs, *args):
        if obs.last():
            self.reset()
        marines = get_units_by_type(obs.observation, units.Terran.Marine)
        minerals = get_units_by_type(obs.observation, 1680)
        self.minerals = set(m[0]['tag'] for m in minerals)

        idle_marines = []
        for marine_tag, mineral_tag in self.pairs.items():
            if mineral_tag not in self.minerals:
                idle_marines.append(marine_tag)
        for marine in idle_marines:
            self.pairs.pop(marine)

        avaliable_minerals = []
        for m in minerals:
            if m[0]['tag'] not in set(self.pairs.values()):
                avaliable_minerals.append(m)

        return self._get_actions(marines, avaliable_minerals)

    def _get_actions(self, marines, minerals):
        if not minerals:
            return FUNCTIONS.no_op()

        selected_marine = None
        for m in marines:
            if m[1]['is_selected']:
                selected_marine = m
                break

        if selected_marine:
            marine_tag = selected_marine[0]['tag']
            if marine_tag not in self.pairs:
                distances = get_distances(selected_marine, minerals)
                mineral = minerals[np.argmin(distances)]
                self.pairs[marine_tag] = mineral[0]['tag']
                return self._move_marine_to_minarel(selected_marine, mineral)

        in_working_marines = set(self.pairs.keys())
        idle_marines = []
        for marine in marines:
            if marine[0]['tag'] not in in_working_marines:
                idle_marines.append(marine)

        if not idle_marines:
            return FUNCTIONS.no_op()

        return select_unit(idle_marines[0])

    def _move_marine_to_minarel(self, marine, mineral):
        # todo , move to closet position randomly
        spatial_action = FUNCTIONS.Move_screen("now", (mineral[1].x, mineral[1].y))
        return spatial_action
