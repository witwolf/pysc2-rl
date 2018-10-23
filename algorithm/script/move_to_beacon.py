# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#
import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import units
from pysc2.lib.actions import FUNCTIONS

from . import get_units_by_type, select_unit, get_distances


class MoveToBeacon(base_agent.BaseAgent):

    def __init__(self, *args):
        super().__init__()
        self.target = None
        self.reset()

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)

    def reset(self):
        super().reset()
        self.target = None

    def step(self, obs, *args):
        if obs.last():
            self.reset()
        marine = get_units_by_type(obs.observation, units.Terran.Marine)[0]
        beacons = get_units_by_type(obs.observation, 317)
        targets = set((b[1]['x'], b[1]['y']) for b in beacons)

        if not targets:
            return FUNCTIONS.no_op()

        if self.target not in targets:
            self.target = None

        if not marine[1]['is_selected']:
            return select_unit(marine)

        if self.target:
            return FUNCTIONS.no_op()

        distance = get_distances(marine, beacons)
        beacon = beacons[np.argmin(distance)]
        self.target = (beacon[1]['x'], beacon[1]['y'])

        return FUNCTIONS.Move_screen("now", (beacon[1].x, beacon[1].y))
