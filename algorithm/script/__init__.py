# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#

import math
import numpy as np

from pysc2.lib.actions import FUNCTIONS


def get_units_by_type(obs, unit_type):
    unit_pairs = []
    for raw_unit, feature_unit in zip(obs.raw_units, obs.feature_units):
        if feature_unit.unit_type == unit_type:
            unit_pairs.append((raw_unit, feature_unit))
    return unit_pairs


def select_unit(uint):
    # todo , select_rect with random rect size
    select_action = FUNCTIONS.select_point("select", (uint[1].x, uint[1].y))
    return select_action


def get_distances(unit, units):
    def get_distance(unit0, unit1):
        return math.sqrt(math.pow(unit1[1].x - unit0[1].x, 2)
                         + math.pow(unit1[1].y - unit0[1].y, 2))

    distances = np.ndarray(shape=[len(units)], dtype=np.float32)
    for i, m in enumerate(units):
        distances[i] = get_distance(unit, m)
    return distances
