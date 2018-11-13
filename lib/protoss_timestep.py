# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/12.
#

import copy
from pysc2.lib.actions import FUNCTIONS
from lib.protoss_macro import _PROTOSS_BUILDINGS_FUNCTIONS
from lib.protoss_macro import _PROTOSS_UNITS_FUNCTIONS
from lib.protoss_macro import _PROTOSS_UNITS
from lib.protoss_macro import _PROTOSS_BUILDINGS
from lib.protoss_macro import _PROTOSS_UNITS_DICT
from lib.protoss_macro import _PROTOSS_BUILDINGS_DICT
from lib.protoss_macro import U


class ProtossTimeStep(object):
    def __init__(self, timestep, step_mul):
        self.macro_success = False
        self._feature_units = {}
        self._feature_unit_counts = {}
        self._raw_units = {}
        self._minimap_units = {}
        self._unit_counts = {}
        self.frame_pass = 0  # second pass
        self.self_bases = {}  # (location, worker assigned)
        self.enemy_bases = {}  # (location, worker assigned)
        self.neutral_bases = {}  # (location, true/false)
        self.self_units = [0 for _ in range(len(_PROTOSS_UNITS))]  # (type, count)
        self.enemy_units = {}  # (tag, type)
        self.self_buildings = [0 for _ in range(len(_PROTOSS_BUILDINGS))]  # (type, count)
        self.self_upgrades = {}  # (type, true/false)
        self.training_queues = [[] for _ in range(len(_PROTOSS_UNITS))]
        self.building_queues = [0 for _ in range(len(_PROTOSS_BUILDINGS))]  # (type, queued_count)
        self._timestep = timestep
        self._fill(step_mul)

    def is_unit(self, unit_type):
        return unit_type in _PROTOSS_UNITS_DICT

    def is_building(self, unit_type):
        return unit_type in _PROTOSS_BUILDINGS_DICT

    def is_upgrade(self, action_id):
        if action_id == FUNCTIONS.Research_Blink_quick.id:
            return "blink"
        if action_id == FUNCTIONS.Research_ProtossGroundArmorLevel1_quick.id:
            return "ground armor 1"
        if action_id == FUNCTIONS.Research_ProtossGroundWeaponsLevel1_quick.id:
            return "ground weapon 1"
        return None

    def _fill(self, step_mul):
        feature_units = self._timestep.observation.feature_units
        for unit in feature_units:
            unit_type = unit.unit_type
            if unit_type not in self._feature_units:
                self._feature_units[unit_type] = []
                self._feature_unit_counts[unit_type] = 0
            self._feature_units[unit_type].append(unit)
            self._feature_unit_counts[unit_type] += 1

        raw_units = self._timestep.observation.raw_units
        for unit in raw_units:
            if unit.alliance == 4 or unit.alliance == 0:
                continue
            if unit.alliance == 1 and int(unit.build_progress) != 100:
                continue
            unit_type = unit.unit_type
            if not unit_type in self._raw_units:
                self._raw_units[unit_type] = []
                self._minimap_units[unit_type] = []
                self._unit_counts[unit_type] = 0
            self._raw_units[unit_type].append(unit)
            minimap_unit = copy.deepcopy(unit)
            minimap_unit.x, minimap_unit.y = U._world_tl_to_minimap_px(unit)
            self._minimap_units[unit_type].append(minimap_unit)
            self._unit_counts[unit_type] += 1

        self.frame_pass += step_mul
        self.self_bases.clear()
        self.enemy_bases.clear()
        self.neutral_bases.clear()

        # check building queue
        for bid in range(len(_PROTOSS_BUILDINGS)):
            last_build_count = self.self_buildings[bid]
            # current build count
            build_count = len([unit for unit in self._timestep.observation.raw_units
                               if unit.unit_type == _PROTOSS_BUILDINGS[bid].unit_type and
                               unit.alliance == 1])
            if build_count > last_build_count:
                self.building_queues[bid] -= (build_count - last_build_count)
            if self.building_queues[bid] < 0:
                self.building_queues[bid] = 0
            self.self_buildings[bid] = 0

        # update training queue
        for uid in range(len(_PROTOSS_UNITS)):
            index = 0
            training_queue = self.training_queues[uid]
            while index < len(training_queue):
                training_queue[index] -= 1
                if training_queue[index] == 0:
                    del training_queue[index]
                else:
                    index += 1
            self.self_units[uid] = 0

        last_actions = self._timestep.observation.last_actions
        for last_action in last_actions:
            # update building queue
            if last_action in _PROTOSS_BUILDINGS_FUNCTIONS:
                bid = _PROTOSS_BUILDINGS_FUNCTIONS[last_action].id
                self.building_queues[bid] += 1
            # update training queue
            elif last_action in _PROTOSS_UNITS_FUNCTIONS:
                uid = _PROTOSS_UNITS_FUNCTIONS[last_action].id
                time = _PROTOSS_UNITS_FUNCTIONS[last_action].time
                self.training_queues[uid].append(time)
            # update upgrades
            upgrade = self.is_upgrade(last_action)
            if upgrade:
                self.self_upgrades[upgrade] = True

        # update unit/building count
        for unit in self._timestep.observation.raw_units:
            if unit.alliance == 1:
                # update unit count
                if self.is_unit(unit.unit_type):
                    self.self_units[_PROTOSS_UNITS_DICT[unit.unit_type].id] += 1
                # update building count
                elif self.is_building(unit.unit_type):
                    self.self_buildings[_PROTOSS_BUILDINGS_DICT[unit.unit_type].id] += 1

    def to_feature(self):
        self_bases = [(key, self.self_bases[key]) for key in sorted(self.self_bases.keys())]
        enemy_bases = [(key, self.enemy_bases[key]) for key in sorted(self.enemy_bases.keys())]
        neutral_bases = [(key, self.neutral_bases[key]) for key in sorted(self.neutral_bases.keys())]
        enemy_units = [(key, self.enemy_units[key]) for key in sorted(self.enemy_units.keys())
                       if self.enemy_units[key] > 0]
        self_upgrades = [(key, self.self_upgrades[key]) for key in sorted(self.self_upgrades.keys())]
        return [self.frame_pass] + self_bases + enemy_bases + neutral_bases + self.self_units + \
               enemy_units + self.self_buildings + self_upgrades + self.training_queues + \
               self.building_queues

    def __getattr__(self, item):
        return getattr(self._timestep, item)


class ProtossTimeStepFactory():
    def __init__(self, timestep, macro_success, step_mul):
        self._timestep = timestep
        self._step_mul = step_mul
        self.macro_success = macro_success

    def __getattr__(self, item):
        return getattr(self._timestep, item)

    def process(self, timestep):
        # TODO 1 , update infomation
        # TODO 2 , mixin timestep
        return ProtossTimeStep(timestep, self._step_mul)

    def reset(self):
        pass
