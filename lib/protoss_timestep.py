# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/12.
#

import copy
from pysc2.lib.actions import FUNCTIONS
from lib.protoss_macro import _PROTOSS_BUILDINGS_FUNCTIONS
from lib.protoss_macro import _PROTOSS_UNITS_FUNCTIONS
from lib.protoss_macro import _PROTOSS_UNITS
from lib.protoss_macro import _PROTOSS_BUILDINGS
from lib.protoss_macro import U


class ProtossTimeStep(object):
    def __init__(self, timestep, timestep_factory):
        self._timestep = timestep
        self._factory = timestep_factory
        self.macro_success = False
        self._feature_units = {}
        self._feature_units_completed = {}
        self._feature_unit_counts = {}
        self._feature_unit_completed_counts = {}
        self._raw_units = {}
        self._minimap_units = {}
        self._unit_counts = {}
        self._raw_units_completed = {}
        self._minimap_units_completed = {}
        self._unit_completed_counts = {}
        self.self_units = [0 for _ in range(len(_PROTOSS_UNITS))]  # (type, count)
        self.self_upgrades = {}  # (type, true/false)
        self._fill()

    def is_upgrade(self, action_id):
        if action_id == FUNCTIONS.Research_Blink_quick.id:
            return "blink"
        if action_id == FUNCTIONS.Research_ProtossGroundArmorLevel1_quick.id:
            return "ground armor 1"
        if action_id == FUNCTIONS.Research_ProtossGroundWeaponsLevel1_quick.id:
            return "ground weapon 1"
        return None

    def _fill(self):
        # update feature units count info
        feature_units = self._timestep.observation.feature_units
        for unit in feature_units:
            unit_type = unit.unit_type
            if unit_type not in self._feature_units:
                self._feature_units[unit_type] = []
                self._feature_unit_counts[unit_type] = 0
            self._feature_units[unit_type].append(unit)
            self._feature_unit_counts[unit_type] += 1
            # completed units
            if int(unit.build_progress) == 100:
                if unit_type not in self._feature_units_completed:
                    self._feature_units_completed[unit_type] = []
                    self._feature_unit_completed_counts[unit_type] = 0
                self._feature_units_completed[unit_type].append(unit)
                self._feature_unit_completed_counts[unit_type] += 1

        # update raw units count info
        raw_units = self._timestep.observation.raw_units
        for unit in raw_units:
            if unit.alliance == 1:
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
                # completed units
                if int(unit.build_progress) == 100:
                    if not unit_type in self._raw_units_completed:
                        self._raw_units_completed[unit_type] = []
                        self._minimap_units_completed[unit_type] = []
                        self._unit_completed_counts[unit_type] = 0
                    self._raw_units_completed[unit_type].append(unit)
                    self._minimap_units_completed[unit_type].append(minimap_unit)
                    self._unit_completed_counts[unit_type] += 1

        self.self_bases.clear()
        self.enemy_bases.clear()
        self.neutral_bases.clear()

        building_queues = self._factory.building_queues
        for bid in range(len(_PROTOSS_BUILDINGS)):
            if building_queues[bid] == 0:
                continue
            # last build count
            last_build_count = self._factory.self_buildings[bid]
            # current build count
            build_count = self._unit_counts.get(_PROTOSS_BUILDINGS[bid].unit_type, 0)
            if build_count > last_build_count:
                building_queues[bid] -= (build_count - last_build_count)
            if building_queues[bid] < 0:
                building_queues[bid] = 0
            self._factory.self_buildings[bid] = build_count

        # update training queue
        for uid in range(len(_PROTOSS_UNITS)):
            index = 0
            training_queue = self._factory.training_queues[uid]
            while index < len(training_queue):
                training_queue[index] -= 1
                if training_queue[index] == 0:
                    del training_queue[index]
                else:
                    index += 1
            self.self_units[uid] = self._unit_counts.get(_PROTOSS_UNITS[uid].unit_type, 0)

        last_actions = self._timestep.observation.last_actions
        for last_action in last_actions:
            # update building queue
            if last_action in _PROTOSS_BUILDINGS_FUNCTIONS:
                bid = _PROTOSS_BUILDINGS_FUNCTIONS[last_action].id
                self._factory.building_queues[bid] += 1
            # update training queue
            elif last_action in _PROTOSS_UNITS_FUNCTIONS:
                uid = _PROTOSS_UNITS_FUNCTIONS[last_action].id
                time = _PROTOSS_UNITS_FUNCTIONS[last_action].time
                self._factory.training_queues[uid].append(time)
            # update upgrades
            upgrade = self.is_upgrade(last_action)
            if upgrade:
                self.self_upgrades[upgrade] = True

    def to_feature(self):
        self_bases = [(key, self._factory.self_bases[key]) for key in sorted(self._factory.self_bases.keys())]
        enemy_bases = [(key, self._factory.enemy_bases[key]) for key in sorted(self._factory.enemy_bases.keys())]
        neutral_bases = [(key, self._factory.neutral_bases[key]) for key in sorted(self._factory.neutral_bases.keys())]
        enemy_units = [(key, self._factory.enemy_units[key]) for key in sorted(self._factory.enemy_units.keys())
                       if self._factory.enemy_units[key] > 0]
        self_upgrades = [(key, self.self_upgrades[key]) for key in sorted(self.self_upgrades.keys())]
        return [self._factory.frame_pass] + self_bases + enemy_bases + neutral_bases + self.self_units + \
               enemy_units + self._factory.self_buildings + self_upgrades + self._factory.training_queues + \
               self._factory.building_queues

    def __getattr__(self, item):
        if hasattr(self._factory, item):
            return getattr(self._factory, item)
        return getattr(self._timestep, item)


class ProtossTimeStepFactory():
    def __init__(self, timestep, macro_success, step_mul):
        self._timestep = timestep
        self._step_mul = step_mul
        self.macro_success = macro_success
        self.frame_pass = 0  # second pass
        self.self_bases = {}  # (location, worker assigned)
        self.enemy_bases = {}  # (location, worker assigned)
        self.neutral_bases = {}  # (location, true/false)
        self.enemy_units = {}  # (tag, type)
        self.self_buildings = [0 for _ in range(len(_PROTOSS_BUILDINGS))]  # (type, count)
        self.training_queues = [[] for _ in range(len(_PROTOSS_UNITS))]
        self.building_queues = [0 for _ in range(len(_PROTOSS_BUILDINGS))]  # (type, queued_count)

    def __getattr__(self, item):
        return getattr(self._timestep, item)

    def process(self, timestep):
        # TODO 1 , update infomation
        # TODO 2 , mixin timestep
        self.frame_pass += self._step_mul
        return ProtossTimeStep(timestep, self)

    def reset(self):
        self.frame_pass = 0  # second pass
        self.self_bases = {}  # (location, worker assigned)
        self.enemy_bases = {}  # (location, worker assigned)
        self.neutral_bases = {}  # (location, true/false)
        self.enemy_units = {}  # (tag, type)
        self.self_buildings = [0 for _ in range(len(_PROTOSS_BUILDINGS))]  # (type, count)
        self.training_queues = [[] for _ in range(len(_PROTOSS_UNITS))]
        self.building_queues = [0 for _ in range(len(_PROTOSS_BUILDINGS))]  # (type, queued_count)
