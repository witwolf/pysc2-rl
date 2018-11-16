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
from pysc2.lib import features, units


class ProtossTimeStep(object):
    def __new__(cls, *args, **kwargs):
        instance = super(ProtossTimeStep, cls).__new__(cls)
        if len(args) == 1:
            instance._timestep = args[0]
        else:
            instance._timestep = None
        instance._macro_success = False
        instance._feature_units = {}
        instance._feature_units_completed = {}
        instance._feature_unit_counts = {}
        instance._feature_unit_completed_counts = {}
        instance._raw_units = {}
        instance._minimap_units = {}
        instance._unit_counts = {}
        instance._raw_units_completed = {}
        instance._minimap_units_completed = {}
        instance._unit_completed_counts = {}
        instance._mixin = {}
        return instance

    def fill(self):
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
            if int(unit.build_progress) != 100:
                continue
            if unit_type not in self._feature_units_completed:
                self._feature_units_completed[unit_type] = []
                self._feature_unit_completed_counts[unit_type] = 0
            self._feature_units_completed[unit_type].append(unit)
            self._feature_unit_completed_counts[unit_type] += 1

        # update raw units count info
        raw_units = self._timestep.observation.raw_units
        for unit in raw_units:
            if unit.alliance != 1:
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
            # completed units
            if int(unit.build_progress) != 100:
                continue
            if not unit_type in self._raw_units_completed:
                self._raw_units_completed[unit_type] = []
                self._minimap_units_completed[unit_type] = []
                self._unit_completed_counts[unit_type] = 0
            self._raw_units_completed[unit_type].append(unit)
            self._minimap_units_completed[unit_type].append(minimap_unit)
            self._unit_completed_counts[unit_type] += 1

    def update(self, **mixin):
        self._mixin.update(mixin)

    def __getattr__(self, item):
        value = self._mixin.get(item, None)
        if value is not None: return value
        return getattr(self._timestep, item)

    def print(self):
        print('_macro_success:', self._macro_success,
              '_feature_units:', self._feature_units,
              '_minimap_units:', self._minimap_units,
              'mixin:', self._mixin)


class ProtossTimeStepFactory():
    def __init__(self, step_mul):
        self._step_mul = step_mul
        self._frames = 0
        self._buildings = None
        self._training_queues = None
        self._building_queues = None
        self._power_map = None
        self._neutral_map = None
        self._neutral_center = None
        self._power_list = None
        self._not_power_list = None
        self.reset()

    def update(self, timestep):
        self._frames += self._step_mul
        timestep = ProtossTimeStep(timestep)
        timestep.fill()

        if len(self._neutral_map) == 0:
            player_relative = timestep.observation.feature_screen.player_relative
            neutral = features.PlayerRelative.NEUTRAL
            ys, xs = (player_relative == neutral). \
                nonzero()
            for pt in zip(xs, ys):
                self._neutral_map.add(pt)
                self._neutral_center[0] += pt[0]
                self._neutral_center[1] += pt[1]
            if len(self._neutral_map) > 0:
                self._neutral_center[0] /= len(self._neutral_map)
                self._neutral_center[1] /= len(self._neutral_map)

        first_base_loc = None
        if len(timestep._feature_units.get(units.Protoss.Nexus, [])) > 0:
            first_base = timestep._feature_units[units.Protoss.Nexus][0]
            first_base_loc = (first_base.x, first_base.y)

        building_queues = self._building_queues
        unit_counts = timestep._unit_counts
        self_units = [0 for _ in range(len(_PROTOSS_UNITS))]
        for bid in range(len(_PROTOSS_BUILDINGS)):
            if building_queues[bid] == 0:
                continue
            # last build count
            last_build_count = self._buildings[bid]
            # current build count
            build_count = unit_counts.get(_PROTOSS_BUILDINGS[bid].unit_type, 0)
            # building changed, recaculate power
            if last_build_count != build_count:
                print("update feature by", bid)
                self._power_list.clear()
                self._not_power_list.clear()

                xs, ys = (timestep.observation.feature_screen.power == 1).nonzero()
                power_center = [0, 0]
                for pt in zip(xs, ys):
                    if pt not in self._neutral_map:
                        dist = 12
                        if first_base_loc:
                            dist = U.get_distance(first_base_loc, pt)
                        if dist < 11:
                            self._power_list.append(pt)
                            power_center[0] += pt[0]
                            power_center[1] += pt[1]

                if len(self._power_list) > 0:
                    power_center[0] /= len(self._power_list)
                    power_center[1] /= len(self._power_list)

                self._power_map = set(self._power_list)

                screen_w = timestep.observation.feature_screen.shape[-1]
                screen_h = timestep.observation.feature_screen.shape[-2]
                for w in range(0, screen_w):
                    for h in range(0, screen_h):
                        pt = (w, h)
                        if pt not in self._power_map and pt not in self._neutral_map:
                            dist = 12
                            if first_base_loc:
                                dist = U.get_distance(first_base_loc, pt)
                            if dist < 11:
                                self._not_power_list.append(pt)
                self._not_power_list.sort(
                    key=lambda p: U.get_distance(power_center, p) - U.get_distance(self._neutral_center, p))

            if build_count > last_build_count:
                building_queues[bid] -= (build_count - last_build_count)
            if building_queues[bid] < 0:
                building_queues[bid] = 0
            self._buildings[bid] = build_count

        # update training queue
        for uid in range(len(_PROTOSS_UNITS)):
            index = 0
            training_queue = self._training_queues[uid]
            while index < len(training_queue):
                training_queue[index] -= 1
                if training_queue[index] == 0:
                    del training_queue[index]
                else:
                    index += 1
                    self_units[uid] = unit_counts.get(
                        _PROTOSS_UNITS[uid].unit_type, 0)

        last_actions = timestep.observation.last_actions
        upgrades = {}
        upgrade_map = {
            FUNCTIONS.Research_Blink_quick.id: 0,
            FUNCTIONS.Research_ProtossGroundArmorLevel1_quick.id: 1,
            FUNCTIONS.Research_ProtossGroundWeaponsLevel1_quick.id: 2
        }
        for last_action in last_actions:
            if last_action in _PROTOSS_BUILDINGS_FUNCTIONS:
                bid = _PROTOSS_BUILDINGS_FUNCTIONS[last_action].id
                self._building_queues[bid] += 1
            elif last_action in _PROTOSS_UNITS_FUNCTIONS:
                uid = _PROTOSS_UNITS_FUNCTIONS[last_action].id
                time = _PROTOSS_UNITS_FUNCTIONS[last_action].time
                self._training_queues[uid].append(time)
            if last_action in upgrade_map:
                upgrades[upgrade_map[last_action]] = True

        timestep.update(
            frame=self._frames,
            building_queues=self._building_queues,
            training_queues=self._training_queues,
            self_units=self_units,
            power_map=self._power_map,
            neutral_map=self._neutral_map,
            power_list=self._power_list,
            not_power_list=self._not_power_list,
            upgrades=upgrades)

        return timestep

    def reset(self):
        self._frames = 0  # second pass
        self._buildings = [0 for _ in range(len(_PROTOSS_BUILDINGS))]
        self._training_queues = [[] for _ in range(len(_PROTOSS_UNITS))]
        self._building_queues = [0 for _ in range(len(_PROTOSS_BUILDINGS))]
        self._power_map = set()
        self._neutral_map = set()
        self._neutral_center = [0, 0]
        self._power_list = []
        self._not_power_list = []
