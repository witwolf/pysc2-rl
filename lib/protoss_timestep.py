# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/12.
#

import copy
import numpy as np
from pysc2.lib.actions import FUNCTIONS
from lib.protoss_macro import _PROTOSS_BUILDINGS_FUNCTIONS
from lib.protoss_macro import _PROTOSS_UNITS
from lib.protoss_macro import _PROTOSS_BUILDINGS
from lib.protoss_macro import _PROTOSS_UNITS_MACROS
from lib.protoss_macro import _PROTOSS_BUILDINGS_MACROS
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
        instance._last_arg = None
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
        self._building_queues = None
        self.reset()

    def update(self, timestep):
        self._frames += self._step_mul
        timestep = ProtossTimeStep(timestep)
        timestep.fill()

        # neutral information
        neutral_map = set()
        neutral_center = [0, 0]
        player_relative = timestep.observation.feature_screen.player_relative
        neutral = features.PlayerRelative.NEUTRAL
        ys, xs = (player_relative == neutral).nonzero()
        for x, y in zip(xs, ys):
            neutral_map.add((x, y))
        neutral_center[0] = np.mean(xs) if neutral_map else 0
        neutral_center[1] = np.mean(ys) if neutral_map else 0

        # base information
        base_location = None
        if len(timestep._feature_units.get(units.Protoss.Nexus, [])) > 0:
            base = timestep._feature_units[units.Protoss.Nexus][0]
            base_location = (base.x, base.y)

        # power information
        power_list = []
        not_power_list = []
        power_center = [0, 0]
        screen_w = timestep.observation.feature_screen.shape[-1]
        screen_h = timestep.observation.feature_screen.shape[-2]
        xs, ys = (timestep.observation.feature_screen.power == 1).nonzero()
        if base_location:
            for pt in zip(xs, ys):
                if pt not in neutral_map:
                    if U.get_distance(base_location, pt) < 11:
                        power_list.append(pt)
        if power_list:
            power_center[0] = np.mean(pt[0] for pt in power_list)
            power_center[1] = np.mean(pt[1] for pt in power_list)
        power_map = set(power_list)
        if base_location:
            for w in range(0, screen_w):
                for h in range(0, screen_h):
                    pt = (w, h)
                    if pt not in power_map and pt not in neutral_map:
                        if U.get_distance(base_location, pt) < 11:
                            not_power_list.append(pt)
        not_power_list.sort(
            key=lambda p: U.get_distance(power_center, p) -
                          U.get_distance(neutral_center, p))

        # building information
        building_queues = self._building_queues
        unit_counts = timestep._unit_counts
        unit_completed_counts = timestep._unit_completed_counts
        self_units = [0 for _ in range(len(_PROTOSS_UNITS))]
        for bid in range(len(_PROTOSS_BUILDINGS)):
            if building_queues[bid] == 0:
                continue
            last_build_count = self._buildings[bid]
            unit_type = _PROTOSS_BUILDINGS[bid].unit_type
            build_count = unit_completed_counts.get(unit_type, 0)
            if build_count > last_build_count:
                building_queues[bid] -= (build_count - last_build_count)
            if building_queues[bid] < 0:
                building_queues[bid] = 0
            self._buildings[bid] = build_count
        last_actions = timestep.observation.last_actions
        for last_action in last_actions:
            if last_action in _PROTOSS_BUILDINGS_FUNCTIONS:
                bid = _PROTOSS_BUILDINGS_FUNCTIONS[last_action].id
                self._building_queues[bid] += 1

        # training information
        training_queues = [0 for _ in range(len(_PROTOSS_UNITS))]
        for uid in range(len(_PROTOSS_UNITS)):
            training_queues[uid] = 0
            self_units[uid] = unit_counts.get(
                _PROTOSS_UNITS[uid].unit_type, 0)
        for orders in timestep.observation.units_orders:
            for order in orders:
                if order.ability_id == 916:
                    training_queues[1] += 1
                elif order.ability_id == 1006:
                    training_queues[0] += 1

        feature_vetor = self.to_feature(timestep, training_queues)
        timestep.update(
            frame=self._frames,
            building_queues=self._building_queues,
            training_queues=training_queues,
            self_units=self_units,
            power_map=power_map,
            neutral_map=neutral_map,
            power_list=power_list,
            not_power_list=not_power_list,
            timestep_information=feature_vetor)

        return timestep

    def reset(self):
        self._frames = 0
        self._buildings = [0 for _ in range(len(_PROTOSS_BUILDINGS))]
        self._building_queues = [0 for _ in range(len(_PROTOSS_BUILDINGS))]

    def to_feature(self, timestep, training_queues):
        # training units
        training_units = [
            training_queues[i] / 20 for i in
            range(len(_PROTOSS_UNITS_MACROS))]
        # complete units
        completed_counts = timestep._unit_completed_counts
        complete_units = [
            completed_counts.get(_PROTOSS_UNITS[i].unit_type, 0) / 20
            for i in range(len(_PROTOSS_UNITS_MACROS))]
        # total units
        total_units = [
            tp[0] + tp[1] for tp in
            zip(training_units, complete_units)]
        units_vector = training_units + complete_units + total_units
        # queued buildings
        queued_buildings = [
            self._building_queues[i] / 20 for i in
            range(len(_PROTOSS_BUILDINGS_MACROS))]
        # exists buildings
        unit_counts = timestep._unit_counts
        exists_buildings = [
            unit_counts.get(_PROTOSS_BUILDINGS[i].unit_type, 0) / 20
            for i in range(len(_PROTOSS_BUILDINGS_MACROS))]
        # total buildings
        total_buildings = [
            tp[0] + tp[1] for tp in
            zip(queued_buildings, exists_buildings)]
        buildings_vector = queued_buildings + exists_buildings + total_buildings

        player = timestep.observation.player
        resource_vector = [
            player.minerals / 1000,
            player.vespene / 1000,
            player.food_cap / 40,
            player.food_used / 40,
            (player.food_cap - player.food_used) / 10]

        return units_vector + buildings_vector + resource_vector
