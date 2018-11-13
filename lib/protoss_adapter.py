# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/12.
#

from pysc2.lib import units
from pysc2.lib.actions import FUNCTIONS
import numpy as np
from lib.adapter import Adapter
from lib.adapter import DefaultObservationAdapter
from lib.protoss_macro import PROTOSS_MACROS
from lib.protoss_macro import _PROTOSS_UNITS_MACROS
from lib.protoss_macro import _PROTOSS_BUILDINGS_MACROS
from lib.protoss_macro import _PROTOSS_BUILDINGS_FUNCTIONS
from lib.protoss_macro import _PROTOSS_UNITS_FUNCTIONS
from lib.protoss_macro import _PROTOSS_UNITS
from lib.protoss_macro import _PROTOSS_BUILDINGS
from lib.protoss_macro import _PROTOSS_UNITS_DICT
from lib.protoss_macro import _PROTOSS_BUILDINGS_DICT


class ProtossRewardAdapter(Adapter):
    def __init__(self, config, memory_step_n=8):
        self._config = config
        self._memory_step_n = memory_step_n
        self._memory_step_obs = [None] * memory_step_n

    def get_feature_vector(self, timestep):
        features = {}
        for unit in timestep.observation.raw_units:
            if unit.alliance == 1:
                if unit.unit_type not in features:
                    features[unit.unit_type] = 1
                else:
                    features[unit.unit_type] += 1
        return (features,
                timestep.observation.player.minerals,
                timestep.observation.player.vespene,
                timestep.observation.player.food_cap -
                timestep.observation.player.food_used)

    def get_reward(self, timestep, action):
        if not timestep.macro_success:
            return 0
        features, minerals, gas, food = self.get_feature_vector(timestep)
        # check unit requirements
        if action.id in _PROTOSS_UNITS_MACROS:
            unit = _PROTOSS_UNITS_MACROS[action.id]
            for requirement in unit.requirement_types:
                # if requirements miss
                if requirement not in features:
                    return -1
            # if worker
            if unit.unit_type == units.Protoss.Probe:
                lack_harvesters = [unit.ideal_harvesters - unit.assigned_harvesters
                                    for unit in timestep.observation.raw_units
                                    if unit.unit_type == units.Protoss.Nexus]
                lack_harvesters = sum(lack_harvesters) if len(lack_harvesters) > 0 else 0
                nexus_num = timestep.information._self_units[_PROTOSS_BUILDINGS_DICT[units.Protoss.Nexus].id]
                probe_in_queue =\
                    len(timestep.information._training_queues[_PROTOSS_UNITS_DICT[units.Protoss.Probe].id])
                return (lack_harvesters - probe_in_queue) / nexus_num
            return float(unit.minerals + unit.gas + unit.gas) / 1000.0

        # check building requirements
        if action.id in _PROTOSS_BUILDINGS_MACROS:
            building = _PROTOSS_BUILDINGS_MACROS[action.id]
            for requirement in building.requirement_types:
                # if requirements miss
                if requirement not in features:
                    return -1
            # first building return 1
            if building.unit_type not in features:
                return 1
            # if need food
            elif building.unit_type == units.Protoss.Pylon:
                # if food urgent, give a big reward
                if food < 2:
                    return 10
                # if enough food, give no reward
                elif food > 8:
                    return 0
                else:
                    return 1
            else:
                return np.exp(-features[building.unit_type])

        if action.id == PROTOSS_MACROS.Callback_Idle_Workers:
            if timestep.observation.player.idle_worker_count == 0:
                return 1
            else:
                return -1

        if action.id == PROTOSS_MACROS.Collect_Gas:
            if timestep.information._self_units[_PROTOSS_BUILDINGS_DICT[units.Protoss.Assimilator].id] == 0:
                return -1
            return 1

        # other function return 0
        return 0

    def get_damage(self, timestep, last_timestep):
        if not last_timestep:
            return 0
        # self health
        self_hp = np.sum([
            unit.health + unit.shield for unit
            in timestep.observation.raw_units
            if unit.alliance == 1])
        self_last_hp = np.sum([
            unit.health + unit.shield
            for unit in last_timestep.observation.raw_units
            if unit.alliance == 1])
        # enemy health
        enemy_hp = np.sum([
            unit.health + unit.shield
            for unit in timestep.observation.raw_units
            if unit.alliance == 4])
        enemy_last_hp = np.sum([
            unit.health + unit.shield
            for unit in last_timestep.observation.raw_units
            if unit.alliance == 4])
        # health delta
        self_hp_delta = self_hp - self_last_hp
        enemy_hp_delta = enemy_hp - enemy_last_hp
        return np.clip(self_hp_delta - enemy_hp_delta, -5, 5)

    def transform(self, timesteps, actions):
        '''
        :param timesteps:
        :param actions:
        :return:
        '''
        step_i = 0
        rewards = []
        game_end = False
        for timestep, action in zip(timesteps, actions):
            if timestep.last():
                game_end = True
            last_timestep = None
            if timestep.macro_success:
                if step_i < self._memory_step_n:
                    last_timestep = self._memory_step_obs[step_i]
                else:
                    last_timestep = timesteps[step_i - self._memory_step_n]
            rewards.append(self.get_reward(timestep, action) +
                           self.get_damage(timestep, last_timestep) +
                           timestep.reward)
            step_i += 1

        if not game_end:
            self._memory_step_obs = timesteps[-self._memory_step_n:]
        else:
            self._memory_step_obs = [None] * self._memory_step_n
        return np.array(rewards)

    def reverse(self, *args, **kwargs):
        raise NotImplementedError()


class ProtossInformationAdapter(Adapter):
    def __init__(self, config):
        self._config = config
        self._frame_pass = 0  # second pass
        self._self_bases = {}  # (location, worker assigned)
        self._enemy_bases = {}  # (location, worker assigned)
        self._neutral_bases = {}  # (location, true/false)
        self._self_units = [0 for _ in range(len(_PROTOSS_UNITS))]  # (type, count)
        self._enemy_units = {}  # (tag, type)
        self._self_buildings = [0 for _ in range(len(_PROTOSS_BUILDINGS))]  # (type, count)
        self._self_upgrades = {}  # (type, true/false)
        self._training_queues = [[] for _ in range(len(_PROTOSS_UNITS))]
        self._building_queues = [0 for _ in range(len(_PROTOSS_BUILDINGS))]  # (type, queued_count)

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

    def update(self, timestep, step_mul):
        self._frame_pass += step_mul
        self._self_bases.clear()
        self._enemy_bases.clear()
        self._neutral_bases.clear()

        # check building queue
        for bid in range(len(_PROTOSS_BUILDINGS)):
            last_build_count = self._self_buildings[bid]
            # current build count
            build_count = len([unit for unit in timestep.observation.raw_units
                               if unit.unit_type == _PROTOSS_BUILDINGS[bid].unit_type and
                               unit.alliance == 1])
            if build_count > last_build_count:
                self._building_queues[bid] -= (build_count - last_build_count)
            if self._building_queues[bid] < 0:
                self._building_queues[bid] = 0
            self._self_buildings[bid] = 0

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
            self._self_units[uid] = 0

        last_actions = timestep.observation.last_actions
        for last_action in last_actions:
            # update building queue
            if last_action in _PROTOSS_BUILDINGS_FUNCTIONS:
                bid = _PROTOSS_BUILDINGS_FUNCTIONS[last_action].id
                self._building_queues[bid] += 1
            # update training queue
            elif last_action in _PROTOSS_UNITS_FUNCTIONS:
                uid = _PROTOSS_UNITS_FUNCTIONS[last_action].id
                time = _PROTOSS_UNITS_FUNCTIONS[last_action].time
                self._training_queues[uid].append(time)
            # update upgrades
            upgrade = self.is_upgrade(last_action)
            if upgrade:
                self._self_upgrades[upgrade] = True

        # update unit/building count
        for unit in timestep.observation.raw_units:
            if unit.alliance == 1:
                # update unit count
                if self.is_unit(unit.unit_type):
                    self._self_units[_PROTOSS_UNITS_DICT[unit.unit_type].id] += 1
                # update building count
                elif self.is_building(unit.unit_type):
                    self._self_buildings[_PROTOSS_BUILDINGS_DICT[unit.unit_type].id] += 1

    def transform(self):
        self_bases = [(key, self._self_bases[key]) for key in sorted(self._self_bases.keys())]
        enemy_bases = [(key, self._enemy_bases[key]) for key in sorted(self._enemy_bases.keys())]
        neutral_bases = [(key, self._neutral_bases[key]) for key in sorted(self._neutral_bases.keys())]
        enemy_units = [(key, self._enemy_units[key]) for key in sorted(self._enemy_units.keys())
                       if self._enemy_units[key] > 0]
        self_upgrades = [(key, self._self_upgrades[key]) for key in sorted(self._self_upgrades.keys())]
        return [self._frame_pass] + self_bases + enemy_bases + neutral_bases + self._self_units + \
               enemy_units + self._self_buildings + self_upgrades + self._training_queues + \
               self._building_queues

    def reverse(self, *args, **kwargs):
        raise NotImplementedError()


class ProtossObservationAdapter(DefaultObservationAdapter):

    def _available_actions(self, timestep):
        action_indexes = self._config._action_indexes
        feature = np.zeros(len(PROTOSS_MACROS))
        for i in range(len(PROTOSS_MACROS)):
            cond = PROTOSS_MACROS[i].cond
            if cond(timestep):
                feature[i] = 1
        return feature[action_indexes]


class ProtossMacroAdapter(Adapter):
    def __init__(self, config):
        self._config = config

    def transform(self, macros):
        values = []

        action_index_table = \
            self._config._action_index_table

        length = len(macros)
        for dim, _ in self._config.policy_dims:
            values.append(np.zeros(shape=(length), dtype=np.int32))
        for i, macro in enumerate(macros):
            macro_id = action_index_table[macro.id]
            values[0][i] = macro_id

    def reverse(self, actions):
        action_indexes = self._config._action_indexes
        macros = []
        for action in actions[0]:
            macro_id = action_indexes[action]
            macros.append(PROTOSS_MACROS[macro_id]())
        return macros
