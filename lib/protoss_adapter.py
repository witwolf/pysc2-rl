# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/12.
#

from pysc2.lib import units
import numpy as np
from lib.adapter import Adapter
from lib.adapter import DefaultObservationAdapter
from lib.protoss_macro import PROTOSS_MACROS
from lib.protoss_macro import _PROTOSS_UNITS_MACROS
from lib.protoss_macro import _PROTOSS_BUILDINGS_MACROS
from lib.protoss_macro import _PROTOSS_UNITS_DICT
from lib.protoss_macro import _PROTOSS_BUILDINGS_DICT


class ProtossRewardAdapter(Adapter):
    def __init__(self, config, memory_step_n=8):
        self._config = config
        self._memory_step_n = memory_step_n
        self._memory_step_obs = [None] * memory_step_n

    def get_feature_vector(self, timestep):
        return (timestep._unit_counts,
                timestep.observation.player.minerals,
                timestep.observation.player.vespene,
                timestep.observation.player.food_cap -
                timestep.observation.player.food_used)

    def get_reward(self, timestep, action):
        if not timestep._macro_success:
            return -1
        features, minerals, gas, food = self.get_feature_vector(timestep)
        food_cap = timestep._unit_counts.get(units.Protoss.Nexus, 0) * 15
        # add future pylon
        food_cap += timestep._unit_counts.get(units.Protoss.Pylon, 0) * 8
        food_cap += timestep.building_queues[_PROTOSS_BUILDINGS_DICT[units.Protoss.Pylon].id] * 8
        food_future = food_cap - timestep.observation.player.food_used

        if action.id == PROTOSS_MACROS.No_op:
            return -1
        # check unit requirements
        if action.id in _PROTOSS_UNITS_MACROS:
            unit = _PROTOSS_UNITS_MACROS[action.id]
            # if worker
            if unit.unit_type == units.Protoss.Probe:
                probe_num = timestep._unit_counts.get(units.Protoss.Probe, 0)
                probe_in_queue = \
                    len(timestep.training_queues[_PROTOSS_UNITS_DICT[units.Protoss.Probe].id])
                # max 16 probe
                if probe_num + probe_in_queue <= 16:
                    return 1
                else:
                    return -1
            # if zealot
            elif unit.unit_type == units.Protoss.Zealot:
                return 1
            return -1

        # check building requirements
        if action.id in _PROTOSS_BUILDINGS_MACROS:
            building = _PROTOSS_BUILDINGS_MACROS[action.id]
            # if need food
            if building.unit_type == units.Protoss.Pylon:
                pylon_num = timestep._unit_counts.get(units.Protoss.Pylon, 0)
                pylon_in_queue = timestep.building_queues[_PROTOSS_BUILDINGS_DICT[units.Protoss.Pylon].id]
                # if first pylon, give a big reward
                if pylon_num + pylon_in_queue >= 1:
                    return 10
                # if food urgent, give a big reward
                elif food_future <= 10 and timestep.observation.player.food_cap < 200:
                    return 10
                # if food cap, give neg reward
                elif timestep.observation.player.food_cap >= 200:
                    return -1
                # if too much food, give a neg reward
                elif food_future >= 24:
                    return -1
                # if much food and no gateway, give a neg reward
                elif food_future >= 12 and features.get(units.Protoss.Gateway, 0) == 0:
                    return -1
                else:
                    return 1
            elif building.unit_type == units.Protoss.Gateway:
                gateway_num = timestep._unit_counts.get(units.Protoss.Gateway, 0)
                gateway_in_queue = timestep.building_queues[_PROTOSS_BUILDINGS_DICT[units.Protoss.Gateway].id]
                if gateway_num + gateway_in_queue <= 4:
                    return 4 - gateway_num - gateway_in_queue + 0.5
                else:
                    return -1
            else:
                return -1

        if action.id == PROTOSS_MACROS.Callback_Idle_Workers:
            if timestep.observation.player.idle_worker_count == 0:
                return 10
            else:
                return -1

        if action.id == PROTOSS_MACROS.Attack_Enemy:
            # if not enough zealot, don't attack
            zealot_num = timestep._unit_counts.get(units.Protoss.Zealot, 0)
            if zealot_num <= 12:
                return -1
            else:
                return 0.1

        # other function return 0
        return 1e-2

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
        step_i = 0
        rewards = []
        game_end = False
        for timestep, action in zip(timesteps, actions):
            if timestep.last():
                game_end = True
            last_timestep = None
            if timestep._macro_success:
                if step_i < self._memory_step_n:
                    last_timestep = self._memory_step_obs[step_i]
                else:
                    last_timestep = timesteps[step_i - self._memory_step_n]
            rewards.append(self.get_reward(timestep, action) +
                           # self.get_damage(timestep, last_timestep) +
                           timestep.reward)
            step_i += 1
        if not game_end:
            self._memory_step_obs = timesteps[-self._memory_step_n:]
        else:
            self._memory_step_obs = [None] * self._memory_step_n
        return np.array(rewards)

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
