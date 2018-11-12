# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/12.
#

from pysc2.lib import units
import numpy as np
from lib.adapter import Adapter
from lib.protoss_macro import PROTOSS_MACROS
from lib.protoss_unit import _PROTOSS_UNITS_MACROS
from lib.protoss_unit import _PROTOSS_BUILDINGS_MACROS


class DefaultActionRewardAdapter(Adapter):
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
            # if resource not enough
            if minerals < unit.minerals or gas < unit.gas or food < unit.food:
                return -1
            for requirement in unit.requirement_types:
                # if requirements miss
                if requirement not in features:
                    return -1
            return float(unit.minerals + unit.gas + unit.gas) / 1000.0

        # check building requirements
        if action.id in _PROTOSS_BUILDINGS_MACROS:
            building = _PROTOSS_BUILDINGS_MACROS[action.id]
            # if resource not enough
            if minerals < building.minerals or gas < building.gas:
                return -1
            for requirement in building.requirement_types:
                # if requirements miss
                if requirement not in features:
                    return -1
            # first building return 1
            if building.unit_type not in features:
                return 1
            # if need food
            elif building.unit_type == units.Protoss.Pylon and food < 2:
                return 1
            else:
                return np.exp(-features[building.unit_type])

        if action.id == PROTOSS_MACROS.Callback_Idle_Workers:
            full_harvesters = [
                unit.assigned_harvesters >= unit.ideal_harvesters
                for unit in timestep.observation.feature_units
                if unit.unit_type == units.Protoss.Nexus]
            # not enough mineral harvester, positive reward
            for full in full_harvesters:
                if not full:
                    return 1
            # full mineral harvester, negative reward
            return -1

        if action.id == PROTOSS_MACROS.Collect_Gas:
            full_harvesters = [unit.assigned_harvesters >= unit.ideal_harvesters
                               for unit in timestep.observation.feature_units
                               if unit.unit_type == units.Protoss.Assimilator]
            # not enough mineral harvester, positive reward
            for full in full_harvesters:
                if not full:
                    return 1
            # full mineral harvester, negative reward
            return -1

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


class DefaultProtossMacroAdapter(Adapter):
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
