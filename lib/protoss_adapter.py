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
        return (timestep._feature_unit_counts,
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
                protos_nexus = timestep._raw_units.get(units.Protoss.Nexus, [])
                lack_harvesters = [unit.ideal_harvesters - unit.assigned_harvesters
                                   for unit in protos_nexus]
                lack_harvesters = sum(lack_harvesters) if len(lack_harvesters) > 0 else 0
                nexus_num = timestep._unit_counts.get(_PROTOSS_BUILDINGS_DICT[units.Protoss.Nexus].id, 0)
                if nexus_num == 0:
                    return -1
                probe_in_queue = \
                    len(timestep.training_queues[_PROTOSS_UNITS_DICT[units.Protoss.Probe].id])
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
            if timestep._unit_counts.get(_PROTOSS_BUILDINGS_DICT[units.Protoss.Assimilator].id, 0) == 0:
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
