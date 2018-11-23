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
from lib.protoss_macro import Pylon, Probe, Gateway, CyberneticsCore


class ProtossRewardAdapter(Adapter):
    def __init__(self, config, memory_step_n=8):
        self._config = config

    def transform(self, timesteps, actions):
        rewards = []
        for timestep, action in zip(timesteps, actions):
            rewards.append(self.get_reward(timestep, action) + 1000 *
                           timestep.reward)
        return np.array(rewards)

    def reverse(self, *args, **kwargs):
        raise NotImplementedError()

    def get_reward(self, timestep, action):
        # macro execute failed
        if not timestep._macro_success:
            return -2
        # no op
        if action.id == PROTOSS_MACROS.No_op:
            return -0.125
        # callback idle workers
        if action.id == PROTOSS_MACROS.Callback_Idle_Workers:
            return 0.02
        # collect gas
        if action.id == PROTOSS_MACROS.Collect_Gas:
            return 0.01
        # train units macros
        if action.id in _PROTOSS_UNITS_MACROS:
            return self._unit_reward(timestep, action)
        # build building macros
        if action.id in _PROTOSS_BUILDINGS_MACROS:
            return self._building_reward(timestep, action)
        if action.id == PROTOSS_MACROS.Attack_Enemy:
            unit_counts = timestep._unit_counts
            zealot_num = unit_counts.get(units.Protoss.Zealot, 0)
            stalker_num = unit_counts.get(units.Protoss.Stalker, 0)
            army_food = zealot_num * 1 + stalker_num * 2
            return -2 if army_food <= 6 else 0.125
        return 1e-2

    def _unit_reward(self, timestep, action):
        unit_counts = timestep._unit_counts
        training_queue = timestep.training_queues
        unit = _PROTOSS_UNITS_MACROS[action.id]
        if unit.unit_type == units.Protoss.Probe:
            probe_num = unit_counts.get(units.Protoss.Probe, 0)
            probe_in_queue = training_queue[Probe.id]
            reward = 1 if probe_num + probe_in_queue <= 16 else -2
            return reward
        elif unit.unit_type == units.Protoss.Zealot:
            return 2
        elif unit.unit_type == units.Protoss.Stalker:
            return 3
        return -2

    def _building_reward(self, timestep, action):
        unit_counts = timestep._unit_counts
        building_queue = timestep.building_queues
        building = _PROTOSS_BUILDINGS_MACROS[action.id]
        gateway_num = unit_counts.get(units.Protoss.Gateway, 0)
        gateway_in_queue = building_queue[Gateway.id]
        gateway_num += gateway_in_queue

        if building.unit_type == units.Protoss.Pylon:
            pylon_num = unit_counts.get(units.Protoss.Pylon, 0)
            pylon_in_queue = building_queue[Pylon.id]
            pylon_num += pylon_in_queue

            food_cap = timestep.observation.player.food_cap
            food_cap += pylon_in_queue * 8
            food_future = food_cap - timestep.observation.player.food_used

            # if first pylon, give a big reward
            if pylon_num == 1:
                return 4
            # if food urgent, give a big reward
            elif food_future <= 10 and timestep.observation.player.food_cap < 200:
                return 4
            # if food cap, give neg reward
            elif timestep.observation.player.food_cap >= 200:
                return -8
            # if too much food, give a neg reward
            elif food_future >= 32:
                return -4
            # if much food and no gateway, give a neg reward
            elif food_future >= 12 and gateway_num == 0:
                return -2
            return 1
        elif building.unit_type == units.Protoss.Gateway:
            if gateway_num <= 5:
                return 2 ** (3 - gateway_num)
            return -2
        elif building.unit_type == units.Protoss.Assimilator:
            return 4
        elif building.unit_type == units.Protoss.CyberneticsCore:
            cybernetics_num = unit_counts.get(units.Protoss.CyberneticsCore, 0)
            cybernetics_in_queue = building_queue[CyberneticsCore.id]
            cybernetics_num += cybernetics_in_queue
            if cybernetics_num <= 2:
                return 3 ** (3 - cybernetics_num)
            return -3
        else:
            return -2

    def _damage_reward(self, timestep, last_timestep):
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


class ProtossObservationAdapter(DefaultObservationAdapter):

    def _available_actions(self, timestep):
        action_indexes = self._config._action_indexes
        feature = np.zeros(len(PROTOSS_MACROS))
        for i in range(len(PROTOSS_MACROS)):
            cond = PROTOSS_MACROS[i].cond
            if cond(timestep):
                feature[i] = 1
        return feature[action_indexes]

    def _nonspatial_feature(self, timestep, field):
        f = getattr(timestep, field, None)
        if not f:
            return timestep.observation[field]
        return f


class ProtossMacroAdapter(Adapter):
    def __init__(self, config):
        self._config = config

    def transform(self, macros):
        values = []
        action_index_table = self._config._action_index_table
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
