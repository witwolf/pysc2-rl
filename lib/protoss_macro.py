# Copyright (c) 2018 horizon robotics. All rights reserved.


import enum
import collections
import numbers
import numpy as np
import logging
import itertools

from random import shuffle
from numpy.random import randint
from pysc2.lib import features
from pysc2.lib import transform
from pysc2.lib import point
from lib.protoss_unit import *
from pysc2.lib.actions import FUNCTIONS

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index


class UnitSize(enum.IntEnum):
    '''units' radius'''
    Nexus = 8
    Pylon = 3
    Assimilator = 5
    Gateway = 5
    CyberneticsCore = 5
    PylonPower = 18
    Stalker = 2


_PROTOSS_BUILDINGS_SIZE = {
    units.Protoss.Nexus: UnitSize.Nexus.value,
    units.Protoss.Pylon: UnitSize.Pylon.value,
    units.Protoss.Assimilator: UnitSize.Assimilator.value,
    units.Protoss.Gateway: UnitSize.Gateway.value,
    units.Protoss.CyberneticsCore: UnitSize.CyberneticsCore.value,
}


class U(object):
    @staticmethod
    def screen_size(obs):
        feature_screen = obs.observation.feature_screen
        screen_shape = feature_screen.shape
        h = screen_shape[-2]
        w = screen_shape[-1]
        return w, h

    @staticmethod
    def minimap_size(obs):
        feature_minimap = obs.observation.feature_minimap
        minimap_shape = feature_minimap.shape
        h = minimap_shape[-2]
        w = minimap_shape[-1]
        return w, h

    @staticmethod
    def base_minimap_location(obs):
        player_relative = \
            obs.observation.feature_minimap.player_relative
        player_self = features.PlayerRelative.SELF
        player_y, player_x = (
                player_relative == player_self).nonzero()
        if player_y.mean() > 31 and player_x.mean() > 31:
            x, y = 40, 46
        else:
            x, y = 19, 24
        return x, y

    @staticmethod
    def enemy_minimap_location(obs):
        base_x, base_y = U.base_minimap_location(obs)
        (x, y) = (40, 46) if base_x < 31 else (19, 24)
        return x, y

    @staticmethod
    def enemy_second_minimap_location(obs):
        enemy_loc = U.enemy_minimap_location(obs)
        if enemy_loc[0] < 31:
            return 20, 47
        else:
            return 39, 23

    @staticmethod
    def worker_type(obs):
        return units.Protoss.Probe

    @staticmethod
    def geyser_type(obs):
        return units.Protoss.Assimilator

    @staticmethod
    def locations_by_type(obs, unit_type):
        _feature_units = obs._feature_units.get(unit_type, [])
        locations = [(unit.x, unit.y) for unit in _feature_units]
        return locations

    @staticmethod
    def rand_unit_location(obs, unit_type, _filter=None):
        feature_units = obs._feature_units_completed.get(unit_type, [])
        if _filter:
            feature_units = list(filter(_filter, feature_units))
        if (len(feature_units)) == 0:
            logging.warning("No units, unit_type:%d" % unit_type)
            return 0, 0
        unit = feature_units[randint(0, len(feature_units))]
        x, y = unit.x, unit.y
        return U._valid_screen_x_y(x, y, obs)

    @staticmethod
    def minimap_units(obs, unit_type, _filter=None):
        minimap_units = obs._minimap_units_completed.get(unit_type, [])
        if _filter:
            minimap_units = list(filter(_filter, minimap_units))
        return minimap_units

    @staticmethod
    def rand_minimap_unit_location(obs, unit_type, _filter=None):
        minimap_units = U.minimap_units(obs, unit_type, _filter)
        if len(minimap_units) == 0:
            logging.warning("No raw units, unit_type:%d", unit_type)
            return None
        minimap_unit = minimap_units[randint(0, len(minimap_units))]
        return minimap_unit.x, minimap_unit.y

    @staticmethod
    def bad_worker_location(obs):
        assimilator_type = units.Protoss.Assimilator
        _feature_units = obs._feature_units.get(assimilator_type, [])
        overload_geysers = [
            (unit.x, unit.y) for unit in _feature_units
            if unit.assigned_harvesters >= 3]
        if len(overload_geysers) > 0:
            geyser_x, geyser_y = overload_geysers[0]
            probes = U.locations_by_type(
                units.Protoss.Probe)
            dists = np.ndarray(shape=[len(probes)], dtype=np.float32)
            for i, (x, y) in zip(range(len(probes)), probes):
                dists[i] = abs(x - geyser_x) + abs(y - geyser_y)
            x, y = probes[np.argmin(dists)]
            return U._valid_screen_x_y(x, y, obs)
        return None

    @staticmethod
    def get_distance(pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    @staticmethod
    def get_distance_list(pos, pos_list):
        p1 = np.array(pos)
        return [np.linalg.norm(p1 - np.array(p2)) for p2 in pos_list]

    @staticmethod
    def _rect_points(l, t, r, b, max_r, max_t):
        points = []
        for x in range(max(0, l), min(r, max_r)):
            if t < max_t:
                points.append((x, t))
            if b >= 0:
                points.append((x, b))
        for y in range(max(0, b), min(t, max_t)):
            if l >= 0:
                points.append((l, y))
            if r < max_r:
                points.append((r, y))
        shuffle(points)
        return points

    @staticmethod
    def _find_place_xy(radius, ava_places, obs):
        w, h = U.screen_size(obs)
        nexus = U.locations_by_type(obs, units.Protoss.Nexus)
        base_x, base_y = w // 2, h // 2
        if len(nexus) > 0:
            base_x, base_y = nexus[0]
        side = 2 * radius
        l = base_x - UnitSize.Nexus - side
        t = base_y + UnitSize.Nexus
        r = base_x + UnitSize.Nexus
        b = base_y - UnitSize.Nexus - side
        max_r, max_t = w - side, h - side
        height_map = np.array(obs.observation.feature_screen.height_map)
        while l >= 0 or t < max_t or r < max_r or b >= 0:
            for left, bottom in U._rect_points(l, t, r, b, max_r, max_t):
                pos = left + radius, bottom + radius
                player_relative = \
                    obs.observation.feature_screen.player_relative
                neutral = features.PlayerRelative.NEUTRAL
                neutral_ys, neutral_xs = (player_relative == neutral).nonzero()
                if len(neutral_ys):
                    neutral_pos = np.mean(neutral_xs), np.mean(neutral_ys)
                    neutral_dist = U.get_distance(pos, neutral_pos)
                    if neutral_dist <= 2 * radius:
                        continue
                xs, ys = [], []
                for x, y in itertools.product(
                        range(left, left + side), range(bottom, bottom + side)):
                    xs.append(x)
                    ys.append(y)
                height = height_map[(ys, xs)]
                if np.min(height) == np.max(height) == 255:
                    if np.all(ava_places[(ys, xs)]):
                        return pos
            l -= 1
            t += 1
            r += 1
            b -= 1

        return None

    @staticmethod
    def new_pylon_location(obs):
        radius = UnitSize.Pylon
        place = U._find_place_xy(radius, obs.not_power, obs)
        if not place:
            logging.warning("power full")
            place = U._find_place_xy(radius, obs.not_building, obs)
        if not place:
            logging.warning("no space to place a pylon")
        return place

    @staticmethod
    def new_gateway_location(obs):
        radius = UnitSize.Gateway
        place = U._find_place_xy(radius, obs.power, obs)
        if not place:
            logging.warning("no space to place a gateway")
        return place

    @staticmethod
    def new_assimilator_location(obs):
        vesps = U.locations_by_type(obs, units.Neutral.VespeneGeyser)
        assimilators = U.locations_by_type(obs, units.Protoss.Assimilator)
        valid_vespenes = list(set(vesps) - set(assimilators))
        if len(valid_vespenes) == 0:
            return None
        x, y = valid_vespenes[randint(0, len(valid_vespenes))]
        return U._valid_screen_x_y(x, y, obs)

    @staticmethod
    def new_cyberneticscore_location(obs):
        radius = UnitSize.CyberneticsCore
        place = U._find_place_xy(radius, obs.power, obs)
        if not place:
            logging.warning("no space to place a cyberneticscore")
        return place

    @staticmethod
    def mineral_location(obs):
        minerals = U.locations_by_type(obs, units.Neutral.MineralField)
        minerals += U.locations_by_type(obs, units.Neutral.MineralField750)
        # if no mineral, go to center
        if len(minerals) == 0:
            return None
        x, y = minerals[randint(0, len(minerals))]
        return U._valid_screen_x_y(x, y, obs)

    @staticmethod
    def gas_location(obs):
        assimilator_type = units.Protoss.Assimilator
        assimilators = obs._feature_units_completed.get(assimilator_type, [])
        for assimilator in assimilators:
            if assimilator.assigned_harvesters < 3:
                return U._valid_screen_x_y(
                    assimilator.x, assimilator.y, obs)
        # if no mineral, go to center
        return None

    @staticmethod
    def screen_top(obs):
        screen_w, screen_h = U.screen_size(obs)
        return screen_w / 2, 0

    @staticmethod
    def screen_topright(obs):
        screen_w, screen_h = U.screen_size(obs)
        return screen_w - 1, 0

    @staticmethod
    def screen_right(obs):
        screen_w, screen_h = U.screen_size(obs)
        return screen_w - 1, screen_h / 2

    @staticmethod
    def screen_bottomright(obs):
        screen_w, screen_h = U.screen_size(obs)
        return screen_w - 1, screen_h - 1

    @staticmethod
    def screen_bottom(obs):
        screen_w, screen_h = U.screen_size(obs)
        return screen_w / 2, screen_h - 1

    @staticmethod
    def screen_bottomleft(obs):
        screen_w, screen_h = U.screen_size(obs)
        return 0, screen_h - 1

    @staticmethod
    def screen_left(obs):
        screen_w, screen_h = U.screen_size(obs)
        return 0, screen_h / 2

    @staticmethod
    def screen_topleft(obs):
        return 0, 0

    @staticmethod
    def army_minimap_location(obs):
        player_relative = \
            obs.observation.feature_minimap.player_relative
        player_self = features.PlayerRelative.SELF
        ys, xs = \
            (player_relative == player_self).nonzero()
        if len(xs) == 0:
            return None
        enemy_x, enemy_y = U.enemy_minimap_location(obs)
        dists = np.ndarray(shape=[len(xs)], dtype=np.float32)
        for (i, x, y) in zip(range(len(xs)), xs, ys):
            dists[i] = abs(x - enemy_x) + abs(y - enemy_y)
        pos = np.argmin(dists)
        x, y = xs[pos], ys[pos]
        return U._valid_minimap_x_y(x, y, obs)

    @staticmethod
    def attack_location(obs):
        player_relative = \
            obs.observation.feature_screen.player_relative
        player_enemy = features.PlayerRelative.ENEMY
        player_self = features.PlayerRelative.SELF
        enemy_ys, enemy_xs = \
            (player_relative == player_enemy).nonzero()
        if len(enemy_xs) == 0:
            # if no enemy on screen follow army-enemy direction
            enemy_dir = U.attack_location_army2enemy(obs)
            return enemy_dir

        # if enemy on screen follow enemy closest
        army_ys, army_xs = (player_relative == player_self).nonzero()
        if len(army_ys) == 0:
            return None
        army_c_x, army_c_y = (army_xs.mean(), army_ys.mean())
        distances = np.ndarray(shape=[len(enemy_xs)], dtype=np.float32)
        for i, x, y in zip(range(len(enemy_xs)), enemy_xs, enemy_ys):
            distances[i] = abs(x - army_c_x) + abs(y - army_c_y)
        pos = np.argmin(distances)
        x, y = enemy_xs[pos], enemy_ys[pos]
        enemy_pos = U._valid_screen_x_y(x, y, obs)
        return enemy_pos

    @staticmethod
    def attack_location_army2enemy(obs):
        army_front = U.army_minimap_location(obs)
        if not army_front:
            return None
        enemy_base = U.enemy_minimap_location(obs)
        # if base is clear, attack next base
        if U.get_distance(army_front, enemy_base) < 1:
            enemy_base = U.enemy_second_minimap_location(obs)
        screen_w, screen_h = U.screen_size(obs)
        attack_direction = (
            enemy_base[0] - army_front[0],
            enemy_base[1] - army_front[1])
        radius = attack_direction[0] * attack_direction[0] \
                 + attack_direction[1] * attack_direction[1]
        radius = np.sqrt(radius)

        radius = np.clip(radius, 1e-1, None)
        radius_screen = min(screen_w, screen_h) / 2.5
        ratio = float(radius_screen) / float(radius)
        x, y = (attack_direction[0] * ratio + screen_h / 2,
                attack_direction[1] * ratio + screen_w / 2)
        return U._valid_screen_x_y(x, y, obs)

    @staticmethod
    def _valid_screen_x_y(x, y, obs):
        screen_w, screen_h = U.screen_size(obs)
        _x = np.clip(x, 0, screen_w - 1)
        _y = np.clip(y, 0, screen_h - 1)
        return _x, _y

    @staticmethod
    def _valid_minimap_x_y(x, y, obs):
        minimap_w, minimap_h = U.minimap_size(obs)
        _x = np.clip(x, 0, minimap_w - 1)
        _y = np.clip(y, 0, minimap_h - 1)
        return _x, _y

    @staticmethod
    def _all_raw_units(obs, unit_type):
        raw_units = obs._raw_units.get(unit_type, [])
        return raw_units

    @staticmethod
    def _resources(obs):
        minerals = obs.observation.player.minerals
        vespene = obs.observation.player.vespene
        food_cap = obs.observation.player.food_cap
        food_used = obs.observation.player.food_used
        food = food_cap - food_used
        return minerals, vespene, food

    @staticmethod
    def _world_tl_to_minimap_px(raw_unit):
        # TODO configurable resolution
        minimap_px = point.Point(64.0, 64.0)
        map_size = point.Point(88.0, 96.0)
        pos_transform = transform.Chain(
            transform.Linear(minimap_px / map_size.max_dim()),
            transform.PixelToCoord())
        screen_pos = pos_transform.fwd_pt(
            point.Point(raw_unit.x, raw_unit.y))
        return screen_pos.x, screen_pos.y

    @staticmethod
    def can_build_pylon(obs):
        # TODO is there  enough space
        if len(U._all_raw_units(obs, Pylon.build_type)) == 0:
            return False
        return obs.observation.player.minerals >= Pylon.minerals

    @staticmethod
    def can_build_gateway(obs):
        # TODO is there  enough space
        if len(U._all_raw_units(obs, Gateway.build_type)) == 0:
            return False
        for building_type in Gateway.requirement_types:
            if len(U._all_raw_units(obs, building_type)) == 0:
                return False
        return obs.observation.player.minerals >= Gateway.minerals

    @staticmethod
    def can_build_assimilator(obs):
        if len(U._all_raw_units(obs, Assimilator.build_type)) == 0:
            return False
        nexus_num = len(U._all_raw_units(obs, units.Protoss.Nexus))
        assimilator_num = len(U._all_raw_units(obs, units.Protoss.Assimilator))
        return (assimilator_num < nexus_num * 2) and (
                obs.observation.player.minerals >= Assimilator.minerals)

    @staticmethod
    def _max_order_filter(length):
        l = lambda u: u.order_length < length
        return l

    @staticmethod
    # TODO is there  enough space
    def can_build_cyberneticscore(obs):
        if len(U._all_raw_units(obs, CyberneticsCore.build_type)) == 0:
            return False
        for building in CyberneticsCore.requirement_types:
            if len(U._all_raw_units(obs, building)) == 0:
                return False
        return obs.observation.player.minerals >= CyberneticsCore.minerals

    @staticmethod
    def can_train_probe(obs):
        if len(U.minimap_units(
                obs, Probe.build_type, _filter=U._max_order_filter(5))) == 0:
            return False
        minerals, vespene, food = U._resources(obs)
        return minerals >= Probe.minerals and food >= Probe.food

    @staticmethod
    def can_train_zealot(obs):
        if len(U.minimap_units(
                obs, Zealot.build_type, U._max_order_filter(5))) == 0:
            return False
        minerals, _, food = U._resources(obs)
        return minerals >= Zealot.minerals and food >= Zealot.food

    @staticmethod
    def can_train_stalker(obs):
        if len(U.minimap_units(
                obs, Stalker.build_type, U._max_order_filter(5))) == 0:
            return False
        minerals, vespene, food = U._resources(obs)
        return minerals >= Stalker.minerals and \
               vespene >= Stalker.gas and \
               food >= Stalker.food

    @staticmethod
    def can_select_army(obs):
        action_id = FUNCTIONS.select_army.id
        return action_id in obs.observation.available_actions

    @staticmethod
    def can_collect_gas(obs):
        assimilator_type = units.Protoss.Assimilator
        assimilators = obs._feature_units_completed.get(assimilator_type, [])
        for assimilator in assimilators:
            if assimilator.assigned_harvesters < 3:
                return True
        return False


def do_nothing():
    funcs = [
        FUNCTIONS.no_op]
    funcs_args = [lambda obs: ()]
    cond = lambda obs: True
    return cond, funcs, funcs_args


def build_a_pylon():
    funcs = [
        # move camera to worker
        FUNCTIONS.move_camera,
        # select all workers
        FUNCTIONS.select_point,
        # move camera to base
        FUNCTIONS.move_camera,
        # build a pylon
        FUNCTIONS.Build_Pylon_screen]
    funcs_args = [
        lambda obs: (U.rand_minimap_unit_location(obs, units.Protoss.Nexus),),
        lambda obs: ("select_all_type", U.rand_unit_location(
            obs, U.worker_type(obs))),
        lambda obs: (U.base_minimap_location(obs),),
        lambda obs: ("now", U.new_pylon_location(obs))]
    cond = U.can_build_pylon
    return cond, funcs, funcs_args


def build_a_gateway():
    funcs = [
        # move camera to worker
        FUNCTIONS.move_camera,
        # select all workers
        FUNCTIONS.select_point,
        # move camera to base
        FUNCTIONS.move_camera,
        # build a gateway
        FUNCTIONS.Build_Gateway_screen]
    funcs_args = [
        lambda obs: (U.rand_minimap_unit_location(obs, U.worker_type(obs)),),
        lambda obs: ("select_all_type", U.rand_unit_location(
            obs, U.worker_type(obs))),
        lambda obs: (U.base_minimap_location(obs),),
        lambda obs: ("now", U.new_gateway_location(obs))]
    cond = U.can_build_gateway
    return cond, funcs, funcs_args


def build_a_assimilator():
    funcs = [
        # move camera to worker
        FUNCTIONS.move_camera,
        # select all workers
        FUNCTIONS.select_point,
        # move camera to base
        FUNCTIONS.move_camera,
        # build a assimilator
        FUNCTIONS.Build_Assimilator_screen]
    funcs_args = [
        lambda obs: (U.rand_minimap_unit_location(obs, U.worker_type(obs)),),
        lambda obs: ("select_all_type", U.rand_unit_location(
            obs, U.worker_type(obs))),
        lambda obs: (U.base_minimap_location(obs),),
        lambda obs: ("now", U.new_assimilator_location(obs))]
    cond = U.can_build_assimilator
    return cond, funcs, funcs_args


def build_a_cyberneticscore():
    cond = U.can_build_cyberneticscore
    funcs = [
        # move camera to worker
        FUNCTIONS.move_camera,
        # select all workers
        FUNCTIONS.select_point,
        # move camera to base
        FUNCTIONS.move_camera,
        # build a CyberneticsCore
        FUNCTIONS.Build_CyberneticsCore_screen]
    funcs_args = [
        lambda obs: (U.rand_minimap_unit_location(obs, U.worker_type(obs)),),
        lambda obs: ("select_all_type", U.rand_unit_location(
            obs, U.worker_type(obs))),
        lambda obs: (U.base_minimap_location(obs),),
        lambda obs: ("now", U.new_cyberneticscore_location(obs))]
    return cond, funcs, funcs_args


def training_a_probe():
    funcs = [
        # move camera to base
        FUNCTIONS.move_camera,
        # select all gateways
        FUNCTIONS.select_point,
        # train a zealot
        FUNCTIONS.Train_Probe_quick]
    funcs_args = [
        lambda obs: (U.rand_minimap_unit_location(
            obs, units.Protoss.Nexus, U._max_order_filter(5)),),
        lambda obs: ("select_all_type", U.rand_unit_location(
            obs, units.Protoss.Nexus, U._max_order_filter(5))),
        lambda obs: ("now",)]
    cond = U.can_train_probe
    return cond, funcs, funcs_args


def training_a_zealot():
    funcs = [
        # move camera to base
        FUNCTIONS.move_camera,
        # select all gateways
        FUNCTIONS.select_point,
        # train a zealot
        FUNCTIONS.Train_Zealot_quick]
    funcs_args = [
        lambda obs: (U.rand_minimap_unit_location(
            obs, units.Protoss.Gateway, U._max_order_filter(5)),),
        lambda obs: ("select_all_type", U.rand_unit_location(
            obs, units.Protoss.Gateway, U._max_order_filter(5))),
        lambda obs: ("now",)]
    cond = U.can_train_zealot
    return cond, funcs, funcs_args


def training_a_stalker():
    funcs = [
        # move camera to base
        FUNCTIONS.move_camera,
        # select all gateways
        FUNCTIONS.select_point,
        # train a stalker
        FUNCTIONS.Train_Stalker_quick]
    funcs_args = [
        lambda obs: (U.rand_minimap_unit_location(
            obs, units.Protoss.Gateway, U._max_order_filter(5)),),
        lambda obs: ("select_all_type", U.rand_unit_location(
            obs, units.Protoss.Gateway, U._max_order_filter(5))),
        lambda obs: ("now",)]
    cond = U.can_train_stalker
    return cond, funcs, funcs_args


def callback_idle_workers():
    action_id = FUNCTIONS.select_idle_worker.id
    cond = lambda obs: action_id in obs.observation.available_actions
    funcs = [
        # select idle workers
        FUNCTIONS.select_idle_worker,
        # move camera to base
        FUNCTIONS.move_camera,
        # collect minerals
        FUNCTIONS.Harvest_Gather_screen]
    funcs_args = [
        lambda obs: ("select_all",),
        lambda obs: (U.base_minimap_location(obs),),
        lambda obs: ("now", U.mineral_location(obs))]

    return cond, funcs, funcs_args


def collect_minerals():
    funcs = [
        # move camera to base
        FUNCTIONS.move_camera,
        # select a worker
        FUNCTIONS.select_point,
        # collect minerals
        FUNCTIONS.Smart_screen]
    funcs_args = [
        lambda obs: (U.base_minimap_location(obs),),
        lambda obs: ("select", U.bad_worker_location(obs)),
        lambda obs: ("now", U.mineral_location(obs))]
    cond = lambda obs: True
    return cond, funcs, funcs_args


def collect_gas():
    cond = U.can_collect_gas
    funcs = [
        # move camera to base
        FUNCTIONS.move_camera,
        # select a worker
        FUNCTIONS.select_point,
        # collect gas
        FUNCTIONS.Harvest_Gather_screen]
    funcs_args = [
        lambda obs: (U.base_minimap_location(obs),),
        # todo select an `idle` worker
        lambda obs: ("select", U.rand_unit_location(
            obs, U.worker_type(obs))),
        lambda obs: ("now", U.gas_location(obs))]

    return cond, funcs, funcs_args


def move_screen_topleft():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # move to one corner
        FUNCTIONS.Move_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.screen_topleft(obs))]
    cond = U.can_select_army
    return cond, funcs, funcs_args


def move_screen_top():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # move to one corner
        FUNCTIONS.Move_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.screen_top(obs))]
    cond = U.can_select_army
    return cond, funcs, funcs_args


def move_screen_topright():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # move to one corner
        FUNCTIONS.Move_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.screen_topright(obs))]
    cond = U.can_select_army
    return cond, funcs, funcs_args


def move_screen_right():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # move to one corner
        FUNCTIONS.Move_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.screen_right(obs))]
    cond = U.can_select_army
    return cond, funcs, funcs_args


def move_screen_bottomright():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # move to one corner
        FUNCTIONS.Move_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.screen_bottomright(obs))]
    cond = U.can_select_army
    return cond, funcs, funcs_args


def move_screen_bottom():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # move to one corner
        FUNCTIONS.Move_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.screen_bottom(obs))]
    cond = U.can_select_army
    return cond, funcs, funcs_args


def move_screen_bottomleft():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # move to one corner
        FUNCTIONS.Move_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.screen_bottomleft(obs))]
    cond = U.can_select_army
    return cond, funcs, funcs_args


def move_screen_left():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # move to one corner
        FUNCTIONS.Move_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.screen_left(obs))]
    cond = U.can_select_army
    return cond, funcs, funcs_args


def attack_enemy():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # attack enemy
        FUNCTIONS.Attack_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.attack_location(obs))]
    cond = U.can_select_army
    return cond, funcs, funcs_args


class ProtossMacro(collections.namedtuple(
    "ProtossMacro", ["id", "name", "ability_id",
                     "general_id", "cond", "func", "args"])):
    __slots__ = ()

    @classmethod
    def ability(cls, id_, name, func, ability_id, general_id=0):
        values = func()
        cond = values[0]
        func = list(zip(*values[1:]))
        return cls(id_, name, ability_id, general_id, cond, func, [])

    def __hash__(self):
        return self.id

    def __call__(self):
        return MacroCall(self)


_PROTOSS_MACROS = [
    # ProtossMacro.ability(0, "Build_Pylon", build_a_pylon, 0),
    # ProtossMacro.ability(1, "Build_Gateway", build_a_gateway, 1),

    # ProtossMacro.ability(9, "Move_TopLeft", move_screen_topleft, 9),
    # ProtossMacro.ability(10, "Move_Top", move_screen_top, 10),
    # ProtossMacro.ability(11, "Move_TopRight", move_screen_topright, 11),
    # ProtossMacro.ability(12, "Move_Right", move_screen_right, 12),
    # ProtossMacro.ability(13, "Move_BottomRight", move_screen_bottomright, 13),
    # ProtossMacro.ability(14, "Move_Bottom", move_screen_bottom, 14),
    # ProtossMacro.ability(15, "Move_BottomLeft", move_screen_bottomleft, 15),
    # ProtossMacro.ability(16, "Move_Left", move_screen_left, 16),
    # ProtossMacro.ability(17, "Attack_Enemy", attack_enemy, 17),
    # ProtossMacro.ability(18, "No_op", do_nothing, 18)
    ProtossMacro.ability(0, "Build_Pylon", build_a_pylon, 0),
    ProtossMacro.ability(1, "Build_Gateway", build_a_gateway, 1),
    ProtossMacro.ability(2, "Train_Probe", training_a_probe, 2),
    ProtossMacro.ability(3, "Train_Zealot", training_a_zealot, 3),
    ProtossMacro.ability(4, "Callback_Idle_Workers", callback_idle_workers, 4),
    ProtossMacro.ability(5, "Build_Assimilator", build_a_assimilator, 5),
    ProtossMacro.ability(6, "Build_CyberneticsCore", build_a_cyberneticscore, 6),
    ProtossMacro.ability(7, "Train_Stalker", training_a_stalker, 7),
    ProtossMacro.ability(8, "Collect_Gas", collect_gas, 8),
    ProtossMacro.ability(9, "Attack_Enemy", attack_enemy, 9),
    ProtossMacro.ability(10, "No_op", do_nothing, 10)
]


class MacroCall(object):

    def __init__(self, macro):
        func = macro.func
        self.id = macro.id
        self.func = func

    def __getitem__(self, item):
        return self.func[item]

    def __iter__(self):
        return iter(self.func)

    def __len__(self):
        return len(self.func)

    def __str__(self):
        return str(self.id)


class ProtossMacros(object):
    def __init__(self, macros):
        macros = sorted(macros, key=lambda f: f.id)
        macros = [f._replace(id=_ProtossMacros(f.id))
                  for f in macros]

        self._macro_list = macros
        self._macro_dict = {f.name: f for f in macros}
        if len(self._macro_dict) != len(self._macro_list):
            raise ValueError("Macro names must be unique.")

    def __getattr__(self, name):
        return self._macro_dict.get(name, None)

    def __getitem__(self, key):
        if isinstance(key, numbers.Integral):
            return self._macro_list[key]
        return self._macro_dict[key]

    def __getstate__(self):
        return self._macro_list

    def __setstate__(self, macros):
        self.__init__(macros)

    def __iter__(self):
        return iter(self._macro_list)

    def __len__(self):
        return len(self._macro_list)

    def __eq__(self, other):
        return self._macro_list == other._macro_list


_ProtossMacros = enum.IntEnum(
    "_ProtossMacros", {f.name: f.id for f in _PROTOSS_MACROS})

PROTOSS_MACROS = ProtossMacros(_PROTOSS_MACROS)

_PROTOSS_UNITS = [
    Probe, Zealot, Stalker, Sentry, Adept,
    HighTemplar, DarkTemplar, Archon, Observer,
    WarpPrism, Immortal,
    Colossus, Disruptor, Phoenix, VoidRay, Oracle,
    Tempest, Carrier, Interceptor, Mothership]

_PROTOSS_UNITS_DICT = {
    units.Protoss.Probe: Probe,
    units.Protoss.Zealot: Zealot,
    units.Protoss.Stalker: Stalker,
    units.Protoss.Sentry: Sentry,
    units.Protoss.Adept: Adept,
    units.Protoss.HighTemplar: HighTemplar,
    units.Protoss.DarkTemplar: DarkTemplar,
    units.Protoss.Archon: Archon,
    units.Protoss.Observer: Observer,
    units.Protoss.WarpPrism: WarpPrism,
    units.Protoss.Immortal: Immortal,
    units.Protoss.Colossus: Colossus,
    units.Protoss.Disruptor: Disruptor,
    units.Protoss.Phoenix: Phoenix,
    units.Protoss.VoidRay: VoidRay,
    units.Protoss.Oracle: Oracle,
    units.Protoss.Tempest: Tempest,
    units.Protoss.Carrier: Carrier,
    units.Protoss.Interceptor: Interceptor,
    units.Protoss.Mothership: Mothership
}

_PROTOSS_UNITS_MACROS = {
    PROTOSS_MACROS.Train_Probe.id: Probe,
    PROTOSS_MACROS.Train_Zealot.id: Zealot,
    PROTOSS_MACROS.Train_Stalker.id: Stalker
}

_PROTOSS_UNITS_FUNCTIONS = {
    FUNCTIONS.Train_Probe_quick.id: Probe,
    FUNCTIONS.Train_Zealot_quick.id: Zealot,
    FUNCTIONS.Train_Stalker_quick.id: Stalker
}

_PROTOSS_BUILDINGS = [
    Pylon, Gateway, Assimilator,
    Forge, CyberneticsCore, Nexus, WarpGate,
    PhotonCannon, ShieldBattery,
    RoboticsFacility, Stargate, TwilightCouncil,
    RoboticsBay, FleetBeacon, TemplarArchive, DarkShrine,
    StasisTrap]

_PROTOSS_BUILDINGS_DICT = {
    units.Protoss.Pylon: Pylon,
    units.Protoss.Gateway: Gateway,
    units.Protoss.Assimilator: Assimilator,
    units.Protoss.Forge: Forge,
    units.Protoss.CyberneticsCore: CyberneticsCore,
    units.Protoss.Nexus: Nexus,
    units.Protoss.WarpGate: WarpGate,
    units.Protoss.PhotonCannon: PhotonCannon,
    units.Protoss.ShieldBattery: ShieldBattery,
    units.Protoss.RoboticsFacility: RoboticsFacility,
    units.Protoss.Stargate: Stargate,
    units.Protoss.TwilightCouncil: TwilightCouncil,
    units.Protoss.RoboticsBay: RoboticsBay,
    units.Protoss.FleetBeacon: FleetBeacon,
    units.Protoss.TemplarArchive: TemplarArchive,
    units.Protoss.DarkShrine: DarkShrine,
    units.Protoss.StasisTrap: StasisTrap
}

_PROTOSS_BUILDINGS_MACROS = {
    PROTOSS_MACROS.Build_Pylon.id: Pylon,
    PROTOSS_MACROS.Build_Gateway.id: Gateway,
    PROTOSS_MACROS.Build_Assimilator.id: Assimilator,
    PROTOSS_MACROS.Build_CyberneticsCore.id: CyberneticsCore
}

_PROTOSS_BUILDINGS_FUNCTIONS = {
    FUNCTIONS.Build_Pylon_screen.id: Pylon,
    FUNCTIONS.Build_Gateway_screen.id: Gateway,
    FUNCTIONS.Build_Assimilator_screen.id: Assimilator,
    FUNCTIONS.Build_CyberneticsCore_screen.id: CyberneticsCore
}
