# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#


from pysc2.lib import actions
from pysc2.lib.actions import *
from pysc2.lib import units


def base_location(*args):
    pass


def random_id(*args):
    pass


def get_worker_type(*args):
    pass


def get_new_pylon_location(*args):
    pass


def get_new_gateway_location(*args):
    pass


def get_new_assimilator_location(*args):
    pass


def get_new_cyberneticscore_location(*args):
    pass


def get_mineral_location(*args):
    pass


def get_gas_location(*args):
    pass


def get_army_location(*args):
    pass


def get_screen_topleft(*args):
    pass


def get_screen_top(*args):
    pass


def get_screen_right(*args):
    pass


def get_screen_topright(*args):
    pass


def get_screen_left(*args):
    pass


def get_attack_location(*args):
    pass


def get_screen_bottomleft(*args):
    pass


def get_screen_bottomright(*args):
    pass


def get_screen_bottom(*args):
    pass


##### TODO

class Macro(collections.namedtuple(
    "Macro", ["id", "name", "ability_id", "general_id", "function_calls",
              "args"])):
    __slots__ = ()

    # todo
    @classmethod
    def ability(cls, id_, name, function_calls, ability_id, general_id=0):
        return cls(id_, name, function_calls, ability_id, general_id)


MACROS = [
    Macro.ability_id(0, "BuildPylon", [
        (move_camera, base_location),
        (select_unit, "select_all_type", random_id),
        ("Build_Pylon_screen", "now", get_new_pylon_location)]),
    Macro.ability_id(1, "Build_Gateway_screen", [
        (move_camera, base_location),
        (select_unit, "select_all_type", random_id),
        ("Build_Gateway_screen", "now", get_new_gateway_location)]),
    Macro.ability_id(2, "Build_Assimilator_screen", [
        (move_camera, base_location),
        (select_unit, "select_all_type", random_id),
        ("Build_Assimilator_screen", "now", get_new_assimilator_location)]),
    Macro.ability_id(3, "Build_CyberneticsCore_screen", [
        (move_camera, base_location),
        (select_unit, "select_all_type", random_id),
        ("Build_CyberneticsCore_screen", "now", get_new_cyberneticscore_location)]),
    Macro.ability_id(4, "Train_Zealot_quick", [
        (move_camera, base_location),
        (select_unit, "select_all_type", units.Protoss.Gateway),
        ("Train_Zealot_quick", "now")]),
    Macro.ability_id(5, "Train_Stalker_quick", [
        (move_camera, base_location),
        (select_unit, "select_all_type", units.Protoss.Gateway),
        ("Train_Stalker_quick", "now")]),
    Macro.ability_id(6, "Collect_Mineral", [
        (move_camera, base_location),
        (select_idle_worker, "select_all"),
        ("Smart_screen", "now", get_mineral_location)]),
    Macro.ability_id(7, "Collect_Gas", [
        (move_camera, base_location),
        (select_unit, "select_all_type", random_id),
        ("Smart_screen", "now", get_gas_location)]),
    Macro.ability_id(8, "Move_Scree_Top_Left", [
        (move_camera, get_army_location),
        (select_army, "select"),
        ("Move_screen", "now", get_screen_topleft)]),
    Macro.ability_id(9, "Move_Scree_Top", [
        (move_camera, get_army_location),
        (select_army, "select"),
        ("Move_screen", "now", get_screen_top)]),
    Macro.ability_id(10, "Move_Scree_Top_Right", [
        (move_camera, get_army_location),
        (select_army, "select"),
        ("Move_screen", "now", get_screen_topright)]),
    Macro.ability_id(11, "Move_Scree_Right", [
        (move_camera, get_army_location),
        (select_army, "select"),
        ("Move_screen", "now", get_screen_right)]),
    Macro.ability_id(12, "Move_Scree_Bottom_Right", [
        (move_camera, get_army_location),
        (select_army, "select"),
        ("Move_screen", "now", get_screen_bottomright)]),
    Macro.ability_id(13, "Move_Scree_Bottom", [
        (move_camera, get_army_location),
        (select_army, "select"),
        ("Move_screen", "now", get_screen_bottom)]),
    Macro.ability_id(14, "Move_Scree_Bottom_Left", [
        (move_camera, get_army_location),
        (select_army, "select"),
        ("Move_screen", "now", get_screen_bottomleft)]),
    Macro.ability_id(15, "Move_Left", [
        (move_camera, get_army_location),
        (select_army, "select"),
        ("Move_screen", "now", get_screen_left)]),
    Macro.ability_id(16, "Attack_screen", [
        (move_camera, get_army_location),
        (select_army, "select"),
        ("Attack_screen", "now", get_attack_location)])]
