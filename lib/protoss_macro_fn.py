from pysc2.lib import actions, features, units
import numpy as np
import random

class HelperFunction(object):
  base_x = -1
  base_y = -1
  enemy_x = -1
  enemy_y = -1
  screen_width = 84
  screen_height = 84

  @classmethod
  def get_base_minimap(cls, obs):
    if cls.base_x < 0 and cls.base_y < 0:
      player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
      if player_y.mean() > 31 and player_x.mean() > 31:
        cls.base_x = 51
        cls.base_y = 49
      else:
        cls.base_x = 10
        cls.base_y = 15
    return cls.base_x, cls.base_y

  @classmethod
  def get_enemy_minimap(cls, obs):
    if cls.enemy_x < 0:
      if cls.base_x < 31:
        cls.enemy_x = 49
        cls.enemy_y = 51
      else:
        cls.enemy_x = 15
        cls.enemy_y = 10
    return cls.enemy_x, cls.enemy_y

  @classmethod
  def get_worker_type(cls, obs):
    return units.Protoss.Probe

  @classmethod
  def get_geyser_type(cls, obs):
    return units.Protoss.Assimilator

  @classmethod
  def get_random_pos(cls, obs, unit_type):
    units_pos = [(unit.x, unit.y) for unit in obs.observation.feature_units if unit.unit_type == unit_type]
    return units_pos[int(np.random.random() * len(units_pos))]

  @classmethod
  def get_new_pylon_location(cls, obs):
    return random.randint(0, 83), random.randint(0, 83)

  @classmethod
  def get_new_gateway_location(cls, obs):
    return random.randint(0, 83), random.randint(0, 83)

  @classmethod
  def get_new_assimilator_location(cls, obs):
    return random.randint(0, 83), random.randint(0, 83)

  @classmethod
  def get_new_cyberneticscore_location(cls, obs):
    return random.randint(0, 83), random.randint(0, 83)

  @classmethod
  def get_minieral_location(cls, obs):
    minerals_x, minerals_y = (obs.observation.feature_screen.unit_type == units.Neutral.MineralField).nonzero()
    # if no mineral, go to center
    if len(minerals_x) == 0:
      return cls.screen_width / 2, cls.screen_height / 2
    mineral_id = random.randint(len(minerals_x))
    return minerals_x[mineral_id], minerals_y[mineral_id]

  @classmethod
  def get_gas_location(cls, obs):
    gas_x, gas_y = (obs.observation.feature_screen.unit_type == cls().get_geyser_type(obs)).nonzero()
    # if no gas, go to center
    if len(gas_x) == 0:
      return cls.screen_width / 2, cls.screen_height / 2
    gas_id = random.randint(len(gas_x))
    return gas_x[gas_id], gas_y[gas_id]

  @classmethod
  def get_screen_top(cls, obs):
    return cls.screen_width / 2, 0

  @classmethod
  def get_screen_topright(cls, obs):
    return cls.screen_width - 1, 0

  @classmethod
  def get_screen_right(cls, obs):
    return cls.screen_width - 1, cls.screen_height / 2

  @classmethod
  def get_screen_bottomright(cls, obs):
    return cls.screen_width - 1, cls.screen_height - 1

  @classmethod
  def get_screen_bottom(cls, obs):
    return cls.screen_width / 2, cls.screen_height - 1

  @classmethod
  def get_screen_bottomleft(cls, obs):
    return 0, cls.screen_height - 1

  @classmethod
  def get_screen_left(cls, obs):
    return 0, cls.screen_height / 2

  @classmethod
  def get_screen_topleft(cls, obs):
    return 0, 0

  '''
    get my army location on minimap, most close to enemy base
    :return: (x,y)
  '''
  @classmethod
  def get_army_minimap(cls, obs):
    poses_xs, poses_ys = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
    if len(poses_xs) == 0:
      return cls().get_base_minimap(obs)
    enemy_loc = cls().get_enemy_minimap(obs)
    min_dist = 100000
    min_pos = (poses_xs[0], poses_ys[0])
    for (x, y) in zip(poses_xs, poses_ys):
      dist = abs(x-enemy_loc[0])+abs(y-enemy_loc[1])
      if dist < min_dist:
        min_dist = dist
        min_pos = (x, y)
    return min_pos

  @classmethod
  def get_attack_location(cls, obs):
    enemy_xs, enemy_ys = (obs.observation.feature_screen.player_relative == features.PlayerRelative.ENEMY).nonzero()
    # if no enemy on screen
    if len(enemy_xs[0]) == 0:
      # follow army-enemy direction
      army_front = cls().get_army_minimap(obs)
      enemy_base = cls().get_enemy_minimap(obs)
      attack_direction = (enemy_base[0] - army_front[0], enemy_base[1] - army_front[1])
      radius = attack_direction[0]*attack_direction[0] + attack_direction[1]*attack_direction[1]
      ratio = float(min(cls.screen_width, cls.screen_height) / 2.5) / float(radius)
      return attack_direction[0] * ratio - cls.screen_width / 2, attack_direction[1] * ratio - cls.screen_height / 2
    # if enemy on screen
    else:
      # follow enemy closest
      armys = (obs.observation.feature_screen.player_relative == features.PlayerRelative.ENEMY).nonzero()
      army_center = (armys[0].mean(), armys[1].mean())
      min_dist = 100000
      min_pos = (enemy_xs[0], enemy_ys[0])
      # find closest enemy
      for id in range(len(enemy_xs)):
        dist = abs(enemy_xs[id]-army_center[0])+abs(enemy_ys[id]-army_center[1])
        if dist < min_dist:
          min_dist = dist
          min_pos = (enemy_xs[id], enemy_ys[id])
      return min_pos

class MacroFunction(object):
  funcs = []
  funcs_args = []

  @classmethod
  def clear_func(cls):
    cls.funcs, cls.funcs_args = [], []

  @classmethod
  def build_a_pylon(cls):
    cls.funcs = [
      # move camera to base
      actions.FUNCTIONS.move_camera,
      # select all workers
      actions.FUNCTIONS.select_point,
      # build a pylon
      actions.FUNCTIONS.Build_Pylon_screen
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_base_minimap(obs),),
      lambda obs:("select_all_type", HelperFunction.get_random_pos(obs, HelperFunction.get_worker_type(obs))),
      lambda obs:("now", HelperFunction.get_new_pylon_location(obs))
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def build_a_gateway(cls):
    cls.funcs = [
      # move camera to base
      actions.FUNCTIONS.move_camera,
      # select all workers
      actions.FUNCTIONS.select_point,
      # build a gateway
      actions.FUNCTIONS.Build_Gateway_screen
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_base_minimap(obs),),
      lambda obs:("select_all_type", HelperFunction.get_random_pos(obs, HelperFunction.get_worker_type(obs))),
      lambda obs:("now", HelperFunction.get_new_gateway_location(obs))
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def build_a_assimilator(cls):
    cls.funcs = [
      # move camera to base
      actions.FUNCTIONS.move_camera,
      # select all workers
      actions.FUNCTIONS.select_point,
      # build a assimilator
      actions.FUNCTIONS.Build_Assimilator_screen
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_base_minimap(obs),),
      lambda obs:("select_all_type", HelperFunction.get_random_pos(obs, HelperFunction.get_worker_type(obs))),
      lambda obs:("now", HelperFunction.get_new_assimilator_location(obs))
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def build_a_CyberneticsCore(cls):
    cls.funcs = [
      # move camera to base
      actions.FUNCTIONS.move_camera,
      # select all workers
      actions.FUNCTIONS.select_point,
      # build a CyberneticsCore
      actions.FUNCTIONS.Build_CyberneticsCore_screen
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_base_minimap(obs),),
      lambda obs:("select_all_type", HelperFunction.get_random_pos(obs, HelperFunction.get_worker_type(obs))),
      lambda obs:("now", HelperFunction.get_new_cyberneticscore_location(obs))
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def training_a_probe(cls):
    cls.funcs = [
      # move camera to base
      actions.FUNCTIONS.move_camera,
      # select all gateways
      actions.FUNCTIONS.select_point,
      # train a zealot
      actions.FUNCTIONS.Train_Probe_quick
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_base_minimap(obs),),
      lambda obs:("select_all_type", HelperFunction.get_random_pos(obs, units.Protoss.Nexus)),
      lambda obs:("now",)
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def training_a_zealot(cls):
    cls.funcs = [
      # move camera to base
      actions.FUNCTIONS.move_camera,
      # select all gateways
      actions.FUNCTIONS.select_point,
      # train a zealot
      actions.FUNCTIONS.Train_Zealot_quick
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_base_minimap(obs),),
      lambda obs:("select_all_type", HelperFunction.get_random_pos(obs, units.Protoss.Gateway)),
      lambda obs:("now",)
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def training_a_stalker(cls):
    cls.funcs = [
      # move camera to base
      actions.FUNCTIONS.move_camera,
      # select all gateways
      actions.FUNCTIONS.select_point,
      # train a stalker
      actions.FUNCTIONS.Train_Stalker_quick
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_base_minimap(obs),),
      lambda obs:("select_all_type", HelperFunction.get_random_pos(obs, units.Protoss.Gateway)),
      lambda obs:("now",)
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def collect_minerals(cls):
    cls.funcs = [
      # move camera to base
      actions.FUNCTIONS.move_camera,
      # select idle workers
      actions.FUNCTIONS.select_idle_worker,
      # collect minerals
      actions.FUNCTIONS.Smart_screen
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_base_minimap(obs),),
      lambda obs:("select_all",),
      lambda obs:("now", HelperFunction.get_mineral_location(obs))
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def collect_gas(cls):
    cls.funcs = [
      # move camera to base
      actions.FUNCTIONS.move_camera,
      # select a worker
      actions.FUNCTIONS.select_point,
      # collect gas
      actions.FUNCTIONS.Smart_screen
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_base_minimap(obs),),
      lambda obs:("select", HelperFunction.get_random_pos(obs, HelperFunction.get_worker_type(obs))),
      lambda obs:("now", HelperFunction.get_gas_location(obs))
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def move_screen_topleft(cls):
    cls.funcs = [
      # move camera to army
      actions.FUNCTIONS.move_camera,
      # select all army
      actions.FUNCTIONS.select_army,
      # move to one corner
      actions.FUNCTIONS.Move_screen
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_base_minimap(obs),),
      lambda obs:("select",),
      lambda obs:("now", HelperFunction.get_screen_topleft(obs))
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def move_screen_top(cls):
    cls.funcs = [
      # move camera to army
      actions.FUNCTIONS.move_camera,
      # select all army
      actions.FUNCTIONS.select_army,
      # move to one corner
      actions.FUNCTIONS.Move_screen
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_army_minimap(obs),),
      lambda obs:("select",),
      lambda obs:("now", HelperFunction.get_screen_top(obs))
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def move_screen_topright(cls):
    cls.funcs = [
      # move camera to army
      actions.FUNCTIONS.move_camera,
      # select all army
      actions.FUNCTIONS.select_army,
      # move to one corner
      actions.FUNCTIONS.Move_screen
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_army_minimap(obs),),
      lambda obs:("select",),
      lambda obs:("now", HelperFunction.get_screen_topright(obs))
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def move_screen_right(cls):
    cls.funcs = [
      # move camera to army
      actions.FUNCTIONS.move_camera,
      # select all army
      actions.FUNCTIONS.select_army,
      # move to one corner
      actions.FUNCTIONS.Move_screen
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_army_minimap(obs),),
      lambda obs:("select",),
      lambda obs:("now", HelperFunction.get_screen_right(obs))
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def move_screen_bottomright(cls):
    cls.funcs = [
      # move camera to army
      actions.FUNCTIONS.move_camera,
      # select all army
      actions.FUNCTIONS.select_army,
      # move to one corner
      actions.FUNCTIONS.Move_screen
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_army_minimap(obs),),
      lambda obs:("select",),
      lambda obs:("now", HelperFunction.get_screen_bottomright(obs))
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def move_screen_bottom(cls):
    cls.funcs = [
      # move camera to army
      actions.FUNCTIONS.move_camera,
      # select all army
      actions.FUNCTIONS.select_army,
      # move to one corner
      actions.FUNCTIONS.Move_screen
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_army_minimap(obs),),
      lambda obs:("select",),
      lambda obs:("now", HelperFunction.get_screen_bottom(obs))
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def move_screen_bottomleft(cls):
    cls.funcs = [
      # move camera to army
      actions.FUNCTIONS.move_camera,
      # select all army
      actions.FUNCTIONS.select_army,
      # move to one corner
      actions.FUNCTIONS.Move_screen
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_army_minimap(obs),),
      lambda obs:("select",),
      lambda obs:("now", HelperFunction.get_screen_bottomleft(obs))
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def move_screen_left(cls):
    cls.funcs = [
      # move camera to army
      actions.FUNCTIONS.move_camera,
      # select all army
      actions.FUNCTIONS.select_army,
      # move to one corner
      actions.FUNCTIONS.Move_screen
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_army_minimap(obs),),
      lambda obs:("select",),
      lambda obs:("now", HelperFunction.get_screen_left(obs))
    ]
    return cls.funcs, cls.funcs_args

  @classmethod
  def attack_enemy(cls):
    cls.funcs = [
      # move camera to army
      actions.FUNCTIONS.move_camera,
      # select all army
      actions.FUNCTIONS.select_army,
      # attack enemy
      actions.Attack_screen
    ]
    cls.funcs_args = [
      lambda obs:(HelperFunction.get_army_minimap(obs),),
      lambda obs:("select",),
      lambda obs:("now", HelperFunction.get_attack_location(obs))
    ]
    return cls.funcs, cls.funcs_args