# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/1.
#


import sys
import logging
from pysc2.env import sc2_env
from pysc2.lib.actions import FUNCTIONS

sys.path.append('.')
from lib.protoss_timestep import ProtossTimeStepFactory


def default_macro_env_maker(kwargs):
    '''
    :param kwargs: map_name, players, ... almost same as SC2Env
    :return: env_maker
    '''
    assert kwargs.get('map_name') is not None

    screen_sz = kwargs.pop('screen_size', 64)
    minimap_sz = kwargs.pop('minimap_size', 64)
    difficulty = kwargs.pop('difficulty',
                            sc2_env.Difficulty.very_easy)
    assert screen_sz == minimap_sz
    if 'agent_interface_format' not in kwargs:
        kwargs['agent_interface_format'] = sc2_env.AgentInterfaceFormat(
            use_feature_units=True,
            use_raw_units=True,
            use_unit_counts=True,
            feature_dimensions=sc2_env.Dimensions(
                screen=(screen_sz, screen_sz),
                minimap=(minimap_sz, minimap_sz)))

    if 'players' not in kwargs:
        kwargs['players'] = [
            sc2_env.Agent(sc2_env.Race.protoss),
            sc2_env.Bot(sc2_env.Race.terran,
                        difficulty)]
    if 'game_steps_per_episode' not in kwargs:
        kwargs['game_steps_per_episode'] = 16000
    if 'visualize' not in kwargs:
        kwargs['visualize'] = False
    if 'step_mul' not in kwargs:
        kwargs['step_mul'] = 4
    return MacroEnv(**kwargs)


class MacroEnv(sc2_env.SC2Env):
    def __init__(self,
                 _only_use_kwargs=None,
                 map_name=None,
                 players=None,
                 agent_race=None,
                 bot_race=None,
                 difficulty=None,
                 screen_size_px=None,
                 minimap_size_px=None,
                 agent_interface_format=None,
                 discount=1.,
                 discount_zero_after_timeout=False,
                 visualize=False,
                 step_mul=None,
                 save_replay_episodes=0,
                 replay_dir=None,
                 replay_prefix=None,
                 game_steps_per_episode=None,
                 score_index=None,
                 score_multiplier=None,
                 random_seed=None,
                 disable_fog=False,
                 ensure_available_actions=True):
        super().__init__(
            _only_use_kwargs,
            map_name, players,
            agent_race, bot_race,
            difficulty, screen_size_px,
            minimap_size_px,
            agent_interface_format,
            discount,
            discount_zero_after_timeout,
            visualize,
            step_mul, save_replay_episodes,
            replay_dir,
            replay_prefix,
            game_steps_per_episode,
            score_index,
            score_multiplier,
            random_seed,
            disable_fog,
            ensure_available_actions)

        self._timestep_factory = ProtossTimeStepFactory(self._step_mul)

    def reset(self):
        obs = super().reset()
        self._timestep_factory.reset()
        self._last_obs = (self._timestep_factory.update(obs[0]),)
        return self._last_obs

    def step(self, macros, update_observation=None):
        macro = macros[0]
        success = True
        for act_func, arg_func in macro:
            obs = self._last_obs[0]
            act = self.valid_func_call(act_func, arg_func, obs, macro)
            if act is None:
                success = False
                act = (FUNCTIONS.no_op(),)
            obs = super().step(act, update_observation)
            self._last_obs = (self._timestep_factory.update(obs[0]),)
            if not success:
                break
        logging.debug("%s execute %s", macro, "success" if success else "failed")
        self._last_obs[0]._macro_success = success
        if self._last_obs[0].last():
            self._timestep_factory.reset()
        return self._last_obs

    def valid_func_call(self, act_func, args_func, obs, macro):
        if not act_func.id in obs.observation.available_actions:
            logging.debug("%s not available in %s", act_func.id, macro)
            return None
        args = args_func(obs)
        if args is None:
            logging.debug("arg none")
            return None
        for arg in args:
            if arg is None:
                logging.debug("arg none")
                return None
        return (act_func(*args),)
