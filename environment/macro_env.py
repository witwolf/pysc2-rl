# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/1.
#


import logging
from pysc2.env import sc2_env
import sys
sys.path.append('.')
from lib.protoss_adapter import ProtossInformationAdapter


def default_macro_env_maker(kwargs):
    '''
    :param kwargs: map_name, players, ... almost same as SC2Env
    :return: env_maker
    '''
    assert kwargs.get('map_name') is not None

    screen_sz = kwargs.pop('screen_size', 64)
    minimap_sz = kwargs.pop('minimap_size', 64)
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
                        sc2_env.Difficulty.very_easy)]
    if 'game_steps_per_episode' not in kwargs:
        kwargs['game_steps_per_episode'] = 0
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
                 debug=False,
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

        self._debug = debug
        self._last_obs = None
        self._information = ProtossInformationAdapter(None)

    def reset(self):
        obs = super().reset()
        self._last_obs = obs
        self._information = ProtossInformationAdapter(None)
        return obs

    def step(self, macros, update_observation=None):
        """Returned with every call to `step` and `reset` on an environment.

        A `TimeStepWrapper` contains the data emitted by an environment at each step of
        interaction. A `TimeStepWrapper` holds a `step_type`, an `observation`, and an
        associated `reward` and `discount, and a 'success'`.

        The first `TimeStepWrapper` in a sequence will have `StepType.FIRST`. The final
        `TimeStep` will have `StepType.LAST`. All other `TimeStepWrapper`s in a sequence will
        have `StepType.MID.

        Attributes:
            step_type: A `StepType` enum value.
            reward: A scalar, or 0 if `step_type` is `StepType.FIRST`, i.e. at the
            start of a sequence.
            discount: A discount value in the range `[0, 1]`, or 0 if `step_type`
            is `StepType.FIRST`, i.e. at the start of a sequence.
            observation: A NumPy array, or a dict, list or tuple of arrays.
            macro_success: A bool, tell whether last macro action succeed
        """

        class TimestepWrapper:
            def __init__(self, timestep, information, success):
                self._timestep = timestep
                self.information = information
                self.macro_success = success

            def __getattr__(self, item):
                return getattr(self._timestep, item)

        macro = macros[0]
        for act_func, arg_func in macro:
            obs = self._last_obs[0]
            # action not available
            if not act_func.id in obs.observation.available_actions:
                if self._debug:
                    logging.warning("%s not available,  macro: %s",
                                    act_func.id, macro)
                    return [TimestepWrapper(obs, self._information, False)]
            args = arg_func(obs)
            act = (act_func(*args),)
            self._last_obs = super().step(act, update_observation)
            self._information.update(obs, self._step_mul)

            #  TODO remove this check temporary

            # last_actions = obs.observation.last_actions
            # if len(last_actions) == 0 or last_actions[0] != act_func.id:
            #     if self._debug:
            #         logging.warning(
            #             "%s execute failed, last_action:%s, macro: %s", act_func.id,
            #             last_actions[0] if len(last_actions) else 'None', macro)
            #     return [TimestepWrapper(self._last_obs[0], False)]

        if self._debug:
            logging.warning("%s execute success", macro)
        return [TimestepWrapper(self._last_obs[0], self._information, True)]
