# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#


from pysc2.env import sc2_env
from pysc2.lib.run_parallel import RunParallel

DEFAULT_SCREEN_SIZE = 64
DEFAULT_MINIMAP_SIZE = 64
DEFAULT_STEP_MUL = 8


def default_env_maker(kwargs):
    '''
    :param kwargs: map_name, players, ... almost same as SC2Env
    :return: env_maker
    '''
    assert kwargs.get('map_name') is not None

    screen_sz = kwargs.pop('screen_size', DEFAULT_SCREEN_SIZE)
    minimap_sz = kwargs.pop('minimap_size', DEFAULT_MINIMAP_SIZE)
    assert screen_sz == minimap_sz
    if 'agent_interface_format' not in kwargs:
        kwargs['agent_interface_format'] = sc2_env.AgentInterfaceFormat(
            use_feature_units=True,
            use_raw_units=True,
            feature_dimensions=sc2_env.Dimensions(
                screen=(screen_sz, screen_sz),
                minimap=(minimap_sz, minimap_sz)))
    if 'visualize' not in kwargs:
        kwargs['visualize'] = False
    if 'step_mul' not in kwargs:
        kwargs['step_mul'] = DEFAULT_STEP_MUL
    return sc2_env.SC2Env(**kwargs)


class ParallelEnvs(object):
    def __init__(self,
                 envs=None,
                 env_num=None,
                 env_args=None,
                 env_makers=default_env_maker):
        self._parallel = RunParallel()
        if envs is not None:
            self._envs = envs
            self._env_num = len(envs)
        else:
            self._env_num = env_num
            if not isinstance(env_makers, list):
                env_makers = [env_makers] * env_num
            if not isinstance(env_args, list):
                env_args = [env_args] * env_num
            self._envs = self._parallel.run(
                (func, args) for func, args in zip(env_makers, env_args))

    # todo support only one player mode

    def action_spec(self):
        return list(env.action_spec()[0] for env in self._envs)

    def observation_spec(self):
        return list(env.observation_spec()[0] for env in self._envs)

    def step(self, actions):
        obs = self._parallel.run(
            (env.step, acts) for env, acts in zip(self._envs, actions))
        return [o[0][0] for o in obs], [o[1] for o in obs]

    def reset(self):
        obs = self._parallel.run(env.reset for env in self._envs)
        return [o[0] for o in obs]

    def close(self):
        self._parallel.run(env.close for env in self._envs)
        self._parallel.shutdown()

    @property
    def state(self):
        return list(env.state for env in self._envs)

    def __getitem__(self, item):
        return self._envs[item]
