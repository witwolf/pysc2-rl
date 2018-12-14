# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#

import cloudpickle
from multiprocessing import Process, Pipe
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


class MultiThreadEnvs(object):
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
        return [o[0] for o in obs]

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


# multi process enviroments

class PickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def working_process(remote, env_maker_wrapper, env_arg):
    env_maker = env_maker_wrapper.x
    env = env_maker(env_arg)
    while True:
        cmd, data = remote.recv()
        if cmd == 'observation_spec':
            remote.send(env.observation_spec()[0])
        elif cmd == 'action_spec':
            remote.send(env.action_spec()[0])
        elif cmd == 'step':
            obs = env.step(data.x)
            remote.send(obs[0])
        elif cmd == 'reset':
            obs = env.reset()
            remote.send(obs[0])
        elif cmd == 'close':
            env.close()
            break


class MultiProcessEnvs(object):
    def __init__(self,
                 env_num=None,
                 env_args=None,
                 env_makers=default_env_maker):
        self._env_num = env_num
        if not isinstance(env_makers, list):
            env_makers = [env_makers] * env_num
        if not isinstance(env_args, list):
            env_args = [env_args] * env_num
        remotes, work_remotes = zip(*[Pipe() for _ in range(env_num)])
        ps = []
        for remote, env_maker, env_arg in zip(work_remotes, env_makers, env_args):
            p = Process(target=working_process, args=(
                remote, PickleWrapper(env_maker), env_arg))
            p.start()
            ps.append(p)

        self._ps = ps
        self._remotes = remotes

    def action_spec(self):
        for remote in self._remotes:
            remote.send(('action_spec', None))
        results = [remote.recv() for remote in self._remotes]
        return results

    def observation_spec(self):
        for remote in self._remotes:
            remote.send(('observation_spec', None))
        results = [remote.recv() for remote in self._remotes]
        return results

    def step(self, actions):
        for remote, action in zip(self._remotes, actions):
            remote.send(('step', PickleWrapper(action)))
        results = [remote.recv() for remote in self._remotes]
        return results

    def reset(self):
        for remote in self._remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self._remotes]
        return results

    def close(self):
        for remote in self._remotes:
            remote.send(('close', None))
        for p in self._ps:
            p.join()


class ParallelEnvs(object):
    cls = {
        'multi_thread': MultiThreadEnvs,
        'multi_process': MultiProcessEnvs}

    @staticmethod
    def new(mode='multi_thread', **kwargs):
        return ParallelEnvs.cls[mode](**kwargs)
