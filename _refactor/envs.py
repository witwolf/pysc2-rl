# -*- coding: utf-8 -*-

import queue
import threading
from pysc2.env import sc2_env
from algorithm.script.collect_mineral_shards import CollectMineralShards
from algorithm.script.move_to_beacon import MoveToBeacon
from pysc2.lib.actions import FunctionCall, FUNCTIONS
import numpy as np

AGENT_CLS = {
    'MoveToBeacon': MoveToBeacon,
    'CollectMineralShards': CollectMineralShards,
}


def make_env(config):
    args = config.args
    sz = args.screen_size
    env = sc2_env.SC2Env(
        map_name=args.map,
        step_mul=args.step_mul,
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            use_feature_units=True,
            use_raw_units=True,
            feature_dimensions=sc2_env.Dimensions(
                screen=(sz, sz),
                minimap=(sz, sz))),
        visualize=False)
    return EnvWrapper(env, config)


def transform_scalar_2_functioncall(actions, config):
    act_id, args = actions[0][0], [v[0] for v in actions[1:]]
    act_args = []
    for arg_type in FUNCTIONS[act_id].args:
        arg_idx = config.arg_idx[arg_type.name]
        act_arg = [args[arg_idx]]
        if arg_type.name in ['screen', 'minimap', 'screen2']:
            sz = config.sz
            act_arg = [
                act_arg[0] % sz,
                act_arg[0] // sz
            ]
        act_args.append(act_arg)
    return FunctionCall(act_id, act_args)


def transform_scalar_2_scalar(actions, config):
    length = len(actions[0])
    values = []
    for dim, is_spatial in config.policy_dims():
        if is_spatial:
            dim = config.sz * config.sz
        values.append(np.zeros(shape=(length, dim), dtype=np.int32))

    for one_hot, label in zip(values, actions):
        for i, v in enumerate(label):
            one_hot[i][v] = 1

    return values


def transform_functioncall_2_scalar(action, config):
    values = []
    for dim, is_spatial in config.policy_dims():
        if is_spatial:
            dim = config.sz * config.sz
        values.append(np.zeros(shape=(dim), dtype=np.int32))
    act_values, arg_values = [values[0]], values[1:]

    act_id = int(action.function)
    act_values[0][act_id] = 1.0

    for i, arg_type in enumerate(FUNCTIONS[act_id].args):
        arg = action.arguments[i]
        if arg_type.name in ['screen', 'minimap', 'screen2']:
            arg = arg[0] + arg[1] * config.sz
        else:
            arg = int(arg[0])
        arg_idx = config.arg_idx[arg_type.name]
        arg_values[arg_idx][arg] = 1

    return values


class EnvWrapper(object):
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def reset(self):
        obs = self.env.reset()
        return self.wrap_results(obs)

    def step(self, actions, transform_action=False):
        if transform_action:
            actions = (transform_scalar_2_functioncall(actions[0], self.config),)
        obs = self.env.step(actions)
        return self.wrap_results(obs)

    def save_replay(self, replay_dir, prefix=None):
        return self.env.save_replay(replay_dir, prefix)

    def close(self):
        self.env.close()

    def observation_spec(self):
        return self.env.observation_spec()

    def action_spec(self):
        return self.env.action_spec()

    def wrap_results(self, obs):
        obs = obs[0]
        observation = self.process_observation(obs.observation)
        reward = obs.reward
        done = obs.last()
        return obs, observation, reward, done

    def process_observation(self, observation):
        spatials = [observation['feature_screen'],
                    observation['feature_minimap']]
        non_spatials = []
        for _type in self.config.feats['non_spatial']:
            feat = observation[_type]
            if _type == 'available_actions':
                acts = np.zeros(len(self.config.acts))
                acts[feat] = 1
                feat = acts
            non_spatials.append(feat)
        return [
                   np.array([spatial]).transpose((0, 2, 3, 1)) for spatial in spatials
               ] + [
                   np.array([non_spatial]) for non_spatial in non_spatials
               ]


def process_observation_list(observations):
    batch_size = len(observations)
    obs_dim = len(observations[0])
    obs = [None] * obs_dim
    for i in range(obs_dim):
        indices = list([observations[j][i] for j in range(batch_size)])
        obs[i] = np.concatenate(indices)
    return obs


def process_action_list(actions):
    batch_size = len(actions)
    acts_dim = len(actions[0])
    acts = [None] * acts_dim
    for i in range(acts_dim):
        indices = list([actions[j][i].reshape((1, -1)) for j in range(batch_size)])
        acts[i] = np.concatenate(indices)
    return acts


def process_batch(observations, actions, next_observations, rewards, dones):
    obs = process_observation_list(observations)
    acts = process_action_list(actions)
    next_obs = process_observation_list(next_observations)
    return obs, acts, next_obs, np.array(rewards), np.array(dones)


class EnvPool(object):
    def __init__(self, config):
        self.config = config
        args = config.args
        self.transition_queue = queue.Queue(maxsize=args.buffer_size)
        self.batch_queue = queue.Queue(maxsize=args.buffer_size // args.batch_size)
        self.collector_running = False
        self.process_running = False

        def collector_loop():
            agent_cls = AGENT_CLS[args.map]
            agent = agent_cls()
            env = make_env(config)
            obs_spec = env.observation_spec()[0]
            action_spec = env.action_spec()[0]
            agent.setup(obs_spec, action_spec)
            obs, o, *_ = env.reset()

            while self.collector_running:
                act = agent.step(obs)
                a = transform_functioncall_2_scalar(act, config)
                obs, next_o, r, d = env.step((act,))
                self.transition_queue.put((o, a, next_o, r, d))
                o = next_o

            env.close()

        def processer_loop():
            while self.process_running:
                batch = [[], [], [], [], []]
                for _ in range(args.batch_size):
                    t = self.transition_queue.get()

                    for c, e in zip(batch, t):
                        c.append(e)
                batch = process_batch(*batch)
                self.batch_queue.put(batch)

        collector_threads = []
        self.collector_running = True
        for _ in range(args.env_nums):
            t = threading.Thread(target=collector_loop,
                                 args=())
            t.daemon = True
            t.start()
            collector_threads.append(t)
        self.collector_threads = collector_threads

        self.process_running = True
        process_thread = threading.Thread(target=processer_loop, args=())
        process_thread.daemon = True
        process_thread.start()

        self.process_thread = process_thread

    def get_batch(self, *args):
        return self.batch_queue.get()

    def close(self):
        # quit process thread
        self.process_running = False
        while not self.batch_queue.empty():
            self.batch_queue.get()
        self.process_thread.join()

        # quit collector threads
        self.collector_running = False
        while not self.transition_queue.empty():
            self.transition_queue.get()
        for t in self.collector_threads:
            t.join()


class SyncEnvPool(object):
    def __init__(self, config, agent):

        args = config.args
        env_num = args.env_nums
        self.agent = agent
        self.config = config

        def collector_loop(cmd_queue, result_queue):
            env = make_env(config)
            while True:
                (cmd, cmd_data) = cmd_queue.get()
                if cmd == 'spec':
                    result_queue.put((env.observation_spec(), env.action_spec()))
                elif cmd == 'step':
                    result_queue.put(env.step([cmd_data], True))
                elif cmd == 'reset':
                    result_queue.put(env.reset())
                elif cmd == 'close':
                    env.close()
                    break
                elif cmd == 'save_replay':
                    env.save_replay(cmd_data)
                else:
                    raise NotImplementedError

        collector_threads = []
        cmd_queues = []
        result_queues = []
        for _ in range(env_num):
            cmd_queue = queue.Queue()
            result_queue = queue.Queue()
            t = threading.Thread(target=collector_loop,
                                 args=(cmd_queue, result_queue))
            t.daemon = True
            t.start()
            cmd_queues.append(cmd_queue)
            result_queues.append(result_queue)
            collector_threads.append(t)

        self.cmd_queues = cmd_queues
        self.result_queues = result_queues
        self.collector_threads = collector_threads

        _, self.o, *_ = self._reset()

    def get_batch(self, *args):
        # step = self.config.args.step

        step = 2
        os = [None] * step
        as_ = [None] * step
        rs = [None] * step
        ds = [None] * step

        for i in range(step):
            acts = self.agent.step(self.o)
            os[i] = self.o
            as_[i] = acts
            _, o, r, d = self._step(acts)
            rs[i] = r
            ds[i] = d
            self.o = o

        last_value = np.array(self.agent.get_value(self.o))
        returns = [None] * (step + 1)
        returns[-1] = last_value
        for i in reversed(range(step)):
            returns[i] = rs[i] + 0.99 * (1 - ds[i]) * returns[i + 1]
        returns = returns[:-1]

        acts = []
        for dim in range(len(as_[0])):
            acts.append(np.concatenate([as_[i][dim] for i in range(step)]))
        # acts = transform_scalar_2_scalar(acts, self.config)
        obs = []
        for dim in range(len(os[0])):
            obs.append(np.concatenate([os[i][dim] for i in range(step)]))
        return obs, acts, np.concatenate(returns)

    def _spec(self):
        for cmd_queue in self.cmd_queues:
            cmd_queue.put(('spec', None))

        results = [result_queue.get() for result_queue in self.result_queues]
        return results[0]

    def _step(self, actions):
        length = len(actions[0])
        dim = len(actions)
        actions = [np.split(d, length) for d in actions]
        acts = []
        for i in range(length):
            indices = [actions[d][i] for d in range(dim)]
            acts.append(indices)

        for cmd_queue, action in zip(self.cmd_queues, acts):
            cmd_queue.put(('step', action))
        results = [result_queue.get() for result_queue in self.result_queues]
        return self._group(results)

    def _reset(self):
        for cmd_queue in self.cmd_queues:
            cmd_queue.put(('reset', None))
        results = [result_queue.get() for result_queue in self.result_queues]
        return self._group(results)

    def _group(self, results):
        rets = [[], [], [], []]
        for res in results:
            for c, e in zip(rets, res):
                c.append(e)
        rets[1] = process_observation_list(rets[1])
        rets[2] = np.array(rets[2])
        rets[3] = np.array(rets[3])
        return rets

    def _close(self):
        for cmd_queue in self.cmd_queues:
            cmd_queue.put(('close', None))
        for t in self.collector_threads:
            t.join()

    def _save_replay(self, replay_dir='PySC2Replays'):
        self.cmd_queues[0].put(('save_replay', replay_dir))
