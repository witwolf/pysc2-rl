# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/19.
#

import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.training.summary_io import SummaryWriterCache
from utils.utility import Utility


class Sample():
    @staticmethod
    def _flatten_state(states):
        columns = len(states[0])
        rets = [[] for _ in range(columns)]
        for row in states:
            for c in range(columns):
                rets[c].append(row[c])
        rets = [np.concatenate(c, axis=0) for c in rets]
        return rets

    def __init__(self):
        self._states = []
        self._actions = []
        self._next_states = []
        self._rewards = []
        self._dones = []
        self._timestamps = []

    def append(self, state, action,
               next_state, reward, done, timestamp):
        self._states.append(state)
        if not self._actions:
            self._actions = [[] for _ in range(len(action))]
        for c, e in zip(self._actions, action):
            c.append(e)
        self._next_states.append(next_state)
        self._rewards.append(reward)
        self._dones.append(done)
        self._timestamps.append(timestamp)

    def select(self, indices):
        self._states = [
            [c[indices] for c in r] for r in self._states]
        self._actions = [
            [c[indices] for c in r] for r in self._actions]
        self._next_states = [
            [c[indices] for c in r] for r in self._next_states]
        self._rewards = [
            r[indices] for r in self._rewards]
        self._dones = [
            r[indices] for r in self._dones]
        self._timestamps = [
            Utility.select(r, indices) for r in self._timestamps]

    def __call__(self, *args, **kwargs):
        states = Sample._flatten_state(self._states)
        next_states = Sample._flatten_state(self._next_states)
        batch = (
            states,
            [np.concatenate(c, axis=0) for c in self._actions],
            next_states, np.concatenate(self._rewards, axis=0),
            np.concatenate(self._dones, axis=0),
            self._timestamps)
        return batch

    def __len__(self):
        return len(self._states)


class EnvRunner(object):
    def __init__(self, env=None,
                 agent=None,
                 observation_adapter=None,
                 action_adapter=None,
                 reward_adapter=None,
                 epoch_n=None,
                 step_n=16,
                 train=True,
                 logdir=None):
        self._env = env
        self._agent = agent
        self._obs_adapter = observation_adapter
        self._act_adapter = action_adapter
        self._reward_adapter = reward_adapter
        self._epoch_n = epoch_n
        self._step_n = step_n
        self._obs = None
        self._train = train
        self._summary_writer = None
        self._logdir = logdir
        self._episode = 0
        self._terminate_obs = None
        self._summary_writer = SummaryWriterCache.get(logdir)

    def run(self, *args, **kwargs):
        self._obs = self._env.reset()
        self._terminate_obs = [[] for _ in range(len(self._obs))]
        for epoch in range(self._epoch_n):
            logging.warning("epoch:%d" % epoch)
            self._batch()

    def _batch(self):
        # train batch
        cache = Sample()
        ss, *_ = self._obs_adapter.transform(self._obs)
        for step_i in range(self._step_n):
            obs = self._obs
            acts_or_funcs = self._agent.step(
                state=ss, obs=obs, evaluate=not self._train)
            acts, func_calls = self._acts_funcs(
                acts_or_funcs, self._act_adapter)
            next_obs = self._env.step([(f,) for f in func_calls])
            ns, rs, ds, _ = self._obs_adapter.transform(next_obs)
            if self._reward_adapter:
                rs = self._reward_adapter.transform(next_obs, func_calls)
            cache.append(ss, acts, ns, rs, ds, obs)
            ss = ns
            self._obs = next_obs
            self._record(ds, self._obs)
            resets = [i for i in range(len(ds)) if ds[i]]
            for i, o in zip(resets, self._env.reset(resets)):
                self._obs[i] = o
        if self._train:
            batch = cache()
            summary, step = self._agent.update(*batch)
            self._summary(summary=summary, step=step)

    def _acts_funcs(self, acts_or_funcs, act_adapter=None):
        if isinstance(acts_or_funcs[0], tuple):
            funcs = acts_or_funcs
            acts = act_adapter.transform(funcs)
        else:
            acts = acts_or_funcs
            funcs = act_adapter.reverse(acts)
        return acts, funcs

    def _record(self, dones, obs):
        for i, done in enumerate(dones):
            if done:
                self._terminate_obs[i].append(obs[i])
        for ts in self._terminate_obs:
            if len(ts) == 0:
                return
        records = {
            'score': 0.0,
            'total_value_units': 0.0,
            'total_value_structures': 0.0,
            'killed_value_units': 0.0,
            'killed_value_structures': 0.0,
            'collected_minerals': 0.0,
            'collected_vespene': 0.0,
            'spent_minerals': 0.0,
            'spent_vespene': 0.0}
        win = 0.0
        for i in range(len(self._terminate_obs)):
            ts = self._terminate_obs[i]
            for f in records.keys():
                records[f] += ts[0].observation.score_cumulative[f]
            win += int(ts[0].reward == 1)
            self._terminate_obs[i] = ts[1:]

        for f in records.keys():
            records[f] = records[f] / len(self._terminate_obs)
        records['win_rate'] = win / len(self._terminate_obs)
        logging.warning("episode %d: %s", self._episode, records)
        self._summary(step=self._episode, **records)
        self._episode += 1

    def _summary(self, step=None, summary=None, **kwargs):
        if not self._train or not self._summary_writer:
            return
        if summary:
            self._summary_writer.add_summary(summary, step)
        for tag, value in kwargs.items():
            summary = tf.Summary()
            summary.value.add(tag=tag, simple_value=np.mean(value))
            self._summary_writer.add_summary(summary, step)


class EnvRunner2(EnvRunner):

    def __init__(self, env=None,
                 agent=None,
                 main_policy_obs_adpt=None,
                 sub_policy_obs_adpts=None,
                 sub_policy_act_adpts=None,
                 sub_policy_rwd_adapts=None,
                 epoch_n=None,
                 K=8,
                 step_n=16,
                 train=True,
                 logdir=None):
        super().__init__(
            env, agent, None, None, None,
            epoch_n, step_n, train, logdir)
        self._K = K
        self._main_p_obs_adpt = main_policy_obs_adpt
        self._sub_p_obs_adpts = sub_policy_obs_adpts
        self._sub_p_act_adpts = sub_policy_act_adpts
        self._sub_p_rwd_adpts = sub_policy_rwd_adapts
        self._policy_steps = None

    def run(self, *args, **kwargs):
        self._obs = self._env.reset()
        env_num = len(self._obs)
        self._terminate_obs = [[] for _ in range(env_num)]
        for epoch in range(self._epoch_n):
            logging.warning("epoch:%d" % epoch)
            self._batch()

    def _batch(self):
        cache = Sample()
        for step_i in range(self._step_n):
            obs = self._obs
            ss, *_ = self._main_p_obs_adpt.transform(self._obs)
            act = self._agent.step(ss, 0)
            rs = self._step_k(act)
            ns, _, ds, _ = self._main_p_obs_adpt.transform(self._obs)
            cache.append(ss, act, ns, rs, ds, obs)
            self._record(ds, self._obs)
            resets = [j for j in range(len(ds)) if ds[j]]
            for j, o in zip(resets, self._env.reset(resets)):
                self._obs[j] = o
        if self._train:
            batch = cache()[:5]
            summary, step = self._agent.update(*batch, 0)
            self._summary(summary=summary, step=step)

    def _step_k(self, act):
        env_num = len(self._obs)
        sub_p_num = len(self._sub_p_act_adpts)
        rewards = np.zeros((env_num,), dtype=np.float32)
        policies = np.zeros((env_num, sub_p_num))
        policies[..., act] = 1
        for i in range(sub_p_num):
            indices = policies[..., i].nonzero()[0]
            if len(indices) == 0:
                continue
            obs_adapter = self._sub_p_obs_adpts[i]
            act_adapter = self._sub_p_act_adpts[i]
            rwd_adapter = self._sub_p_rwd_adpts[i]
            cache = Sample()
            for _ in range(self._K):
                if len(indices) == 0:
                    break
                obs = Utility.select(self._obs, indices)
                ss, *_ = obs_adapter.transform(obs)
                acts = self._agent.step(ss, i + 1)
                func_calls = act_adapter.reverse(acts)
                next_obs = self._env.step([(f,) for f in func_calls], indices)
                ns, rs, ds, _ = obs_adapter.transform(next_obs)
                rs = rwd_adapter.transform(next_obs, func_calls)
                cache.append(ss, acts, ns, rs, ds, obs)
                rewards[indices] += rs
                obs = next_obs
                for j, o in zip(indices, obs):
                    self._obs[j] = o
                _indices = [j for j, o in enumerate(obs) if not o.last()]
                if len(_indices) != len(indices):
                    cache.select(_indices)
                    indices = indices[_indices]
            if len(indices) == 0:
                logging.warning("all terminated, drop this fragment")
                continue
            if self._train:
                batch = cache()[:5]
                summary, step = self._agent.update(*batch, i + 1)
                self._summary(summary=summary, step=step)
        return rewards
