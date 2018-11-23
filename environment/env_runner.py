# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/19.
#

import logging
import numpy as np
import tensorflow as tf
import queue
from tensorflow.python.training.summary_io import SummaryWriterCache


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
        states, actions, next_states = [], [], []
        dones, rewards, timestamps = [], [], []
        ss, *_ = self._obs_adapter.transform(self._obs)
        for step_i in range(self._step_n):
            states.append(ss)
            timestamps.extend(self._obs)
            acts_or_funcs = self._agent.step(
                state=ss, obs=self._obs, evaluate=not self._train)
            acts, func_calls = self._acts_funcs(acts_or_funcs)
            if not actions:
                actions = [[] for _ in range(len(acts))]
            for c, e in zip(actions, acts):
                c.append(e)
            self._obs = self._env.step([(f,) for f in func_calls])
            ss, rs, ds, _ = self._obs_adapter.transform(self._obs)
            if self._reward_adapter:
                rs = self._reward_adapter.transform(self._obs, func_calls)
            self._record(rs, ds, self._obs)
            next_states.append(ss)
            rewards.append(rs)
            dones.append(ds)
        if self._train:
            states = self._flatten_state(states)
            next_states = self._flatten_state(next_states)
            batch = (states, [np.concatenate(c, axis=0) for c in actions],
                     next_states, np.concatenate(rewards, axis=0),
                     np.concatenate(dones, axis=0), timestamps)
            summary, step = self._agent.update(*batch)
            self._summary(summary=summary, step=step)

    def _acts_funcs(self, acts_or_funcs):
        if isinstance(acts_or_funcs[0], tuple):
            funcs = acts_or_funcs
            acts = self._act_adapter.transform(funcs)
        else:
            acts = acts_or_funcs
            funcs = self._act_adapter.reverse(acts)
        return acts, funcs

    def _flatten_state(self, states):
        columns = len(states[0])
        rets = [[] for _ in range(columns)]
        for row in states:
            for c in range(columns):
                rets[c].append(row[c])
        rets = [np.concatenate(c, axis=0) for c in rets]
        return rets

    def _record(self, rewards, dones, obs):
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
