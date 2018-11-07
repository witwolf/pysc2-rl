# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/19.
#

import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.training.summary_io import SummaryWriterCache


class EnvRunner(object):
    def __init__(self, env=None,
                 test_env=None,
                 agent=None,
                 observation_adapter=None,
                 action_adapter=None,
                 reward_adapter=None,
                 epoch_n=None,
                 batch_n=None,
                 step_n=16,
                 test_after_epoch=True,
                 train=True,
                 logdir=None):
        self._env = env
        self._test_env = test_env
        self._agent = agent
        self._obs_adapter = observation_adapter
        self._act_adapter = action_adapter
        self._reward_adapter = reward_adapter
        self._epoch_n = epoch_n
        self._batch_n = batch_n
        self._step_n = step_n
        self._test_after_epoch = test_after_epoch
        self._obs = None
        self._train = train
        self._summary_writer = None
        self._logdir = logdir
        self._summary_writer = SummaryWriterCache.get(logdir)

    def run(self, *args, **kwargs):
        if self._train:
            self._obs = self._env.reset()
        for epoch in range(self._epoch_n):
            if self._train:
                for batch in range(self._batch_n):
                    logging.info("epoch:%d,batch:%d" % (epoch, batch))
                    self._batch()
            if self._test_after_epoch and self._test_env:
                self._test(epoch)

    def _batch(self):
        # train batch
        states, actions, next_states = [], [], []
        dones, rewards, timestamps = [], [], []
        ss, *_ = self._obs_adapter.transform(self._obs)
        for _ in range(self._step_n):
            states.append(ss)
            timestamps.extend(self._obs)
            acts = self._agent.step(state=ss, obs=self._obs)
            if not actions:
                actions = [[] for _ in range(len(acts))]
            for c, e in zip(actions, acts):
                c.append(e)
            func_calls = self._act_adapter.reverse(acts)
            self._obs = self._env.step([(f,) for f in func_calls])
            ss, rs, ds, _ = self._obs_adapter.transform(self._obs)
            if self._reward_adapter:
                rs = self._reward_adapter.transform(self._obs, func_calls)
            next_states.append(ss)
            rewards.append(rs)
            dones.append(ds)
        states = self._flatten_state(states)
        next_states = self._flatten_state(next_states)
        batch = (states, [np.concatenate(c, axis=0) for c in actions],
                 next_states, np.concatenate(rewards, axis=0),
                 np.concatenate(dones, axis=0), timestamps)
        summary, step = self._agent.update(*batch)
        self._record(summary=summary, step=step)

    def _flatten_state(self, states):
        columns = len(states[0])
        rets = [[] for _ in range(columns)]
        for row in states:
            for c in range(columns):
                rets[c].append(row[c])
        rets = [np.concatenate(c, axis=0) for c in rets]
        return rets

    def _test(self, epoch):
        obs = self._test_env.reset()
        sparse_reward = 0
        dense_reward = 0
        while not obs[0].last():
            state, *_ = self._obs_adapter.transform(obs)
            action = self._agent.step(state=state, evaluate=True)
            function_calls = self._act_adapter.reverse(action)
            rewards = self._reward_adapter.transform(obs, function_calls)
            obs = self._test_env.step([(f,) for f in function_calls])
            if self._reward_adapter:
                dense_reward += self._reward_adapter.transform(
                    obs, function_calls)[0]
            sparse_reward += rewards[0]
        records = {
            'reward': sparse_reward,
            'dense_reward': dense_reward}
        record_fields = [
            'score',
            'total_value_units',
            'total_value_structures',
            'killed_value_units',
            'killed_value_structures',
            'collected_minerals',
            'collected_vespene',
            'spent_minerals',
            'spent_vespene']
        for f in record_fields:
            records[f] = obs[0].observation.score_cumulative[f]
        self._record(step=epoch, **records)

    def _record(self, step=None, summary=None, **kwargs):
        if not self._train or not self._summary_writer:
            return
        if summary:
            self._summary_writer.add_summary(summary, step)
        for tag, value in kwargs.items():
            summary = tf.Summary()
            summary.value.add(tag=tag, simple_value=np.mean(value))
            self._summary_writer.add_summary(summary, step)


class EnvRunner2(EnvRunner):
    def _batch(self):
        super()._batch()
