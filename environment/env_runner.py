# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/19.
#

import logging
import numpy as np
import os
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
            if self._test_after_epoch:
                total_reward = self._test()
                logging.info("epoch:%d,reward:%d" % (epoch, total_reward))
                self._record(step=epoch, reward=total_reward)

    def _batch(self):
        # train batch
        ## todo refactor
        obs = []
        func_calls = []
        rewards = []
        acts = None
        for _ in range(self._step_n):
            obs.append(self._obs)
            state = self._obs_adapter.transform(self._obs, state_only=True)
            funcs_or_acts = self._agent.step(state=state, obs=self._obs)
            if isinstance(funcs_or_acts[0], tuple):
                function_calls = funcs_or_acts
                func_calls.extend(function_calls)
            else:
                if not acts:
                    acts = [[] for _ in range(len(funcs_or_acts))]
                for c, e in zip(acts, funcs_or_acts):
                    c.append(e)
                function_calls = self._act_adapter.reverse(funcs_or_acts)
            reward = self._reward_adapter.transform(self._obs, function_calls)
            rewards.append(reward)
            self._obs = self._env.step([(f,) for f in function_calls])
        obs.append(self._obs)

        (states, next_states,
         _, dones, timestamps) = self._obs_adapter.transform(obs)

        if not acts:
            acts = self._act_adapter.transform(func_calls)
        else:
            acts = [np.concatenate(c, axis=0) for c in acts]
        batch = (states, acts,
                 next_states, rewards, dones, timestamps)
        summary, step = self._agent.update(*batch)
        self._record(summary=summary, step=step)

    def _test(self):
        obs = self._test_env.reset()
        total_reward = 0
        while not obs[0].last():
            state = self._obs_adapter.transform(obs, state_only=True)
            action = self._agent.step(state=state, evaluate=True)
            function_calls = self._act_adapter.reverse(action)
            rewards = self._reward_adapter.transform(obs, function_calls)
            obs = self._test_env.step([(f,) for f in function_calls])
            total_reward += rewards[0]
        return total_reward

    def _record(self, step=None, summary=None, **kwargs):
        if not self._train or not self._summary_writer:
            return
        if summary:
            self._summary_writer.add_summary(summary, step)
        for tag, value in kwargs.items():
            summary = tf.Summary()
            summary.value.add(tag=tag, simple_value=np.mean(value))
            self._summary_writer.add_summary(summary, step)
