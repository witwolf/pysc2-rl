# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/19.
#

import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.training.summary_io import SummaryWriterCache


class EnvRunner(object):
    def __init__(self, env,
                 agent,
                 observation_adapter=None,
                 action_adapter=None,
                 epoch_n=None,
                 batch_n=None,
                 step_n=16,
                 test_after_epoch=True,
                 train=True,
                 logdir=None):
        self._env = env
        self._agent = agent
        self._obs_adapter = observation_adapter
        self._act_adapter = action_adapter
        self._epoch_n = epoch_n
        self._batch_n = batch_n
        self._step_n = step_n
        self._test_after_epoch = test_after_epoch
        self._obs = None
        self._train = train
        self._summary_writer = None
        if logdir:
            self._summary_writer = SummaryWriterCache.get(logdir)

    def run(self, *args, **kwargs):
        # train epochs and test after every epoch

        for epoch in range(self._epoch_n):
            self._agent.reset()
            self._obs = self._env.reset()
            if self._train:
                for batch in range(self._batch_n):
                    logging.info("epoch:%d,batch:%d" % (epoch, batch))
                    self._batch()
                self._agent.save()
            if self._test_after_epoch:
                total_reward = self._test()
                logging.info("epoch:%d,reward:%d" % (epoch, total_reward))
                if self._train:
                    self._record(step=epoch, reward=total_reward)

    def _batch(self):
        # train batch
        obs = []
        func_calls = []
        for _ in range(self._step_n):
            obs.append(self._obs)
            # action = self._agent.step(self._obs)
            # if not actions:
            #     actions = [[] for _ in range(len(action))]
            # for c, e in zip(actions, action):
            #     c.append(e)
            # function_calls = self._act_adapter.reverse(action)
            # self._obs = self._env.step(function_calls)
            function_calls = self._agent.step(self._obs)
            func_calls.extend(function_calls)
            self._obs = self._env.step([(f,) for f in function_calls])
        obs.append(self._obs)

        (states, next_states,
         rewards, dones, timestamps) = self._obs_adapter.transform(obs)

        actions = self._act_adapter.transform(func_calls)
        batch = (states,
                 [np.concatenate(c, axis=0) for c in actions],
                 next_states,
                 rewards,
                 dones,
                 timestamps)
        summary, step = self._agent.update(*batch)
        self._record(summary=summary, step=step)

    def _test(self):
        obs = self._obs
        if isinstance(obs, list):
            obs = obs[:1]

        def is_terminate(o):
            if isinstance(o, list):
                return o[0].last()
            return o.last()

        def get_reward(o):
            if isinstance(o, list):
                return o[0].reward
            else:
                return o.reward

        total_reward = 0
        while not is_terminate(obs):
            state = self._obs_adapter.transform(obs, state_only=True)
            action = self._agent.step(state=state, evaluate=True)
            function_calls = self._act_adapter.reverse(action)
            obs = self._env.step([(f,) for f in function_calls])
            total_reward += get_reward(obs)
        return total_reward

    def _record(self, step=None, summary=None, **kwargs):
        if self._summary_writer is None:
            return
        if summary:
            self._summary_writer.add_summary(summary, step)
        for tag, value in kwargs.items():
            summary = tf.Summary()
            summary.value.add(tag=tag, simple_value=np.mean(value))
            self._summary_writer.add_summary(summary, step)
