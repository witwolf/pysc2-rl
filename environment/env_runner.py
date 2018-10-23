# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/19.
#

import logging


class EnvRunner(object):
    def __init__(self, env,
                 agent,
                 observation_adapter=None,
                 action_adapter=None,
                 epoch_n=None,
                 batch_n=None,
                 step_n=16,
                 test_after_epoch=True):
        self._env = env
        self._agent = agent
        self._obs_adapter = observation_adapter,
        self._act_adapter = action_adapter
        self._epoch_n = epoch_n
        self._batch_n = batch_n
        self._step_n = step_n
        self._test_after_epoch = test_after_epoch
        self._obs = None
        pass

    def run(self, *args, **kwargs):
        # train epochs and test after every epoch
        for epoch in self._epoch_n:
            self._obs = self._env.reset()
            for batch in self._batch_n:
                logging.info("epoch:%d,batch:%d" % (epoch, batch))
                self._batch()
            if self._test_after_epoch:
                total_reward = self._test()
                logging.info("epoch:%d, test total reward:%d" % (epoch, total_reward))
            self._agent.save()

    def _batch(self):
        # train batch
        batch = []
        for _ in range(self._step_n):
            actions = self._agent.step(self._obs)
            self._obs = self._env.step(actions)
        self._agent.update(batch)

    def _test(self):
        obs = self._obs
        if isinstance(obs, list):
            obs = obs[:1]

        def is_terminate(o):
            if isinstance(o, list):
                return o[0][0].last()
            return o[0].last()

        def get_reward(o):
            if isinstance(o):
                return o[0][0].reward
            else:
                return o[0].reward

        total_reward = 0
        while not is_terminate(obs):
            actions = self._agent.step(obs)
            obs = self._env.step(actions)
            total_reward += get_reward(obs)
        return total_reward
