# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/19.
#

import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.training.summary_io import SummaryWriterCache
from utils.utility import Utility
from lib.sample import Trajectory


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
        self._env_num = None
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
        self._env_num = len(self._obs)
        self._terminate_obs = [[] for _ in range(self._env_num)]
        for epoch in range(self._epoch_n):
            logging.warning("epoch:%d" % epoch)
            self._batch()

    def _batch(self):
        # train batch
        traj = Trajectory()
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
            traj.add_transition(ss, acts, ns, rs, ds)
            ss = ns
            self._obs = next_obs
            self._record(self._obs)
            self._check_env_reset()
        if self._train:
            summary, step = self._agent.update(*traj())
            self._summary(summary=summary, step=step)

    def _check_env_reset(self):
        terminates = [
            i for i in range(self._env_num)
            if self._obs[i].last()]
        for i, o in zip(terminates, self._env.reset(terminates)):
            self._obs[i] = o

    def _acts_funcs(self, acts_or_funcs, act_adapter=None):
        if isinstance(acts_or_funcs[0], tuple):
            funcs = acts_or_funcs
            acts = act_adapter.transform(funcs)
        else:
            acts = acts_or_funcs
            funcs = act_adapter.reverse(acts)
        return acts, funcs

    def _record(self, obs):
        for ts, o in zip(self._terminate_obs, obs):
            if o.last():
                ts.append(o)
        for ts in self._terminate_obs:
            if len(ts) == 0:
                return
        records = {
            'score': 0.0,
            'collected_minerals': 0.0, 'spent_minerals': 0.0,
            'collected_vespene': 0.0, 'spent_vespene': 0.0,
            'win': 0.0, 'lost': 0.0, 'tie': 0.0}

        for i in range(self._env_num):
            ts = self._terminate_obs[i]
            for f in records.keys():
                records[f] += ts[0].observation.score_cumulative[f]
                records[['tie', 'win', 'lost'][int(ts[0].reward)]] += 1
            self._terminate_obs[i] = ts[1:]
        for f in records.keys():
            records[f] = records[f] / self._env_num
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
                 step_n=16,
                 step_n2=16,
                 train=True,
                 logdir=None):
        super().__init__(
            env, agent, None, None, None,
            epoch_n, step_n, train, logdir)
        self._step_n2 = step_n2
        self._main_p_obs_adpt = main_policy_obs_adpt
        self._sub_p_obs_adpts = sub_policy_obs_adpts
        self._sub_p_act_adpts = sub_policy_act_adpts
        self._sub_p_rwd_adpts = sub_policy_rwd_adapts
        self._policy_steps = None

    def _batch(self):
        traj = Trajectory()
        for step_i in range(self._step_n):
            ss, *_ = self._main_p_obs_adpt.transform(self._obs)
            act = self._agent.step(ss, 0)
            rs = self._batch2(act)
            ns, _, ds, _ = self._main_p_obs_adpt.transform(self._obs)
            traj.add_transition(ss, act, ns, rs, ds)
            self._record(self._obs)
            self._check_env_reset()
        if self._train:
            summary, step = self._agent.update(*traj(), 0)
            self._summary(summary=summary, step=step)

    def _batch2(self, act):
        sub_p_num = len(self._sub_p_act_adpts)
        rewards = np.zeros((self._env_num,), dtype=np.float32)
        policies = np.zeros((self._env_num, sub_p_num))
        policies[..., act] = 1
        for i in range(sub_p_num):
            indices = policies[..., i].nonzero()[0]
            obs_adapter = self._sub_p_obs_adpts[i]
            act_adapter = self._sub_p_act_adpts[i]
            rwd_adapter = self._sub_p_rwd_adpts[i]
            traj = Trajectory()
            for _ in range(self._step_n2):
                obs = Utility.select(self._obs, indices)
                # if the env terminates, it needs a new main policy
                # drop this env's traj
                _indices = [j for j, o in enumerate(obs) if not o.last()]
                if len(_indices) != len(indices):
                    traj.select(_indices)
                    indices = indices[_indices]
                    obs = Utility.select(self._obs, indices)
                if len(indices) == 0:
                    break
                ss, *_ = obs_adapter.transform(obs)
                acts = self._agent.step(ss, i + 1)
                func_calls = act_adapter.reverse(acts)
                next_obs = self._env.step([(f,) for f in func_calls], indices)
                ns, _, ds, _ = obs_adapter.transform(next_obs)
                rs = rwd_adapter.transform(next_obs, func_calls)
                traj.add_transition(ss, acts, ns, rs, ds)
                rewards[indices] += rs
                obs = next_obs
                for j, o in zip(indices, obs):
                    self._obs[j] = o
            if self._train and len(indices) != 0:
                summary, step = self._agent.update(*traj(), i + 1)
                self._summary(summary=summary, step=step)
        return rewards
