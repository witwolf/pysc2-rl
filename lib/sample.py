# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/12/1.
#


import numpy as np


# TODO common abstract for sampler

class Trajectory(object):
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

    def add_transition(self, state, action,
                       next_state, reward, done):
        self._states.append(state)
        if not self._actions:
            self._actions = [[] for _ in range(len(action))]
        for c, e in zip(self._actions, action):
            c.append(e)
        self._next_states.append(next_state)
        self._rewards.append(reward)
        self._dones.append(done)

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

    def __call__(self, *args, **kwargs):
        states = Trajectory._flatten_state(self._states)
        next_states = Trajectory._flatten_state(self._next_states)
        batch = (
            states,
            [np.concatenate(c, axis=0) for c in self._actions],
            next_states, np.concatenate(self._rewards, axis=0),
            np.concatenate(self._dones, axis=0))
        return batch

    def __len__(self):
        return len(self._states)
