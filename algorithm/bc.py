# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#


import tensorflow as tf
from tensorflow.contrib import layers

from pysc2.agents.base_agent import BaseAgent
from lib.base import BaseDeepAgent
from utils.network import Utils


class BehaviorClone(BaseDeepAgent, BaseAgent):

    def __init__(self,
                 sess=None,
                 network=None,
                 network_creator=None,
                 td_step=16,
                 script_agents=None,
                 lr=1e-3,
                 v_coef=0.01,
                 discount=0.99,
                 **kwargs):

        if bool(network) == bool(network_creator):
            raise Exception()

        self._sess = sess
        self._network = network
        self._network_creator = network_creator
        self._td_step = td_step
        self._script_agents = script_agents
        self._lr = lr
        self._v_coef = v_coef
        self._discount = discount

        super().__init__(sess, **kwargs)

        self._sess.run(tf.global_variables_initializer())

    def init_network(self, **kwargs):
        if not self._network:
            self._network = self._network_creator(**kwargs)
        inputs = self._network.inputs
        self._state_input = inputs['state']
        self._action_input = inputs['action']
        self._reward_input = inputs['reward']
        self._done_input = inputs['done']
        self._value_input = inputs['value']
        self._value = self._network['value']
        self._policies = self._network['policies']
        self._actions = self._network['actions']

    def init_updater(self, **kwargs):
        policy_loss = 0.
        for label, y in zip(self._action_input, self._policies):
            one_hot = tf.one_hot(label, depth=y.shape[-1])
            clipped_y = tf.clip_by_value(y, 1e-6, 1.0)
            cross_entropy = -tf.reduce_sum(
                one_hot * tf.log(clipped_y), axis=1)
            policy_loss += cross_entropy

        policy_loss = tf.reduce_mean(policy_loss)

        target_value = Utils.td_value(
            self._reward_input, self._done_input,
            self._value_input, self._discount)
        target_value = tf.stop_gradient(target_value)

        value_loss = self._v_coef * tf.reduce_mean(
            tf.square(target_value - self._value))
        loss = policy_loss + value_loss

        summary = tf.summary.merge([
            tf.summary.scalar('bc/value', tf.reduce_mean(self._value)),
            tf.summary.scalar('bc/target_value', tf.reduce_mean(target_value)),
            tf.summary.scalar('bc/policy_loss', policy_loss),
            tf.summary.scalar('bc/value_loss', value_loss),
            tf.summary.scalar('bc/loss', loss)])

        opt = tf.train.RMSPropOptimizer(
            learning_rate=self._lr, decay=0.99, epsilon=1e-5)
        step = tf.Variable(0, trainable=False)
        train_op = layers.optimize_loss(
            loss=loss, optimizer=opt,
            learning_rate=None, global_step=step, clip_gradients=1.0)

        # todo
        self._train_op, self._step, self._summary = train_op, step, summary
        return None

    def step(self, obs=None, state=None, evaluate=False):
        if not evaluate:
            return self._script_agents.step(obs)
        actions = self._sess.run(
            self._actions,
            feed_dict=dict(zip(self._state_input, state)))
        return actions

    def update(self, states, actions,
               next_states, rewards, dones, *args):
        begin = -len(dones) // self._td_step
        last_state = [e[begin:] for e in next_states]
        value = self._sess.run(self._value, feed_dict=dict(
            zip(self._state_input, last_state)))

        input_vars = self._state_input + self._action_input + [
            self._reward_input, self._done_input, self._value_input]
        input_vals = states + actions + [rewards, dones, value]
        feed_dict = dict(zip(input_vars, input_vals))

        _, step, summary = self._sess.run(
            [self._train_op, self._step, self._summary],
            feed_dict=feed_dict)

        return summary, step

    def reset(self):
        self._script_agents.reset()
