# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/20.
#

from pysc2.tests import utils
from absl.testing import absltest

import tensorflow as tf
from tensorflow.contrib import layers
import random
import numpy as np
from utils.network import Utils
import logging


class TestNetworkUtils(utils.TestCase):
    def test_td_value(self):
        state_dim = random.randint(4, 8)
        state_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        value = layers.fully_connected(state_ph, 1)
        value = tf.squeeze(value, axis=1)
        rewards_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        dones_ph = tf.placeholder(dtype=tf.int32, shape=[None])

        discount_ph = tf.placeholder(dtype=tf.float32, shape=())
        td_value = Utils.tf_td_value(rewards_ph, dones_ph,
                                     value, discount_ph)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        test_times = 100
        for _ in range(test_times):
            td_step = random.randint(1, 4) * 8
            parallel_num = random.randint(1, 4) * 8
            discount = random.random()
            batch_size = parallel_num * td_step

            last_state = np.random.random(size=(parallel_num, state_dim))
            rewards = np.random.random(size=(batch_size))
            dones = np.random.randint(0, 2, size=(batch_size))

            value_, td_value_ = sess.run([value, td_value], feed_dict={
                state_ph: last_state,
                dones_ph: dones,
                rewards_ph: rewards,
                discount_ph: discount
            })

            # cal td value manually
            last_value = value_[-parallel_num:]
            td_value__ = Utils.td_value(rewards, dones, last_value, discount)

            np.testing.assert_almost_equal(td_value_, td_value__, decimal=4)


if __name__ == "__main__":
    absltest.main()
