# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/19.
#

import tensorflow as tf


class Utils(object):
    @staticmethod
    def td_value(rewards, dones, last_value, discount):
        batch_size = tf.shape(rewards)[0]
        parallel_num = tf.shape(last_value)[0]
        td_step = tf.div(batch_size, parallel_num)

        def cond(i, value, last_value):
            return i >= 0

        def body(i, value, last_value):
            begin = i * parallel_num
            end = begin + parallel_num
            rs = rewards[begin:end]
            ds = dones[begin:end]
            _value = rs + tf.to_float(1 - ds) * last_value * discount
            last_value = _value
            value = tf.concat([_value, value], axis=0)
            return i - 1, value, last_value

        value = last_value
        _, value, _ = tf.while_loop(cond, body, [td_step - 1, value, last_value])
        return value[:-parallel_num]
