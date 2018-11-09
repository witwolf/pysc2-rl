# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/19.
#

import tensorflow as tf
import numpy as np


class Utils(object):
    @staticmethod
    def tf_td_value(rewards, dones, last_value, discount):
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

    @staticmethod
    def td_value(rewards, dones, last_value, discount):
        parallel_num = len(last_value)
        td_step = len(rewards) // parallel_num

        values = [None] * (td_step + 1)
        values[-1] = last_value
        for i in reversed(range(td_step)):
            s = i * parallel_num
            e = s + parallel_num
            values[i] = rewards[s:e] + (
                    discount * (1 - dones[s:e]) * values[i + 1])

        values = values[:-1]
        return np.concatenate(values, axis=0)

    @staticmethod
    def clip_log(input, min=0.0, max=1.0):
        tf.clip_by_value(input, min, max)
