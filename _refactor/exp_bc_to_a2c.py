# -*- coding: utf-8 -*-

import tensorflow as tf
from lib.config import Config
from _refactor.envs import EnvPool
from _refactor.envs import SyncEnvPool
from _refactor.envs import make_env
from _refactor.model import Model

from absl import flags
import sys

FLAGS = flags.FLAGS
flags.DEFINE_string('map', 'MoveToBeacon', 'Map name')
flags.DEFINE_integer('screen_size', 64, 'Feature screen resolution')
flags.DEFINE_integer('minimap_size', 64, 'Feature minimap resolution')
flags.DEFINE_integer('step_mul', 8, 'How many game steps per algorithm step')
flags.DEFINE_bool('render', False, "Whether to render with pygame.")
flags.DEFINE_integer('env_nums', 16, "Env parallel number")
flags.DEFINE_float('lr', 1e-4, "Learning rate")
flags.DEFINE_integer('batch_size', 64, 'Batch  size')
flags.DEFINE_integer('imitation_batch', 4096, 'Batch num')
flags.DEFINE_integer('reinforce_batch', 4096, 'Batch num')
flags.DEFINE_integer('step', 16, 'steps')
flags.DEFINE_integer('buffer_size', 4096, 'buffer size')
flags.DEFINE_integer('test_interval', 256, 'Test interval (after n batch update)')
flags.DEFINE_string('logdir', 'logs', 'logdir')
flags.DEFINE_integer('save_interval', 256, 'Save snapshot interval (after n batch update)')
flags.DEFINE_string('snapshot', 'snapshots', 'snapshot directory')
flags.DEFINE_bool('restore', False, 'restore')
flags.DEFINE_bool('tests', False, 'tests')

flags.DEFINE_string('job_name', 'worker', 'job type')
flags.DEFINE_string('ps_hosts', '', 'ps hosts')
flags.DEFINE_string('worker_hosts', '', 'worker hosts')
flags.DEFINE_integer('task_index', 0, 'task index')


def main(_):
    if FLAGS.job_name == 'ps':
        import time
        while True:
            time.sleep(1200)

    conf = Config(FLAGS)
    model = Model(conf)

    def test(env, episode, tag_name='score'):
        score = 0
        _, observation, _, done = env.reset()
        while not done:
            act = model.step(observation)
            _, observation, reward, done = env.step((act,), True)
            score += reward

        kwargs = {tag_name: score}
        model.summarize(episode, **kwargs)
        print('Episode:%d , Score:%d' % (episode, score))

    if FLAGS.test:
        test_env = make_env(conf)
        for i in range(100):
            test(test_env, i)
        test_env.close()
    else:
        test_env = None

        env_pool = None
        # imitation train first

        for step in range(FLAGS.imitation_batch):
            if not env_pool:
                env_pool = EnvPool(conf)
            batch = env_pool.get_batch()
            model.imitation_train(*batch)
            if step and (step % FLAGS.test_interval == 0):
                if not test_env:
                    test_env = make_env(conf)
                test(test_env, step // FLAGS.test_interval, 'imitation_score')

            if step and (step % FLAGS.save_interval == 0):
                model.save()
        if env_pool:
            env_pool.close()
            env_pool = None

        # reinforcement

        for step in range(FLAGS.reinforce_batch):
            if not env_pool:
                env_pool = SyncEnvPool(conf, model)
            batch = env_pool.get_batch()
            model.a2c_train(*batch)
            if step and (step % FLAGS.test_interval == 0):
                if not test_env:
                    test_env = make_env(conf)
                test(test_env, step // FLAGS.test_interval, 'reinforce_score')

            if step and (step % FLAGS.save_interval == 0):
                model.save()
        if env_pool:
            env_pool.close()

        if test_env:
            test_env.close()


if __name__ == '__main__':
    FLAGS(sys.argv)
    tf.app.run(main)
