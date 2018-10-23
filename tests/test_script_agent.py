# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from future.builtins import range
from pysc2.tests import utils
from absl import flags
from algorithm.script.collect_mineral_shards import CollectMineralShards
from algorithm.script.move_to_beacon import MoveToBeacon
from environment.parallel_env import default_env_maker

AGENT_CLS = {
    'MoveToBeacon': MoveToBeacon,
    'CollectMineralShards': CollectMineralShards,
}

FLAGS = flags.FLAGS
flags.DEFINE_string('map_name', 'CollectMineralShards', 'Map name')
flags.DEFINE_integer('screen_size', 64, 'Feature screen resolution')
flags.DEFINE_integer('minimap_size', 64, 'Feature minimap resolution')
flags.DEFINE_integer('step_mul', 8, 'How many game steps per algorithm step')
flags.DEFINE_bool('visualize', False, "Whether to render with pygame.")
flags.DEFINE_integer('step', 10000, 'Total steps')


class TestScriptAgent(utils.TestCase):

    def test_script_agent(self):
        map_name = FLAGS.map_name
        script_agent_cls = AGENT_CLS[map_name]

        env_arg = {
            'map_name': FLAGS.map_name,
            'screen_size': FLAGS.screen_size,
            'minimap_size': FLAGS.minimap_size,
            'step_mul': FLAGS.step_mul,
            'visualize': FLAGS.visualize}

        with default_env_maker(env_arg) as env:
            obs_spec = env.observation_spec()[0]
            action_spec = env.observation_spec()[0]
            agent = script_agent_cls()
            agent.setup(obs_spec, action_spec)
            obs = env.reset()[0]
            episode = 0
            total_reward = 0
            for _ in range(FLAGS.step):
                act = agent.step(obs)
                obs = env.step((act,))[0]
                total_reward += obs.reward
                if obs.last():
                    print('Episode:%d, score:%d' % (episode, total_reward))
                    total_reward = 0
                    episode += 1


if __name__ == "__main__":
    absltest.main()
