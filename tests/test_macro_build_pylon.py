import sys
sys.path.append('.')
from pysc2.tests import utils
from absl.testing import absltest
from pysc2.env import sc2_env
from pysc2.lib import features
from tests.agent_test.protoss_buildpylon_agent import ProtossBuildPylonAgent

class TestProtossMacro(utils.TestCase):

    def test_protoss_agent(self):
        agent = ProtossBuildPylonAgent()
        with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.protoss),
                         sc2_env.Bot(sc2_env.Race.terran,
                                     sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True, use_raw_units=True, use_unit_counts=True),
                step_mul=4,
                game_steps_per_episode=0,
                visualize=True) as env:
            agent.setup(env.observation_spec(), env.action_spec())
            timesteps = env.reset()
            agent.reset()
            while True:
                step_actions = [agent.step(timesteps[0])]
                if timesteps[0].last():
                    break
                timesteps = env.step(step_actions)


if __name__ == "__main__":
    absltest.main()
