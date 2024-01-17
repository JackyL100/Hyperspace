import gym
import envs
from agents.DDPG.ddpg import DDPG

env = gym.make('Hyperspace-v0')

agent = DDPG(observation_space=env.observation_space.shape[0],
             action_space=2)
print(agent.actor.device)
agent.train(env=env, 
            episodes=3000,
            eval_interval=50)
agent.eval(env)