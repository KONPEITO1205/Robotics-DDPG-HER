import numpy as np
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from subprocess import CalledProcessError
from conv_ddpg_agent import ddpg_agent
import gym_dobot.envs as envs
import glfw
from types import MethodType

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env):
    obs = env.reset()
    # close the environment
    # params = {'obs': obs['observation'].shape[0],
    params = {'obs': 3,
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def env_wrapper(env):
    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None

    env.unwrapped.close = MethodType(close, env.unwrapped)
    return env

def launch(args):
    # create the ddpg_agent
    # env = gym.make(args.env_name)
    env = env_wrapper(gym.make(args.env_name, rand_dom=0))
    # get the environment parameters
    env_params = get_env_params(env)
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()


def test(args):
    env = env_wrapper(gym.make(args.env_name, rand_dom=1))
    env_params = get_env_params(env)
    ddpg_trainer = ddpg_agent(args, env, env_params)
    _ = ddpg_trainer._eval_agent()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
    # test(args)
