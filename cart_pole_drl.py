# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 22:23:17 2019

@author: jaydeep thik
"""

import gym
import numpy as np

from keras import models
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'CartPole-v0'

env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n

model = models.Sequential()
model.add(Flatten(input_shape = (1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=5000, window_length = 1)
dqn = DQNAgent(model = model, policy=policy, nb_actions=nb_actions, memory = memory, nb_steps_warmup =10, target_model_update=1e-2)
dqn.compile(Adam(1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)
dqn.test(env, nb_episodes=5, visualize=True)