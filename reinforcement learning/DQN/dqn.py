from __future__ import division
import warnings
import core
from memory import SequentialMemory
from policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from agents import DQNAgent
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K


N_zern = 5
Z = 0.75
initial_state = Z * np.ones(N_zern)

enviro = core.PsfEnv(N_zern=N_zern, initial_state=initial_state)
nb_actions = len(enviro.action_space)
WINDOW_LENGTH = 1
PIX = enviro.PSF.pix

input_shape = (WINDOW_LENGTH, PIX, PIX)
model = Sequential()
if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    model.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_dim_ordering.')
model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)
dqn.test(env, nb_episodes=10, visualize=False)