from __future__ import division
import warnings
import core
from memory import SequentialMemory
from policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from agents import DQNAgent
from callbacks import FileLogger, ModelIntervalCheckpoint
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K


N_zern = 5
Z = 1.8
initial_state = Z * np.random.uniform(-1., 1., size=N_zern)
initial_state = initial_state.round(decimals=1)

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
model.add(Convolution2D(32, (4, 4), strides=(2, 2)))
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
#
# policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
#                               nb_steps=1000000)

episode_len = 400
N_episodes = 500
N_steps = N_episodes * episode_len

# N_anneal = episode_len * 200
N_anneal = 10000

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=N_anneal)

# dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
#                processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
#                train_interval=4, delta_clip=1.)
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=None, nb_steps_warmup=1000, gamma=.99, target_model_update=1000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

# dqn.fit(enviro, callbacks=None, nb_steps=1750000, log_interval=10000)
weights_filename = 'dqn_{}_weights.h5f'.format('PSF')
checkpoint_weights_filename = 'dqn_' + 'PSF' + '_weights_{step}.h5f'
log_filename = 'dqn_{}_log.json'.format('PSF')
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
callbacks += [FileLogger(log_filename, interval=100)]

dqn.fit(enviro, callbacks=None, verbose=2, nb_steps=N_steps, log_interval=1000, nb_max_episode_steps=episode_len)


dqn.test(enviro, nb_episodes=1, visualize=False)

new_state = Z * np.random.uniform(-1., 1., size=N_zern)
enviro.x0 = new_state.copy()
_obs = enviro.reset()
dqn.test(enviro, nb_episodes=1, visualize=False)