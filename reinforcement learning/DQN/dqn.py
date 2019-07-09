from __future__ import division
import core
from memory import SequentialMemory
from policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from agents import DQNAgent
from callbacks import FileLogger, ModelIntervalCheckpoint
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K


N_zern = 2
Z = 2.5
initial_state = Z * np.random.uniform(-1., 1., size=N_zern)
# initial_state = initial_state.round(decimals=1)

enviro = core.PsfEnv(N_zern=N_zern, initial_state=initial_state, Z=Z)
nb_actions = len(enviro.action_space)
WINDOW_LENGTH = 1
PIX = enviro.PSF.pix

enviro.PSF.plot_PSF(initial_state)
plt.show()

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
model.add(Convolution2D(8, (6, 6), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(16, (5, 5), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (4, 4), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (4, 4), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=100000000, window_length=WINDOW_LENGTH)
#
# policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
#                               nb_steps=1000000)

episode_len = 500
N_episodes = 1000
N_steps = N_episodes * episode_len

N_anneal = episode_len * 800
# N_anneal = 150000

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=N_anneal)

# dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
#                processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
#                train_interval=4, delta_clip=1.)
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, enable_double_dqn=True,
               processor=None, nb_steps_warmup=5*episode_len, gamma=.90, target_model_update=100,
               train_interval=1, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

# dqn.fit(enviro, callbacks=None, nb_steps=1750000, log_interval=10000)
weights_filename = 'dqn_{}_weights.h5f'.format('PSF')
checkpoint_weights_filename = 'dqn_' + 'PSF' + '_weights_{step}.h5f'
log_filename = 'dqn_{}_log.json'.format('PSF')
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
callbacks += [FileLogger(log_filename, interval=100)]

dqn.fit(enviro, callbacks=None, verbose=2, nb_steps=N_steps, action_repetition=1,
        log_interval=1000, nb_max_episode_steps=episode_len)


# dqn.test(enviro, nb_episodes=1, visualize=False)



# new_state = Z * np.random.uniform(-1., 1., size=N_zern)
# # new_state = np.array([1, 0.5])
# enviro.x0 = new_state.copy()
# _obs = enviro.reset()
# dqn.test(enviro, nb_episodes=1, nb_max_start_steps=0, visualize=False)
#
# # Try with lower gamma, more immediate reward
#
# # Check what's going on
#
# success = np.loadtxt('success.txt')
# failure = np.loadtxt('failure.txt')
# stuck = np.loadtxt('stuck.txt')
#
# start = stuck[:,:2]
# end = stuck[:, 3:-1]
#
# plt.figure()
# plt.xlim([-Z, Z])
# plt.ylim([-Z, Z])
# plt.xlabel(r'$a_1$')
# plt.ylabel(r'$a_2$')
# plt.grid(True)
# plt.scatter(success[:,0], success[:, 1], color='green', marker='^',label='Success')
# plt.scatter(failure[:,0], failure[:, 1], color='red', marker='s', label='Failure')
# plt.scatter(start[:,0], start[:, 1], color='orange', label='Stuck')
#
# for i in range(stuck.shape[0]):
#     x0, y0 = start[i,0], start[i, 1]
#     lx, ly = end[i,0] - x0, end[i, 1] - y0
#     plt.arrow(x0, y0, lx, ly, color='black', linestyle='-.')
#
# plt.legend()
# plt.show()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.figure()
M = 5
N = np.array([2, 3, 4, 5])
K = np.array([5, 10, 15, 20])
for n in N:
    # r = M**K / (n*M**(K/n))
    # log_r = np.log10(r)
    log_r = K * (1 - 1/n) *np.log10(M) - np.log10(n)
    plt.plot(K, log_r, label=n)
# plt.axhline(y=1.0, color='black', linestyle='--')
plt.ylim([0, 10])
plt.xlim([5, 20])
plt.legend(title=r'$N$ Networks')
plt.show()
