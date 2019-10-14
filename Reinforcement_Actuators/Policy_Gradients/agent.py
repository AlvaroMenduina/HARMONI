"""

Policy Gradient

"""


import itertools

import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import scipy.misc

import os
import keras
import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten, Permute
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, Adamax, RMSprop
from keras import backend as K

from keras.backend.tensorflow_backend import set_session
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))

# PARAMETERS
Z = 1.5                    # Strength of the aberrations -> relates to the Strehl ratio
pix = 30                    # Pixels to crop the PSF
N_PIX = 256                 # Pixels for the Fourier arrays
RHO_APER = 0.5              # Size of the aperture relative to the physical size of the Fourier arrays
RHO_OBSC = 0.15             # Central obscuration

# ==================================================================================================================== #
#                                   Deformable Mirror - ACTUATOR MODEL functions
# ==================================================================================================================== #

def actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, radial=True):
    """
    Computes the (Xc, Yc) coordinates of actuator centres
    inside a circle of rho_aper, assuming there are N_actuators
    along the [-1, 1] line

    :param N_actuators: Number of actuators along the [-1, 1] line
    :param rho_aper: Relative size of the aperture wrt [-1, 1]
    :param rho_obsc: Relative size of the obscuration
    :param radial: if True, we add actuators at the boundaries RHO_APER, RHO_OBSC
    :return: [act (list of actuator centres), delta (actuator separation)], max_freq (max spatial frequency we sense)
    """

    x0 = np.linspace(-1., 1., N_actuators, endpoint=True)
    delta = x0[1] - x0[0]
    N_in_D = 2*RHO_APER/delta
    print('%.2f actuators in D' %N_in_D)
    max_freq = N_in_D / 2                   # Max spatial frequency we can sense
    xx, yy = np.meshgrid(x0, x0)
    x_f = xx.flatten()
    y_f = yy.flatten()

    act = []    # List of actuator centres (Xc, Yc)
    for x_c, y_c in zip(x_f, y_f):
        r = np.sqrt(x_c ** 2 + y_c ** 2)
        if r < (rho_aper - delta/2) and r > (rho_obsc + delta/2):   # Leave some margin close to the boundary
            act.append([x_c, y_c])

    if radial:  # Add actuators at the boundaries, keeping a constant angular distance
        for r in [rho_aper, rho_obsc]:
            N_radial = int(np.floor(2*np.pi*r/delta))
            d_theta = 2*np.pi / N_radial
            theta = np.linspace(0, 2*np.pi - d_theta, N_radial)
            # Super important to do 2Pi - d_theta to avoid placing 2 actuators in the same spot... Degeneracy
            for t in theta:
                act.append([r*np.cos(t), r*np.sin(t)])

    total_act = len(act)
    print('Total Actuators: ', total_act)
    return [act, delta], max_freq

def plot_actuators(centers):
    """
    Plot the actuators given their centre positions [(Xc, Yc), ..., ]
    :param centers: act from actuator_centres
    :return:
    """
    N_act = len(centers[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    circ1 = Circle((0,0), RHO_APER, linestyle='--', fill=None)
    circ2 = Circle((0,0), RHO_OBSC, linestyle='--', fill=None)
    ax.add_patch(circ1)
    ax.add_patch(circ2)
    for i, c in enumerate(centers[0]):
        ax.scatter(c[0], c[1], s=20, label=i+1)
    ax.set_aspect('equal')
    plt.legend()
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.title('%d actuators' %N_act)

def actuator_matrix(centres, alpha=0.75, rho_aper=RHO_APER, rho_obsc=RHO_OBSC):
    """
    Computes the matrix containing the Influence Function of each actuator
    Returns a matrix of size [N_PIX, N_PIX, N_actuators] where each [N_PIX, N_PIX, k] slice
    represents the effect of "poking" one actuator

    Current model: Gaussian function

    :param centres: [act, delta] list from actuator_centres containing the centres and the spacing
    :param alpha: scaling factor to control the tail of the Gaussian and avoid overlap / crosstalk
    :param rho_aper:
    :param rho_obsc:
    :return:
    """
    # TODO: Update the Model to something other than a Gaussian

    cent, delta = centres
    N_act = len(cent)
    matrix = np.empty((N_PIX, N_PIX, N_act))
    x0 = np.linspace(-1., 1., N_PIX, endpoint=True)
    xx, yy = np.meshgrid(x0, x0)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    pupil = (rho <= rho_aper) & (rho >= rho_obsc)

    for k in range(N_act):
        xc, yc = cent[k][0], cent[k][1]
        r2 = (xx - xc) ** 2 + (yy - yc) ** 2
        matrix[:, :, k] = pupil * np.exp(-r2 / (alpha * delta) ** 2)

    mat_flat = matrix[pupil]

    return matrix, pupil, mat_flat

# ==================================================================================================================== #
#                                          Point Spread Function
# ==================================================================================================================== #

class PointSpreadFunction(object):
    """
    Faster version of the PSF that uses a single FFT operation to generate multiple images
    """
    minPix, maxPix = (N_PIX + 1 - pix) // 2, (N_PIX + 1 + pix) // 2

    def __init__(self, N_actuators):

        centers, MAX_FREQ = actuator_centres(N_actuators, radial=False)
        self.N_act = len(centers[0])
        plot_actuators(centers)
        matrices = actuator_matrix(centers)
        self.RBF_mat = matrices[0].copy()
        self.pupil_mask = matrices[1].copy()
        self.RBF_flat = matrices[2].copy()

        # Defocus phase
        # self.defoc_coef = np.random.uniform(low=-1, high=1, size=self.N_act)
        # self.defoc_phase = np.dot(self.RBF_mat, self.defoc_coef)

        self.PEAK = self.peak_PSF()
        self.PerfectPSF, _s = self.compute_PSF(np.zeros(self.N_act))

        # Deformable Mirror parameters
        self.stroke = 0.1

        self.state = np.random.uniform(low=-Z, high=Z, size=self.N_act)
        plt.show()

    def peak_PSF(self):
        """
        Compute the PEAK of the PSF without aberrations so that we can
        normalize everything by it
        """
        im, strehl = self.compute_PSF(np.zeros(self.N_act))
        return strehl

    def compute_PSF(self, coef):
        """
        Compute the PSF and the Strehl ratio
        """

        phase = np.dot(self.RBF_mat, coef)
        pupil_function = self.pupil_mask * np.exp(1j * phase)
        image = (np.abs(fftshift(fft2(pupil_function))))**2

        # focus_pupil = self.pupil_mask * np.exp(1j * (phase + self.defoc_phase))
        # image_focus = (np.abs(fftshift(fft2(focus_pupil)))) ** 2

        try:
            image /= self.PEAK
            # image_focus /= self.PEAK

        except AttributeError:
            # If self.PEAK is not defined, self.compute_PSF will compute the peak
            pass

        image = image[self.minPix:self.maxPix, self.minPix:self.maxPix]
        strehl = np.max(image)
        # image_focus = image_focus[self.minPix:self.maxPix, self.minPix:self.maxPix]

        # image = np.concatenate([image, image_focus], axis=1)

        return image, strehl

    def update_state(self, action, s0):
        if action%2 == 0 and action != 2*self.N_act:
            self.state[action//2] += 1.*self.stroke
            act_s = '(+)'
        elif action%2 != 0 and action != 2*self.N_act:
            self.state[action//2] -= 1.*self.stroke
            act_s = '(-)'
        template = '[ ' + self.N_act*' {:.3f} ' + ' ]'
        # print(' || Strehl: %.3f | '%s0, template.format(*self.state), ' Action: %d ' %(action//2+1) + act_s)
        # print(' || Strehl: %.3f | '%s0, ' Action: %d ' %(action//2+1) + act_s)

    def plot_PSF(self, coef, i=0):
        """
        Plot an image of the PSF
        :param coef:
        :param i: iteration (for labelling purposes)
        """

        PSF, strehl = self.compute_PSF(coef)

        plt.figure()
        plt.imshow(PSF)
        plt.title('Iter: %d Strehl: %.3f' %(i, strehl))
        plt.colorbar()
        plt.clim(vmin=0, vmax=1)

class PsfEnv(object):
    """
    Environment class for Focal Plane Sharpening using Reinforcement Learning
    """
    threshold = 0.80            # Threshold for Stopping: Strehl ratio > 0.80
    low_strehl = 0.20
    def __init__(self, N_actuators):
        self.Z = Z
        self.PSF = PointSpreadFunction(N_actuators)

        self.action_space = list(range(2*self.PSF.N_act))
        self.success, self.failure = 0, 0
        self.PSFs_learned = 0
        self.iter = 0
        self.max_iter = 25 * 2*self.PSF.N_act

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info)
        # Arguments
            action (object): An action provided by the environment.

        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """

        # Compute the previous Strehl
        im0, s0 = self.PSF.compute_PSF(self.PSF.state)
        diff0 = np.sum(np.abs(im0 - self.PSF.PerfectPSF)[:, :pix])

        # Update state and recompute
        self.PSF.update_state(action, s0)
        new_state = self.PSF.state.copy()
        image, strehl = self.PSF.compute_PSF(self.PSF.state)
        observation = image         # An image of the PSF
        diff = np.sum(np.abs(image - self.PSF.PerfectPSF)[:, :pix])

        # Reward according to the gain in Strehl ratio
        rew_strehl = strehl - s0             # TODO: other possible rewards
        rew_img = diff0 - diff
        # print(rew_strehl, rew_img)
        r3 = -0.05          # Discourage (+) then (-) same action
        reward = rew_img + rew_strehl
        # print("Strehl: %.3f" %strehl)
        # template = '\nStrehl gain: {:.4f} || Core gain: {:.4f} || Total Reward {:.4f}'
        # print(template.format(r1, r2, reward))

        self.iter += 1


        # End episode if aberrations too high, Strehl too low or Strehl good enough
        abss = [True if np.abs(x) > 3.5 else False for x in new_state]
        failure = any(abss) or strehl < self.low_strehl or self.iter > self.max_iter
        success = True if strehl > self.threshold else False

        if failure:

            self.failure += 1
            total = self.success + self.failure
            pcent = 100*self.failure / total
            print("\n------- FAILED -------- (%d/%d [%.2f pc])" % (self.failure, total, pcent))
            # reward -= 50
            done = True
            info = 0

        elif success:

            self.success += 1
            total = self.success + self.failure
            pcent = 100*self.success / total
            print("\n------- SUCCESS -------- (%d/%d [%.2f pc])" %(self.success, total, pcent))
            # reward += 1
            # Successful calibration. Increase counter
            done = True
            info = 1

        else:
            done = False
            info = -1

        return (observation, reward, done, info)

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        s = 1.0
        # Make sure you do not start with a Strehl that's already higher than the treshold
        while True:
            # print("Current state: ", self.PSF.state.copy())
            x0 = self.Z * np.random.uniform(-1., 1., size=self.PSF.N_act)
            self.x0 = x0.copy()
            self.PSF.state = self.x0.copy()
            image, strehl = self.PSF.compute_PSF(self.PSF.state)
            if 1.2*self.low_strehl < strehl < 0.8*self.threshold:
                # print("Reseting to a RANDOM case: ")
                # print("Reseting to a RANDOM case: ", self.PSF.state)
                observation = image
                self.iter = 0
                break

        return observation

N_actuators = 10
enviro = PsfEnv(N_actuators)
nb_actions = len(enviro.action_space)

WINDOW_LENGTH = 1
input_shape = (pix, pix, WINDOW_LENGTH)

# Reference: https://github.com/mkturkcan/Keras-Pong/blob/master/keras_pong.py
def discount_rewards(r):
    gamma = 0.99
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# Sum up losses instead of  mean
def categorical_crossentropy(target, output):
    _epsilon = tf.convert_to_tensor(10e-8, dtype=output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    return tf.reduce_sum(- tf.reduce_sum(target * tf.log(output), axis=len(output.get_shape()) - 1),axis=-1)

class Agent(object):
    def __init__(self, env):
        self.env = env

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        This function must exist in agent
        Input:
            When running dqn:
                observation: np.array
                    stack 4 last preprocessed frames, shape: (84, 84, 4)
            When running pg:
                observation: np.array
                    current RGB screen of game, shape: (210, 160, 3)
        Return:
            action: int
                the predicted action from trained model
        """
        raise NotImplementedError("Subclasses should implement this!")

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        raise NotImplementedError("Subclasses should implement this!")


class Agent_PG(Agent):
    def __init__(self, env):
        super(Agent_PG, self).__init__(env)

        self.log_path = os.getcwd() + 'pg.log'
        # self.model_path = args.save_network_path + 'pong_model_checkpoint.h5'
        self.env = env
        self.actions_ = nb_actions

        # self.learning_rate = args.learning_rate
        self.learning_rate = 0.00025
        # Model for Breakout #
        model = Sequential()
        # if K.image_dim_ordering() == 'tf':
        #     # (width, height, channels)
        #     model.add(Permute((2, 3, 1), input_shape=input_shape))
        # elif K.image_dim_ordering() == 'th':
        #     # (channels, width, height)
        #     model.add(Permute((1, 2, 3), input_shape=input_shape))
        # else:
        #     raise RuntimeError('Unknown image_dim_ordering.')

        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(self.actions_, activation='softmax'))
        model.summary()

        opt = Adam(lr=self.learning_rate)
        model.compile(loss=categorical_crossentropy, optimizer=opt)
        self.model = model

    def init_game_setting(self):
        self.prev_x = None

    def train(self):
        # Init
        log = open(self.log_path, 'w')
        log.write('reward,avg_reward\n')
        batch_size = 1
        frames, prob_actions, dlogps, drs = [], [], [], []
        tr_x, tr_y = [], []
        self.avg_reward = []
        reward_sum = 0
        ep_number = 0
        prev_x = None
        observation = self.env.reset()

        ### Training info
        self.failures, self.successes, self.total = [], [], []
        fail, success, tot = 0, 0, 0

        # Training progress
        plt.figure()
        plt.xlabel('Episode')
        plt.ylabel('Result')
        while True:
            # Get observe
            # cur_x = prepro(observation)
            cur_x = observation
            # Consider frame difference and take action.
            # x = cur_x - prev_x if prev_x is not None else np.zeros(cur_x.shape)
            x = cur_x
            # print(x.shape)
            prev_x = cur_x
            aprob = self.model.predict(x.reshape((1, pix, pix, 1)), batch_size=1).flatten()
            # aprob = self.model.predict(x, batch_size=1).flatten()
            template = '[ ' + len(aprob) * ' {:.3f} ' + ' ]'
            # print(template.format(*aprob))
            frames.append(x)
            prob_actions.append(aprob)
            action = np.random.choice(self.actions_, 1, p=aprob.reshape((self.actions_)))[0]
            y = np.zeros([self.actions_])
            y[action] = 1
            observation, reward, done, info = self.env.step(action)

            reward_sum += reward
            drs.append(reward)
            dlogps.append(np.array(y).astype('float32') - aprob)

            if done:
                if ep_number >= 2500:
                    break
                tot += 1
                self.total.append(tot)
                if info == 0:
                    fail += 1
                elif info == 1:
                    success += 1
                self.failures.append(fail)
                plt.scatter(ep_number, fail, color='red', s=10)
                self.successes.append(success)
                plt.scatter(ep_number, success, color='green', s=10)

                template = '[ ' + len(aprob) * ' {:.3f} ' + ' ]'
                print(template.format(*aprob))

                ep_number += 1
                ep_x = np.vstack(frames)
                ep_dlogp = np.vstack(dlogps)
                ep_reward = np.vstack(drs)
                # print(ep_reward)
                # Discount and normalize rewards
                discounted_ep_reward = discount_rewards(ep_reward)
                # print(discounted_ep_reward)
                discounted_ep_reward -= np.mean(discounted_ep_reward)
                discounted_ep_reward /= np.std(discounted_ep_reward)
                ep_dlogp *= discounted_ep_reward

                # Store current episode into training batch
                tr_x.append(ep_x)
                tr_y.append(ep_dlogp)
                frames, dlogps, drs = [], [], []
                if ep_number % batch_size == 0:
                    # print(tr_y.shape)
                    input_tr_y = prob_actions + self.learning_rate * np.squeeze(np.vstack(tr_y))
                    # print(input_tr_y)
                    # print(np.vstack(tr_x).reshape(-1, pix, pix, 1).shape)
                    # print(input_tr_y.shape)
                    self.model.train_on_batch(np.vstack(tr_x).reshape(-1, pix, pix, 1), input_tr_y)
                    tr_x, tr_y, prob_actions = [], [], []
                    # Checkpoint
                    # os.remove(self.model_path) if os.path.exists(self.model_path) else None
                    # self.model.save(self.model_path)

                self.avg_reward.append(float(reward_sum))
                if len(self.avg_reward) > 30: self.avg_reward.pop(0)
                print('Epsidoe {:} reward {:.2f}, Last 30ep Avg. rewards {:.2f}.'.format(ep_number, reward_sum,
                                                                                         np.mean(self.avg_reward)))
                print('{:.4f},{:.4f}'.format(reward_sum, np.mean(self.avg_reward)), end='\n', file=log, flush=True)
                reward_sum = 0
                observation = self.env.reset()
                prev_x = None

    def make_action(self, observation, test=True):
        """
        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
        Return:
            action: int
                the predicted action from trained model
        """
        # cur_x = prepro(observation)
        cur_x = observation
        # x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(cur_x.shape)
        x = cur_x
        self.prev_x = cur_x
        aprob = self.model.predict(x.reshape((1, pix, pix, 1)), batch_size=1).flatten()
        print(aprob)

        return np.argmax(aprob)

    def test(self):

        img0 = self.env.reset()
        observation = img0.copy()
        done = False
        while not done:
            action = self.make_action(observation)
            print(action)
            observation, reward, done, info = self.env.step(action)

agent = Agent_PG(enviro)
agent.train()

def test(agent, N_samples=5, N_iter=100):

    N_act = agent.env.PSF.N_act
    template = '[ ' + N_act * ' {:.3f} ' + ' ]'


    strehls = []
    plt.figure()
    for k in range(N_samples):
        print('\nSample Number %d' %(k+1))
        img0 = agent.env.reset()
        observation = img0.copy()
        done = False
        initial_state = agent.env.x0.copy()
        print('Initial State:', template.format(*initial_state))

        s = []
        states = [initial_state]

        for i in range(N_iter):
            if done:
                break
            action = agent.make_action(observation)
            actuator = action//2
            sign = action%2
            if sign == 0:
                print('Actuator %d (+)' % actuator)
            if sign == 1:
                print('Actuator %d (-)' % actuator)
            observation, reward, done, info = agent.env.step(action)
            new_state = agent.env.PSF.state.copy()
            print('New State:', template.format(*new_state))
            states.append(new_state)

        states = np.array(states)
        print(states.shape)
        plt.figure()
        for j in range(N_act):
            plt.plot(states[:,j], label='%d' % (j + 1))

        plt.xlabel('Iteration')
        plt.ylabel('Coefficient')
        plt.legend(title='Actuator')

        #     s.append(np.max(observation))
        #     # plt.scatter(i+1, np.max(observation), color='black')
        #
        # plt.plot(s, color='black')
        # strehls.append(s)


test(agent)
plt.show()


if __name__ == '__main__':

    # args = parse()
    pass
    print('A')