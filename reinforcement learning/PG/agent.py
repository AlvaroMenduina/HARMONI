import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
import zern_core as zern

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


class PointSpreadFunction(object):
    """
    PointSpreadFunction is in charge of computing the PSF
    for a given set of Zernike coefficients
    """

    ### Parameters
    rho_aper = 0.25         # Size of the aperture relative to 1.0
    N_pix = 512
    pix = 25                # Number of pixels for the Zoom of the PSF
    minPix, maxPix = (N_pix - pix) // 2, (N_pix + pix) // 2
    stroke = 0.1

    def __init__(self, N_zern, initial_state):

        ### Zernike Wavefront
        x = np.linspace(-1, 1, self.N_pix, endpoint=True)
        xx, yy = np.meshgrid(x, x)
        rho, theta = np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)
        self.pupil = rho <= self.rho_aper
        rho, theta = rho[self.pupil], theta[self.pupil]
        zernike = zern.ZernikeNaive(mask=self.pupil)
        _phase = zernike(coef=np.zeros(N_zern + 3), rho=rho/self.rho_aper, theta=theta, normalize_noll=False,
                         mode='Jacobi', print_option='Silent')
        H_flat = zernike.model_matrix[:, 3:]  # remove the piston and tilts
        self.H_matrix = zern.invert_model_matrix(H_flat, self.pupil)

        # Update the number of aberrations to match the dimensions of H
        self.N = N_zern
        self.N_zern = self.H_matrix.shape[-1]

        self.PEAK = self.peak_PSF()

        # Keep track of the STATE of the system
        self.state = initial_state.copy()

    def peak_PSF(self):
        """
        Compute the PEAK of the PSF without aberrations so that we can
        normalize everything by it
        """

        im, strehl, c0 = self.compute_PSF(np.zeros(self.N_zern))

        return strehl

    def compute_PSF(self, zern_coef):
        """
        Compute the PSF and the Strehl ratio
        """

        if zern_coef.shape != self.N_zern:      # Zero-pad to match dimensions
            extra_zeros = np.zeros((self.N_zern - zern_coef.shape[0]))
            zern_coef = np.concatenate([zern_coef, extra_zeros])

        phase = np.dot(self.H_matrix, zern_coef)
        pupil_function = self.pupil * np.exp(1j * phase)
        image = (np.abs(fftshift(fft2(pupil_function))))**2
        image = image[self.minPix:self.maxPix, self.minPix:self.maxPix]

        try:
            image /= self.PEAK

        except AttributeError:
            # If self.PEAK is not defined, self.compute_PSF will compute the peak
            pass

        strehl = np.max(image)

        # Core intensity
        pix = 9
        minPix, maxPix = (self.pix - pix) // 2 + 1, (self.pix + pix) // 2 + 1
        im_core = image[minPix:maxPix, minPix:maxPix]
        core = np.mean(im_core)
        # print(image.shape)

        return image.reshape((1, 25, 25, 1)), strehl, core

    def update_state(self, action, s0):
        if action%2 == 0 and action != 2*self.N:
            self.state[action//2] += 1.*self.stroke
            act_s = '(+)'
        elif action%2 != 0 and action != 2*self.N:
            self.state[action//2] -= 1.*self.stroke
            act_s = '(-)'
        template = '[ ' + self.N*' {:.3f} ' + ' ]'
        print(' || Strehl: %.3f | '%s0, template.format(*self.state), ' Action: %d ' %(action//2+1) + act_s)

    def plot_PSF(self, zern_coef, i=0):
        """
        Plot an image of the PSF
        :param zern_coef:
        :param i: iteration (for labelling purposes)
        """

        PSF, strehl, c = self.compute_PSF(zern_coef)
        # PSF_zoom = PSF[self.minPix:self.maxPix, self.minPix:self.maxPix]

        plt.figure()
        plt.imshow(PSF)
        plt.title('Iter: %d Strehl: %.3f' %(i, strehl))
        plt.colorbar()
        plt.clim(vmin=0, vmax=1)

class PsfEnv(object):
    """
    Environment class for Focal Plane Sharpening using Reinforcement Learning
    """
    threshold = 0.85            # Threshold for Stopping: Strehl ratio > 0.85
    low_strehl = 0.15
    def __init__(self, N_zern, initial_state, Z):
        self.Z = Z
        self.x0 = initial_state.copy()
        print(self.x0)
        self.PSF = PointSpreadFunction(N_zern=N_zern, initial_state=initial_state)

        self.action_space = list(range(2*self.PSF.state.shape[0]))
        self.success, self.failure = 0, 0
        self.PSFs_learned = 0

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
        im0, s0, c0 = self.PSF.compute_PSF(self.PSF.state)

        # Update state and recompute
        self.PSF.update_state(action, s0)
        new_state = self.PSF.state.copy()
        image, strehl, core = self.PSF.compute_PSF(self.PSF.state)
        observation = image         # An image of the PSF

        # Reward according to the gain in Strehl ratio
        r1 = strehl - s0             # TODO: other possible rewards
        r2 = 10*(core - c0)
        r3 = -0.05          # Discourage (+) then (-) same action
        reward = r1 + r2 + r3
        # print("Strehl: %.3f" %strehl)
        # template = '\nStrehl gain: {:.4f} || Core gain: {:.4f} || Total Reward {:.4f}'
        # print(template.format(r1, r2, reward))

        info = strehl

        # End episode if aberrations too high, Strehl too low or Strehl good enough
        abss = [True if np.abs(x) > 3.5 else False for x in new_state]
        failure = any(abss) or strehl < self.low_strehl
        success = True if strehl > self.threshold else False


        if failure:

            self.failure += 1
            total = self.success + self.failure
            print("\n------- FAILED -------- (%d/%d)" % (self.failure, total))
            reward -= 50
            done = True

        elif success:

            self.success += 1
            total = self.success + self.failure
            print("\n------- SUCCESS -------- (%d/%d)" %(self.success, total))
            reward += 1
            # Successful calibration. Increase counter
            done = True

        else:
            done = False

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
            print("Current state: ", self.PSF.state.copy())
            x0 = self.Z * np.random.uniform(-1., 1., size=self.PSF.N)
            self.x0 = x0.copy()
            self.PSF.state = self.x0.copy()
            image, strehl, core = self.PSF.compute_PSF(self.PSF.state)
            if 1.2* self.low_strehl < strehl < 0.8*self.threshold:
                print("Reseting to a RANDOM case: ", self.PSF.state)
                observation = image
                break

        return observation



N_zern = 2
Z = 2.5
initial_state = Z * np.random.uniform(-1., 1., size=N_zern)
enviro = PsfEnv(N_zern=N_zern, initial_state=initial_state, Z=Z)
nb_actions = len(enviro.action_space)

PIX = 25
WINDOW_LENGTH = 1
input_shape = (PIX, PIX, WINDOW_LENGTH)



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

        model.add(Conv2D(32, kernel_size=(6, 6), strides=2, activation='relu', input_shape=input_shape,
                         init='he_uniform'))
        model.add(Conv2D(16, kernel_size=(4, 4), strides=2, activation='relu', init='he_uniform'))
        model.add(Flatten())
        model.add(Dense(self.actions_, activation='softmax'))

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
        avg_reward = []
        reward_sum = 0
        ep_number = 0
        prev_x = None
        observation = self.env.reset()
        # Training progress
        while True:
            # Get observe
            # cur_x = prepro(observation)
            cur_x = observation
            # Consider frame difference and take action.
            x = cur_x - prev_x if prev_x is not None else np.zeros(cur_x.shape)
            # print(x.shape)
            prev_x = cur_x
            # aprob = self.model.predict(x.reshape((1, PIX, PIX, 1)), batch_size=1).flatten()
            aprob = self.model.predict(x, batch_size=1).flatten()
            # print(aprob)
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
                    input_tr_y = prob_actions + self.learning_rate * np.squeeze(np.vstack(tr_y))
                    self.model.train_on_batch(np.vstack(tr_x).reshape(-1, PIX, PIX, 1), input_tr_y)
                    tr_x, tr_y, prob_actions = [], [], []
                    # Checkpoint
                    # os.remove(self.model_path) if os.path.exists(self.model_path) else None
                    # self.model.save(self.model_path)

                avg_reward.append(float(reward_sum))
                if len(avg_reward) > 30: avg_reward.pop(0)
                print('Epsidoe {:} reward {:.2f}, Last 30ep Avg. rewards {:.2f}.'.format(ep_number, reward_sum,
                                                                                         np.mean(avg_reward)))
                print('{:.4f},{:.4f}'.format(reward_sum, np.mean(avg_reward)), end='\n', file=log, flush=True)
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
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(cur_x.shape)
        self.prev_x = cur_x
        aprob = self.model.predict(x.reshape((1, PIX, PIX, 1)), batch_size=1).flatten()

        return np.argmax(aprob)

agent = Agent_PG(enviro)
agent.train()


if __name__ == '__main__':

    # args = parse()
    pass
    print('A')