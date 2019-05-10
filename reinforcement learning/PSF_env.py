"""
Environment for Reinforcement Learning using OpenAI gym and Q-learning
applied to NCPA calibration


"""
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
import itertools
import sys
from collections import defaultdict

import gym

import zern_core as zern

class PSFEnv(gym.Env):

    N_pix = 512
    rho_aper = 0.25

    pix = 50
    minPix, maxPix = (N_pix - pix)//2, (N_pix + pix)//2

    tol = 0.10      # Tolerance on the Strehl ratio. If S = 1 - tol, finish the run

    def __init__(self, N_zern, initial_state, DM_stroke=0.05):
        """
        Q-learning is based on DISCRETE actions. Therefore, if we want to correct for
        Zernike aberrations which are continuous, we have to discretize the action space

        :param N_zern: Number of Zernike aberrations we are expected to correct
        :param DM_stroke: Deformable Mirror correction step in waves
        """
        aberration_correction = [DM_stroke, -DM_stroke]
        self.ACTION = N_zern * aberration_correction

        ### Initialize Zernike polynomials

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
        self.N_zern = self.H_matrix.shape[-1]

        self.PEAK = self.peak_PSF()
        self.initial_state = initial_state
        self.state = [initial_state]
        self.rewards = [0]

    def peak_PSF(self):
        """
        Compute the PEAK of the PSF without aberrations so that we can
        normalize everything by it
        """
        return self.compute_PSF(np.zeros(self.N_zern))

    def compute_PSF(self, zern_coef, show_PSF=False):

        ### Zero pad the zernike coefficients to match the dimension of H
        if zern_coef.shape == self.N_zern:
            x = zern_coef.copy()
        elif zern_coef.shape != self.N_zern:
            extra_zeros = np.zeros((self.N_zern - zern_coef.shape[0]))
            x = np.concatenate([zern_coef, extra_zeros])

        phase = np.dot(self.H_matrix, x)
        pupil_function = self.pupil * np.exp(1j * phase)
        image = (np.abs(fftshift(fft2(pupil_function))))**2

        try:
            image /= self.PEAK
            if show_PSF:
                plt.figure()
                plt.imshow(image[self.minPix:self.maxPix, self.minPix:self.maxPix])
                plt.colorbar()

        except AttributeError:
            pass

        return np.max(image)


    def step(self, action, t):
        # get current state
        current_state = self.state[t]
        updated_state = current_state.copy()
        updated_state[action//2] += self.ACTION[action]    # Apply the DM correction

        reward = self.compute_reward(current_state, updated_state)    # Reward is the updated Strehl ratio
        self.rewards.append(reward)

        if reward > 0:
            self.state.append(updated_state)
        else:
            self.state.append(current_state)

        done = True if (1.0 - self.compute_PSF(current_state)) < self.tol else False

        return t + 1, reward, done, {}

    def compute_reward(self, current_state, updated_state):
        old_peak = self.compute_PSF(current_state)
        new_peak = self.compute_PSF(updated_state)
        print("\nInitial Strehl ratio: %.4f" %old_peak)
        # print("Update Strehl: ", new_peak)
        if new_peak > old_peak:
            return (new_peak - old_peak)
        if new_peak < old_peak:
            return -10

    def reset(self):
        self.state = [self.initial_state]
        return 0


def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    """
    Creates an epsilon-greedy policy based
    on a given Q-function and epsilon.

    Returns a function that takes the state
    as an input and returns the probabilities
    for each action in the form of a numpy array
    of length of the action space(set of possible actions).
    """

    def policyFunction(state):
        Action_probabilities = np.ones(num_actions,
                                       dtype=float) * epsilon / num_actions

        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities

    return policyFunction


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, max_correct=5):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """


    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.random.uniform(0., 0.5, size=len(env.ACTION)))

    # The policy we're following
    policy = createEpsilonGreedyPolicy(Q, epsilon, len(env.ACTION))

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in range(max_correct):
            print(t)

            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action, t)

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            print(Q[state][action])
            Q[state][action] += alpha * td_delta
            print(Q[state])

            if done:
                break

            state = next_state

    return Q


if __name__ == "__main__":

    N_zern = 10

    ### Create the PSF environment
    coef = np.random.uniform(-1., 1., size=N_zern)

    psf = PSFEnv(N_zern, coef)

    psf.compute_PSF(coef, show_PSF=True)
    plt.show()

    Q = q_learning(psf, num_episodes=1)

    nor = [np.linalg.norm(s) for s in psf.state]
    # print(nor)