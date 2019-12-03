import torch
import numpy as np
import csv
import matplotlib.pyplot as plt

from stateSpace import StateSpace
from controller import Controller

SPACE_SIZE = 10
MAX_TRIALS = 600 # maximum number of models generated
EXPLORATION = 0.8  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength
HIDDEN_UNITS = 35 # number of hidden units on each layer
EMBEDDING_DIM = 100  # dimension of the embeddings for each state
ACCURACY_BETA = 0.8  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range
UPDATE_STEP = 8


def scoreL2C(archi, n):
    """cf article: retourne
    1 pour des nombres dans l'ordre
    un nombre entre 0 et 1 sinon"""
    a = archi

    s = a[0] ** 2

    for k in range(0, n - 1):
        s += (a[k + 1] - a[k]) ** 2

    s += (a[n - 1] - (n + 1)) ** 2

    return (n + 1) / s

def main_training():
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = torch.cuda.device_count()
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # construct a state space
    state_space = StateSpace()

    # add states
    for i in range(SPACE_SIZE):
        state_space.add_state(name='layer{}'.format(i+1), values=np.arange(1,SPACE_SIZE+1))

    # state_space.print_state_space()

    previous_acc = 0.0
    all_rewards = []

    controller = Controller(1, state_space)

    state = state_space.get_random_state_space(1)
    print("Initial Random State : ", state)

    for trial in range(MAX_TRIALS):
        actions, prob_actions = controller.get_action(state)  # get an action for the previous state

        # print the action probabilities
        #state_space.print_actions(prob_actions.tolist())

        print("Predicted actions : ", actions.tolist())

        reward = scoreL2C(actions.tolist(), SPACE_SIZE)
        all_rewards.append(reward)

        print("Reward : ", reward)

        # actions and states are equivalent, save the state and reward
        state = actions
        prob_state = prob_actions
        controller.store_rollout(state, reward, prob_state)

        if trial % UPDATE_STEP == 0 and trial > 0:
            # train the controller on the saved state and the discounted rewards
            loss = controller.update_policy()
            print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))

        # write the results of this trial into a file
        with open('train_history.csv', mode='a+') as f:
            data = [previous_acc, reward]
            data.extend(state)
            writer = csv.writer(f)
            writer.writerow(data)

    plt.plot(all_rewards)
    plt.show()

if __name__ == '__main__':
    main_training()
