import torch
import numpy as np
import csv
import matplotlib.pyplot as plt

from stateSpace import StateSpace
from controller import Controller, NasLSTM
from torch import nn
from torch import optim

SPACE_SIZE = 3
MAX_TRIALS = 600 # maximum number of models generated
EXPLORATION = 0.0  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength
HIDDEN_UNITS = 200 # number of hidden units on each layer
EMBEDDING_DIM = 200  # dimension of the embeddings for each state
ACCURACY_BETA = 0.8  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range
UPDATE_STEP = 6
LEARNING_RATE = 1e-3


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

    controller = Controller(1, state_space, hidden_units=HIDDEN_UNITS, exploration=EXPLORATION, learning_rate=LEARNING_RATE)

    num_distinct_states = np.sum([state_space[i]['size'] for i in range(state_space.size)])

    policy_network = NasLSTM(num_embeddings=num_distinct_states, embedding_dim=EMBEDDING_DIM,
                             hidden_dim=HIDDEN_UNITS,
                             state_space=state_space,
                             num_layers=1)

    for name, param in policy_network.named_parameters():
        if "nas_cell_layer" in name or "hidden2state" in name:
            nn.init.uniform_(param, a=-0.08, b=0.08)

    policy_optim = optim.Adam(policy_network.parameters(), lr=1e-2)
    policy_network.float()
    policy_network.train()

    for trial in range(MAX_TRIALS):
        # ANCIENNE VERSION
        # actions, prob_actions = controller.get_action()

        if np.random.random() < controller.exploration:
            # print("Generating random action to explore")
            actions = []
            prob_actions = []

            for i in range(state_space.size):
                state_ = state_space[i]
                size = state_['size']
                prob = torch.zeros(size)

                ind = np.random.choice(size, size=1)[0]
                prob[ind] = 1
                actions.append(ind)
                prob_actions.append(prob)

            actions = torch.tensor(actions)
            prob_actions = torch.stack(prob_actions)
            prob_actions.requires_grad = True
        else:
            actions, prob_actions = policy_network()

        # print the action probabilities
        #state_space.print_actions(prob_actions.tolist())

        reward = scoreL2C(actions.tolist(), SPACE_SIZE)
        all_rewards.append(reward)

        if trial % 50 == 0:
            print("Predicted actions : ", actions.tolist())
            print("Reward : ", reward)

        # actions and states are equivalent, save the state and reward
        state = actions
        prob_state = prob_actions
        controller.store_rollout(state, reward, prob_state)

        # train the controller on the saved state and the discounted rewards
        loss = controller.update_policy(reward, policy_optim)

        # print("Reward : ", reward)
        # print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))

        # write the results of this trial into a file
        with open('train_history.csv', mode='a+') as f:
            data = [previous_acc, reward]
            data.extend(state)
            writer = csv.writer(f)
            writer.writerow(data)

    # plt.plot(all_rewards)
    # plt.show()

    filtre_passe_bas = []
    RANGE = 10
    for k in range(len(all_rewards) - RANGE):
        filtre_passe_bas.append(np.mean(all_rewards[k:k + RANGE]))

    plt.plot(filtre_passe_bas)
    plt.show()

    #print(controller.policy_network.hidden2state[0].weight)

if __name__ == '__main__':
    main_training()
