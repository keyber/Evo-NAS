import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class NasLSTM(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, hidden_dim, state_sizes, num_layers=2):
        """
            Controller LSTM which is tasked to generate architectural hyperparameters of neural networks

        :param num_embeddings: total number of states (= number of architectural choices for each type of layer)
        :param embedding_dim: embedding dimension
        :param hidden_dim: number of hidden units
        :param state_sizes: list storing each state size of the state space
                            Suppose the following state space: [kernel1, filter1, kernel2, filter2],
                            state_sizes will be [kernel_size, filter_size, kernel_size, filter_size]
        :param num_layers: number of layers (default value as defined in 'Neural Architecture Search With Reinforcement
                           Learning' (Zoph and Le, ICLR 2017))
        """
        super(NasLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.nas_cell_layer1 = nn.LSTMCell(input_size=embedding_dim, hidden_size=hidden_dim)
        self.nas_cell_layer2 = nn.LSTMCell(input_size=embedding_dim, hidden_size=hidden_dim)
        #self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.hidden2state = [nn.Linear(hidden_dim, size) for size in state_sizes]

    def forward(self, input_seq):
        seq_len = input_seq.shape[0]
        batch_size = input_seq.shape[1]

        embeds = self.word_embeddings(input_seq)

        h1 = torch.zeros((batch_size, self.hidden_dim))
        c1 = torch.zeros((batch_size, self.hidden_dim))
        h2 = torch.zeros((batch_size, self.hidden_dim))
        c2 = torch.zeros((batch_size, self.hidden_dim))
        input_layer1 = embeds[0]

        state_scores = []

        for i in range(seq_len):
            h1, c1 = self.nas_cell_layer1(input_layer1, (h1, c1))
            h2, c2 = self.nas_cell_layer2(h1, (h2, c2))
            prob = self.hidden2state[i](h2)

            prob = F.log_softmax(prob, dim=1)
            state_scores.append(prob)
            pred = torch.argmax(prob, dim=1)

            input_layer1 = pred

        return state_scores

class Controller:
    '''
    Utility class to manage the RNN Controller
    '''

    def __init__(self, num_layers, state_space,
                 reg_param=0.001,
                 discount_factor=0.99,
                 exploration=0.8,
                 hidden_units=35,
                 embedding_dim=20,
                 clip_norm=0.0,
                 restore_controller=False,
                 init_upper_bound=0.08,
                 learning_rate=6e-4):

        self.num_layers = num_layers
        self.state_space = state_space
        self.state_size = self.state_space.size

        self.hidden_units = hidden_units
        self.embedding_dim = embedding_dim
        self.reg_strength = reg_param
        self.discount_factor = discount_factor
        self.exploration = exploration
        self.restore_controller = restore_controller
        self.clip_norm = clip_norm

        self.reward_buffer = []
        self.state_buffer = []
        self.prob_buffer = []

        self.cell_outputs = []
        self.policy_classifiers = []
        self.policy_actions = []
        self.policy_labels = []

        self.policy_network = None
        self.policy_optim = None
        self.upper_bound = init_upper_bound
        self.lr = learning_rate
        self.global_step = 0
        self.build_policy_network()

    def get_action(self, states):
        '''
        Gets a one hot encoded action list, either from random sampling or from the Controller RNN

        :param state: a list of one hot encoded states, whose first value is used as initial state for the controller RNN
        :return: a one hot encoded action list
        '''
        if np.random.random() < self.exploration:
            print("Generating random action to explore")
            actions = []

            for i in range(self.state_size * self.num_layers):
                state_ = self.state_space[i]
                size = state_['size']

                sample = np.random.choice(size, size=1)
                sample = state_['index_map_'][sample[0]]
                action = self.state_space.onehot_encode(i, sample)
                actions.append(action)
            return actions, actions

        else:
            print("Prediction action from Controller")
            initial_state = self.state_space[0]
            size = initial_state['size']

            states = self.state_space.get_embedding_ids(states)
            input_state = states[0]
            state_input_size = self.state_space[0]['size']
            input_state = input_state[0].reshape((1, state_input_size)).astype('int32')

            print("State input to Controller for Action : ", states[0].flatten())

            prob_actions = self.policy_network(input_state)
            actions = torch.argmax(prob_actions, dim=2)

            return actions, prob_actions

    def build_policy_network(self):
        num_distinct_states = np.sum([self.state_space[i]['size'] for i in range(self.state_size)])

        self.policy_network = NasLSTM(num_embeddings=num_distinct_states, embedding_dim=self.embedding_dim,
                                hidden_dim=self.hidden_units,
                                state_sizes=self.state_space.get_state_sizes(self.num_layers))

        for name, param in self.policy_network.lstm.named_parameters():
            nn.init.xavier_uniform(param, a=-self.upper_bound, b=self.upper_bound)

        self.policy_optim = optim.Adam(self.policy_network.parameters(), lr=self.lr)

    def store_rollout(self, state, reward, prob):
        self.reward_buffer.append(reward)
        self.state_buffer.append(state)
        self.prob_buffer.append(prob)

        # dump buffers to file if it grows larger than 50 items
        if len(self.reward_buffer) > 20:
            with open('buffers.txt', mode='a+') as f:
                for i in range(20):
                    state_ = self.state_buffer[i]
                    state_list = self.state_space.parse_state_space_list(state_)
                    state_list = ','.join(str(v) for v in state_list)

                    f.write("%0.4f,%s\n" % (self.reward_buffer[i], state_list))

                print("Saved buffers to file `buffers.txt` !")

            self.reward_buffer = [self.reward_buffer[-1]]
            self.state_buffer = [self.state_buffer[-1]]

    def discount_rewards(self):
        '''
            Compute discounted rewards over the entire reward buffer
        :return: Discounted reward value
        '''
        discounted_rewards = []  # one discounted reward per step

        for t in range(len(self.reward_buffer)):
            Gt = 0
            pw = 0
            for r in self.reward_buffer[t:]:
                Gt = Gt + self.discount_factor ** pw * r
                pw += 1
            discounted_rewards.append(Gt)

        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (
                np.std(discounted_rewards) + 1e-9)  # normalize discounted rewards

        return discounted_rewards

    # todo: à revoir
    def update_policy(self):
        """
            Perform a single train step on the Controller RNN
        :return: the training loss
        """
        probs = self.prob_buffer[-1]

        # parse the state space to get real value of the states,
        # then one hot encode them for comparison with the predictions
        state_list = self.state_space.parse_state_space_list(states)

        # the discounted reward value
        rewards = self.discount_rewards()

        print("Training RNN (States ip) : ", state_list)
        print("Training RNN (Reward ip) : ", rewards)

        policy_gradient = []
        for log_prob, Gt in zip(probs, rewards):
            policy_gradient.append(-log_prob * Gt)

        self.policy_network.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.policy_network.optimizer.step()

        # reduce exploration after many train steps
        if self.global_step != 0 and self.global_step % 20 == 0 and self.exploration > 0.5:
            self.exploration *= 0.99

        self.global_step += 1

        return policy_gradient.item()

    def remove_files(self):
        files = ['train_history.csv', 'buffers.txt']

        for file in files:
            if os.path.exists(file):
                os.remove(file)
