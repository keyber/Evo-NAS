import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NetworkManager:
    """
    Helper class to manage the generation of subnetwork training given a dataset
    """

    def __init__(self, dataset, device, epochs=50, acc_beta=0.8, clip_rewards=0.0,
                 weight_decay=1e-4, learning_rate=0.1, momentum=0.9, nesterov=True):
        """
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.

        :param dataset: a tuple of 4 arrays (X_train, y_train, X_val, y_val)
        :param device: device on which the code is run
        :param epochs: number of epochs to train the subnetworks
        :param acc_beta: exponential weight for the accuracy
        :param clip_rewards: float - to clip rewards in [-range, range] to prevent
                large weight updates. Use when training is highly unstable.
        :param weight_decay: weight decay for L2 penalty
        :param learning_rate: learning rate for training the subnetworks
        :param momentum: momentum factor
        :param nesterov: whether to use Nesterov momentum

        Default values for epochs, weight_decay, learning_rate, momentum and nesterov defined in 'Neural Architecture
        Search With Reinforcement Learning' (Zoph and Le, ICLR 2017)
        """

        self.dataset = dataset
        self.epochs = epochs
        self.clip_rewards = clip_rewards

        self.device = device

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

        self.criterion = nn.CrossEntropyLoss()

        self.beta = acc_beta
        self.beta_bias = acc_beta
        self.moving_acc = 0.0


    def evaluate(self, network, data):
        network.eval()
        network = network.float()
        losses = []
        accuracies = []

        with torch.no_grad():
            for _, (input, target) in enumerate(data):
                input = input.to(self.device)
                target = target.to(self.device)

                output = network(input)

                loss = self.criterion(output, target)

                losses.append(loss.item())

                correct = torch.sum(torch.argmax(output, dim=-1) == target).item()
                batch_accuracy = correct / len(target)

                accuracies.append(batch_accuracy)

        return np.mean(losses), np.mean(accuracies)

    def train_network(self, network, optimizer, train, valid):
        """
            Trains the given network using train data and evaluates performance with valid data

        :param network: child model to train
        :param optimizer: optimizer object
        :param train: dataloader storing training data
        :param valid: dataloader storing validation data
        :return: best accuracy computed on validation data along with the corresponding model weights
        """
        best_acc = 0
        train_loss = None

        network.train()
        print("Training child model...")
        for epoch in range(self.epochs):
            print("Epoch {}, last training loss={}".format(epoch, train_loss))
            train_loss = []
            for _, (input, target) in enumerate(train):
                optimizer.zero_grad()

                input = input.to(self.device)
                target = target.to(self.device)

                output = network(input)

                loss = self.criterion(output, target)
                train_loss.append(loss.item())
                loss.backward()

                optimizer.step()

            train_loss = np.mean(train_loss)
            if epoch in range(self.epochs - 5, self.epochs):
                mean_test_loss, mean_test_acc = self.evaluate(network, valid)

                if mean_test_acc > best_acc:
                    best_acc = mean_test_acc
                    # best_params = network.state_dict()

        return np.power(best_acc, 3)

    def get_rewards(self, model):
        """
            Creates a subnetwork given the actions predicted by the controller RNN,
            trains it on the provided dataset, and then returns a reward.

        :param model: a child network
        :return: a reward for training a model with the given actions along with the best computed accuracy
        """
        # unpack the dataset
        train_dataloader, valid_dataloader, test_dataloader = self.dataset

        # set the optimizer
        optimizer = optim.SGD(model.parameters(), weight_decay=self.weight_decay, lr=self.learning_rate,
                              momentum=self.momentum, nesterov=self.nesterov)

        # train and evaluate the model
        acc = self.train_network(model, optimizer, train_dataloader, valid_dataloader)

        # compute the reward
        reward = (acc - self.moving_acc)

        # if rewards are clipped, clip them in the range -0.05 to 0.05
        if self.clip_rewards:
            reward = np.clip(reward, -0.05, 0.05)

        # update moving accuracy with bias correction for 1st update
        if 0.0 < self.beta < 1.0:
            self.moving_acc = self.beta * self.moving_acc + (1 - self.beta) * acc
            self.moving_acc = self.moving_acc / (1 - self.beta_bias)
            self.beta_bias = 0

            reward = np.clip(reward, -0.1, 0.1)

        print("Manager: EWA Accuracy = ", self.moving_acc)

        return reward, acc

