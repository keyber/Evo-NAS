import torch
from stateSpace import StateSpace
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
import csv
from child_network import ChildNetwork
from manager import NetworkManager
from controller import Controller
from torch.utils.data.sampler import SubsetRandomSampler

datasets.CIFAR10.url = "http://webia.lip6.fr/~robert/cours/rdfia/cifar-10-python.tar.gz"
CUDA = False

NUM_LAYERS = 4  # number of layers of the state space
MAX_TRIALS = 250  # maximum number of models generated

CHILD_BATCHSIZE = 64  # batchsize of the child models
                      # defined in 'Densely Connected Convolutional Networks' (Huang et al., CVPR 2016)
EXPLORATION = 0.8  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength
HIDDEN_UNITS = 35 # number of hidden units on each layer
EMBEDDING_DIM = 20  # dimension of the embeddings for each state
ACCURACY_BETA = 0.8  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range
RESTORE_CONTROLLER = True  # restore controller to continue training


def get_dataset(batch_size, path):
    '''
        loads dataset and transforms each image
    :param batch_size: batch size for each data loader
    :param path: data path
    :return: train dataloader, validation dataloader, test dataloader
    '''
    """
    Cette fonction charge le dataset et effectue des transformations sur chaqu
    image (listées dans `transform=...`).
    """

    train_dataset = datasets.CIFAR10(path, train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=None, pad_if_needed=False, fill=0,
                                                               padding_mode='constant'),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
                                     ])) # todo: whiten all images as part of the preprocessing
    train_dataset, val_dataset = random_split(train_dataset,
                                                (int(0.1 * len(train_dataset)),
                                                 int(0.9 * len(train_dataset))))
                                    # not sure whether to apply center crop or random crop on validation data
    test_dataset = datasets.CIFAR10(path, train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.CenterCrop(32),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
                                   ])) # todo: whiten all images as part of the preprocessing

    train_subset = torch.randint(0, len(train_dataset), (100,))
    val_subset = torch.randint(0, len(val_dataset), (100,))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, #shuffle=True)
                                               sampler=SubsetRandomSampler(train_subset))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size, #shuffle=False)
                                             sampler = SubsetRandomSampler(val_subset))

    return train_loader, val_loader, test_loader

def main_training():
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = torch.cuda.device_count()
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    train_loader, val_loader, test_loader = get_dataset(CHILD_BATCHSIZE, '/tmp/datasets/cifar-10')
    dataloaders = [train_loader, val_loader, test_loader]

    # construct a state space
    state_space = StateSpace()

    # add states
    state_space.add_state(name='kernel', values=[1, 3, 5, 7])
    state_space.add_state(name='filters', values=[24, 36, 48, 64])

    previous_acc = 0.0
    total_reward = 0.0

    controller = Controller(NUM_LAYERS, state_space)
    manager = NetworkManager(dataloaders, device, epochs=2)

    # get an initial random state space if controller needs to predict an action from the initial state
    state = state_space.get_random_state_space(NUM_LAYERS)
    print("Initial Random State : ", state_space.parse_state_space_list(state))

    for trial in range(MAX_TRIALS):
        actions, prob_actions = controller.get_action(state)  # get an action for the previous state

        # print the action probabilities
        state_space.print_actions(actions)
        print("Predicted actions : ", state_space.parse_state_space_list(actions))

        # build a model, train and get reward and accuracy from the network manager
        model = ChildNetwork(state_space.parse_state_space_list(actions))
        reward, previous_acc = manager.get_rewards(model)
        print("Rewards : ", reward, "Accuracy : ", previous_acc)

        total_reward += reward
        print("Total reward : ", total_reward)

        # actions and states are equivalent, save the state and reward
        state = actions
        prob_state = prob_actions
        controller.store_rollout(state, reward, prob_state)

        # train the controller on the saved state and the discounted rewards
        loss = controller.update_policy()
        print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))

        # write the results of this trial into a file
        with open('train_history.csv', mode='a+') as f:
            data = [previous_acc, reward]
            data.extend(state_space.parse_state_space_list(state))
            writer = csv.writer(f)
            writer.writerow(data)

        print()
        print("Total Reward : ", total_reward)

if __name__ == '__main__':
    main_training()
