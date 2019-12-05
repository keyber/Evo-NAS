import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()



class ConvNetwork(nn.Module):
    #networkDesc est une liste de couple avec comme premier element le nom du paramettre et le deuxieme l'indice dans le dict
    #Exemple : [("Number_of_filters",0),("Filter_size",3),("Stride",2)]
    def __init__(self, networkDesc, dict_params):
        #print(dict_params)
        #print(networkDesc)
        super(ConvNetwork, self).__init__()
        i = 0
        self.out_channels = dict_params[networkDesc[i][0]][networkDesc[i][1]]
        i+=1
        self.kernel_size = dict_params[networkDesc[i][0]][networkDesc[0][1]]
        i+=1
        self.stride = dict_params[networkDesc[i][0]][networkDesc[0][1]]

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.out_channels , kernel_size=self.kernel_size, stride=self.stride)
        self.pool = nn.MaxPool2d(2, 2)
        #Peut etre a revoir la taille d'entrée
        #print("stride",self.stride)
        #print("kernel size",self.kernel_size)
        #Formule generale avec le padding : partie_sup( (x+2*p-k+1) / s )
        taille_sortie_conv_pool = int (ceil( (32-self.kernel_size+1) / self.stride ) / 2 )
        #print("ouput conv", taille_sortie_conv_pool  )
        self.fc1 = nn.Linear(self.out_channels*taille_sortie_conv_pool*taille_sortie_conv_pool, 10)

    def forward(self, x):
        #print(x.shape)
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        #Equivalent flatten
        x = x.view(batch_size,-1)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        return x

class ControllerNetwork(nn.Module):
    #list_desc example : list_desc=["Number_of_filters","Filter_size","Stride"]
    def __init__(self,list_desc,dict_params, hidden_size, num_layers):
        super(ControllerNetwork, self).__init__()
        self.list_desc = list_desc
        self.dict_params = dict_params
        self.taille = len(list_desc)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #Le input size est de la taille de l'embedding
        self.rnn = nn.LSTM(1, hidden_size, num_layers)
        self.list_embed_weights = []
        for desc in self.list_desc:
            self.list_embed_weights.append(nn.Linear(hidden_size, len(dict_params[desc])))
        self.list_embed_weights = nn.ModuleList(self.list_embed_weights)
        self.soft = nn.Softmax(dim=0)


    def forward(self, inutile):
        h = torch.zeros([self.num_layers, 1, self.hidden_size],requires_grad=True)
        c = torch.zeros([self.num_layers, 1, self.hidden_size],requires_grad=True)
        x = torch.zeros([1,1,1],requires_grad=True)

        #print("h req",h.requires_grad)
        #print("c req",c.requires_grad)
        all_decision = torch.tensor([],requires_grad=True)
        all_probs = torch.tensor([],requires_grad=True)
        #print("requires grad des concat")
        #print(all_decision.requires_grad)
        #On fait N passage ou seq_len est égale à 1 pour chaqu'un car c'est la sortie que l'on remet dans le réseaux
        for embed in self.list_embed_weights:
            _, (h_n, c_n) = self.rnn(x, (h, c))
            h = h_n[-1,0]
            #c = c_n[-1,0]
            before_soft = embed(h)
            #print(before_soft.shape)
            decision = self.soft(before_soft)
            #print("Distribution de proba sur un des paramettres")
            #print(decision)
            #Attention : Peut etre sampling ou max
            prob_dist = torch.distributions.Categorical(decision)
            indice_max = prob_dist.sample()
            #print(indice_max)
            proba = decision[indice_max]
            #print(proba)
            indice_max = indice_max.float()
            #proba = decision.max(0)[0].float()
            #indice_max = decision.max(0)[1].float()
            #print("proba requires grad",proba.requires_grad)
            
            all_decision = torch.cat((all_decision, indice_max.unsqueeze(0)))
            all_probs = torch.cat((all_probs,proba.unsqueeze(0)))
            x = indice_max
            x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            h = h_n
            c = c_n
        #print("allprobs requires grad",all_probs.requires_grad)
        return all_decision.int(),all_probs


class NetworkTester():
    def __init__(self,nb_epoch):
        self.nb_epoch = nb_epoch
        #Définition des différentes transformations que l'on effectue sur les images
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        #Chargement de la base CIFAR10 pour le train,validation et test
        #Chargement train et val set
        allset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
       
       #On utilise que 10% pour le train pour tester plus rapidement et 70% inutilisé -> pourrais etre utilisé pour l'éval !!!
        sizes = [int(.1*len(allset)), int(.2*len(allset)), int(.7*len(allset))]
        
        train_set, val_set, unuse_set = torch.utils.data.random_split(allset, sizes)
        #Transformation en dataloader
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=True, num_workers=2)

        #Chargement du test set
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
        #Transformation en dataloader
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=2)

        self.criterion = nn.CrossEntropyLoss()

    #Prend un dataloader avec comme premier element tout le dataset, cad batch_size=size_dataset, pour le test et le val
    def accuracy(self,network,dataloader):
        #print(type(dataloader))
        all_data = next(iter(dataloader))
        data = all_data[0]
        target = all_data[1]
        #print(data)
        #print(type(data))
        #print(data.shape)
        output = network(data)
        max_index = output.max(dim = 1)[1]
        #print(max_index)
        #print(target)
        nb_correct = (max_index == target).sum()
        #print(nb_correct)
        return int(nb_correct.numpy())/len(data)


    def get_reward(self,net):
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        for epoch in range(self.nb_epoch):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        return self.accuracy(net,self.val_loader)

    def test_score(self,net):
        return self.accuracy(net,self.test_loader)


def update_controler(controler, optimizer, reward, log_probs,epoch,nb_traj_echant,GAMMA=0.99):
    nb_hyperparams = len(log_probs)
    rewards = [reward]*nb_hyperparams
    #Ancienne facon
    """
    discounted_rewards = []
    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)
    """
    #new discounted
    discounted_rewards = rewards

    #Use a baseline
    #discounted_rewards = torch.tensor(discounted_rewards)
    #discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    #print("variable reinforce")
    #print(log_probs)
    #print(discounted_rewards)
    #print(policy_gradient)

    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    
    if( (epoch+1)%nb_traj_echant == 0):
        optimizer.step()
        optimizer.zero_grad()
   

def learning_to_count_reward(liste_pred):
    #On rajoute 1 pour passer de 1 à n au lieux de 0 à n-1
    #Normalement on devrais passer par dict param pour etre générique mais la flemme
    liste_pred = list(liste_pred.detach().numpy()) 
    liste_pred = [q+1 for q in liste_pred]
    #print(liste_pred)
    n = len(liste_pred)
    somme = liste_pred[0]**2
    for k in range(len(liste_pred)-1):
        somme+=(liste_pred[k+1]-liste_pred[k])**2
    somme+=(liste_pred[-1] - (n+1) )**2
    return (n+1)/somme


NUMBER = 8
dict_params = dict()
for s in range(NUMBER):
    dict_params["Number_"+str(s+1)] = list(range(NUMBER))



MAX_ITER = 5000
HIDDEN_SIZE = 200
NUM_LAYER = 1
NB_EPOCH = 2
#Equivalent de m dans la papier
NB_TRAJECTOIRE_ECHANTILLON = 6

list_desc=["Number_"+str(i+1) for i in range(NUMBER)]
controler = ControllerNetwork(list_desc,dict_params,HIDDEN_SIZE,NUM_LAYER)
#Change weight init of the controller
def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(-0.08 ,0.08)
        m.bias.data.fill_(0)
#Initialize linear weight
controler.apply(weights_init_uniform)
#Initialize LSTM weight
for layer_p in controler.rnn._all_weights:
    for p in layer_p:
        if 'weight' in p:
            weights_init_uniform(controler.rnn.__getattr__(p))
"""
for name, param in controler.named_parameters():
    if param.requires_grad:
        print(name, param.data)
"""

#Attention lr trop HAUT, juste pour le test
controler_optimizer = optim.Adam(controler.parameters(), lr=5e-4)
#netTester = NetworkTester(NB_EPOCH)
best_model = None
Rmax = -99999999

all_reward = []

somme = None
for k in range(MAX_ITER):
    indice_choix_controller,log_probs = controler(torch.tensor([]))
    if(k%100==0):
        print("Itération ",k,"/",MAX_ITER)
        print("choix controller :",indice_choix_controller)
    networkDesc = []
    for q in range(len(indice_choix_controller)):
        couple = (list_desc[q],int(indice_choix_controller[q].detach().numpy()))
        networkDesc.append(couple)
    #On multiplie la reward par 100 pour avoir un chiffre entre 0 et 100 au lieu de 0 et 1
    R = learning_to_count_reward(indice_choix_controller)
    writer.add_scalars('Reward', {'Reward_RL_REINFORCE':R}, k)

    all_reward.append(R)
    if(k%100==0):
        print("Reward :",R)
    if(R>Rmax):
        #best_model = model_to_test
        Rmax = R
    
    if(somme is None):
        somme = R
    else:
        R -= (somme/k)
        somme += R 

    update_controler(controler, controler_optimizer, R, log_probs, k , NB_TRAJECTOIRE_ECHANTILLON)

#print(netTester.test_score(best_model))

#plt.plot(all_reward)
#plt.show()

filtre_passe_bas = []
RANGE = 10
for k in range(len(all_reward)-RANGE):
    filtre_passe_bas.append(np.mean(all_reward[k:k+RANGE]))

plt.plot(filtre_passe_bas)
plt.show()