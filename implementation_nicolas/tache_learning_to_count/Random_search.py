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

class Controller_evolution():
    #list_desc example : list_desc=["Number_of_filters","Filter_size","Stride"]
    def __init__(self,list_desc,dict_params):
        self.list_desc = list_desc
        self.dict_params = dict_params
        self.taille = len(list_desc)

    def generate_network(self):
        all_choix = []
        for name_generate in self.list_desc:
            liste_possibilite = dict_params[name_generate]
            choix = np.random.randint(0,len(liste_possibilite), size=1)[0]
            all_choix.append(choix)
        return all_choix
    
    def mutate_network(self,description_actuelle):
        #Muter la description
        numero_of_parametter = len(description_actuelle)
        for choix_parammettre in range(numero_of_parametter):
            alea = np.random.random()
            if(alea<0.3):
                choix_possible = self.dict_params[self.list_desc[choix_parammettre]]
                valeur_param = np.random.randint(0,len(choix_possible),1)[0]
                description_actuelle[choix_parammettre] = valeur_param
        return description_actuelle
    
    def best_in_sample(self,sample):
        maxi = -np.inf
        best_model = None
        for model in sample:
            if(model[1]>maxi):
                maxi = model[1]
                best_model=model
        return best_model



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


def learning_to_count_reward(liste_pred):
    #On rajoute 1 pour passer de 1 à n au lieux de 0 à n-1
    #Normalement on devrais passer par dict param pour etre générique mais la flemme
    #liste_pred = list(liste_pred.detach().numpy()) 
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



MAX_ITER = 2000
NB_EPOCH = 2
P = 500
S = 50
#Nombre de génération/répétition
C = 4500

list_desc=["Number_"+str(i+1) for i in range(NUMBER)]
controler = Controller_evolution(list_desc,dict_params)

#netTester = NetworkTester(NB_EPOCH)
best_model = None
Rmax = -99999999

all_reward = []

population = []
all_model = []

k=0
while len(population) < P+C:
    desc = controler.generate_network()
    R = learning_to_count_reward(desc)
    writer.add_scalars('Reward', {'Reward_Random':R}, k)
    all_reward.append(R)
    model = (desc,R)
    population.append(model)
    k+=1

