import torch
import torchvision
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler
from PIL import Image
import json

import json
import os, copy
import random
from torch.utils.data import DataLoader, Dataset
import numpy as np


class MNISTDataset(Dataset):
    """EMNIST dataset"""
    def __init__(self, feature, target, transform=None):
        # self.X = []
        self.Y = target
        self.transform = transform
        self.X = feature
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.X[idx]), self.Y[idx]
        return self.X[idx], self.Y[idx]


def femnist_dataloaders(root="./femnist", batch_size=64, clients=10):
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_data = torch.load(os.path.join(root, "train_data.pt"))
    test_data = torch.load(os.path.join(root, "test_data.pt"))
    
    # flatten image format

    if isinstance(clients, int):
        print("Filter clients by position.")
        if clients > len(train_data['users']):
            raise ValueError(
                "Request clients({}) larger than dataset provide({}).".format(clients, len(train_data['users'])))

        train_data['users'] = train_data['users'][:clients]
        test_data['users'] = test_data['users'][:clients]
    elif isinstance(clients, list):
        print("Filter clients by name_list.")
        for name in clients:
            if name not in train_data['users']:
                raise ValueError("Client {} not found in dataset.".format(name))
        train_data['users'] = clients

    #############################################################################
    # in this script we will load all clients' data into a datalist.
    
    # dict format:
    
    # train_data = {
    #   "users": ['f4015_05', 'f4067_23', ......],
    #   "num_samples": [154, 162, 121, ......]
    #   "user_data": {
    #       'f4015_05':{
    #           "x": [[1.0, 0.99, 0.59, 1.0, 0.96, 0.41, ,...], [...]],
    #           "y": [1,12,4, ....],
    #                  }
    #       'f4067_23':{
    #           "x": [[0.98, 0.99, 0.19, 1.0, 0.76, 0.42, ,...], [...]],
    #           "y": [19,22,48, ....],
    #                  }
    #  }
    #}
    
    # total 3560 clients.
    # In LAB3 we take first 50 clients
    
    #############################################################################
    
    # example dataloader
    example_dataset = MNISTDataset(feature=torch.tensor(train_data['user_data']['f4015_05']['x']).view(-1, 28, 28),    
                                   target = torch.tensor(train_data['user_data']['f4015_05']['y']), 
                                   transform=data_transform)

    example_dataloader = DataLoader(example_dataset, batch_size=32, shuffle=True)

    #############################################################################
    train_data_all_x = []
    train_data_all_y = []
    train_idx = []

    trainloaders = []
    for u in train_data["users"]:
        #train_data["user_data"][u]['x']
        x = torch.tensor(train_data['user_data'][u]['x']).view(-1, 28, 28)
        y = torch.tensor(train_data['user_data'][u]['y'])

        train_data_all_x += x
        train_data_all_y += y

        traset = MNISTDataset(feature=x, target = y, transform=data_transform)
        traloader = torch.utils.data.DataLoader(traset, batch_size=batch_size, shuffle=False)
        trainloaders.append(traloader)

    test_data_all_x = []
    test_data_all_y = []
    test_idx = []
    testloaders = []
    for i in test_data["users"]:
        # cut data to fit batch
        x = torch.tensor(test_data['user_data'][i]['x']).view(-1, 28, 28)
        y = torch.tensor(test_data['user_data'][i]['y'])

        test_data_all_x += x
        test_data_all_y += y
        test_idx.append(len(x))

        testset = MNISTDataset(feature=x, target = y, transform=data_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        trainloaders.append(testloader)

    #############################################################################
    train_dataset =  MNISTDataset(feature=train_data_all_x, target =train_data_all_y, transform=data_transform)
    train_data_global = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = MNISTDataset(feature = test_data_all_x, target = test_data_all_y, transform=data_transform)
    test_data_global = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    #############################################################################
    
    train_data_local_dict = {}
    for i,v in enumerate(trainloaders):
        train_data_local_dict[i] = v

    
    test_data_local_dict = {}
    for i,v in enumerate(testloaders):
        test_data_local_dict[i] = v        
    #############################################################################
    train_data_local_num_dict = {}
    
    train_data_num = 0
    for k in train_data_local_dict.keys():
        train_data_local_num_dict[k] = len(train_data_local_dict[k].dataset)
        train_data_num += len(trainloaders[k].dataset)

    test_data_num = len(testloader.dataset)
    #############################################################################

    # [
    #     train_data_num,             -> totoal image in train dataset
    #     test_data_num,              -> totoal image in test dataset
    #     train_data_global,          -> dataloader with all train dataset
    #     test_data_global,           -> dataloader with all test dataset
    #     train_data_local_num_dict,  -> dict of amount of train data in each clients. ex: {0:123, 1:80, 2:231, ...}
    #     train_data_local_dict,      -> dict of amount of train data in each clients. ex: {0:<dataloader>, 1:<dataloader>, ...}
    #     test_data_local_dict,       -> dict of amount of test data in each clients. ex: {0:<dataloader>, 1:<dataloader>, ...}
    #     class_num                   -> mumber of calss. femnist: 62
    # ]

    class_num = 62
    
    dataset =  [
        train_data_num, 
        test_data_num, 
        train_data_global, 
        test_data_global,
        train_data_local_num_dict, 
        train_data_local_dict, 
        test_data_local_dict, 
        class_num
    ]
    return dataset