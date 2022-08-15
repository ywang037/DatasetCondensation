# import os
import imp
import random
import copy
import torch
from utils import copy_parameters
# from torch.utils.data import DataLoader
# import numpy as np
# from sklearn.cluster import AgglomerativeClustering, SpectralClustering

class ServerDC(object):
    def __init__(self, args, net_train, clients):

        # Set up the main attributes
        self.device = args.device
        self.global_model = copy.deepcopy(net_train).to(self.device)
        self.global_model_weights = copy.deepcopy(list(self.global_model.parameters()))
        self.global_model_state = copy.deepcopy(self.global_model.state_dict())
        self.clients = clients
        self.selected_clients = []

    def select_clients(self, clients, frac=1.0):
        '''selects clients from possible_clients
        Args:
            clients: number of client objectes to select
        
        Return:
            list of selected clients objects
        '''
        
        if frac == 1.0:
            print("All users are selected")
            self.selected_clients = self.clients
            # return self.clients
        else:
            self.selected_clients = random.sample(clients, int(len(clients)*frac))
            # return random.sample(clients, int(len(clients)*frac)) 

    def push_model_to_clients(self, clients):
        """ send global model parameters to a client's local cache
            to be used before local training steps
        """
        for client in clients:
            copy_parameters(target=client.model_train.parameters(), source=self.global_model_param)    

    def add_parameters(self, model, client, ratio):
        """used for model aggregation, adding a client's model parameter scaled by ratio to
            the corresponding parameters in the input model
        """
        for server_param, client_param in zip(model, client.get_parameters()):
            server_param.data += client_param.data.clone() * ratio
   
    def net_weights_aggregation(self, selected_clients):
        self.global_model_state = self.add_net_state(clients=selected_clients)
        self.global_model.load_state_dict(self.global_model_state)
        copy_parameters(target=self.global_model_weights, source=self.global_model.parameters())

    
    def add_net_state(self, clients):
        """ an alternative way of computing model aggregation
            the same as standard FedAvg code library
        """
        n_train_list = [client.num_local_data_train for client in clients]
        ratio = [n/sum(n_train_list) for n in n_train_list]
        w_list = [client.local_model_state for client in clients]
        w_avg = w_list[0]
        for i, w in enumerate(w_list):
            for key in w.keys():
                if i==0:
                    w_avg[key] = ratio[i]*w[key]
                else:    
                    w_avg[key]+=ratio[i]*w[key]
        return w_avg

    # def model_aggregation(self, fed_model, clients):
    #     """ WARNING: this method has been deprecated.
    #         used for aggregating local model updates, model parameters from clients will be added and saved to fed_model
    #         fed_model can be the global model of FedAvg; or the cluster model
    #     arg:
    #         fed_model: the aggregated model to be computed
    #         clients: a list of client objects whose model parameters will be added up
    #     """
    #     # assert (self.users is not None and len(self.users) > 0)
        
    #     # set the global model to zero weights
    #     for param in fed_model:
    #         param.data = torch.zeros_like(param.data, requires_grad=False)

    #     # get the number of training data
    #     total_train = 0
    #     for client in clients:
    #         total_train += client.n_data_train
        
    #     # compute the ratio and conduct model aggregation
    #     for client in clients:
    #         ratio = client.n_data_train / total_train
    #         # self.add_parameters(fed_model, client, ratio)
    #         for server_param, client_param in zip(fed_model, client.get_parameters()):
    #             server_param.data += client_param.data.clone() * ratio