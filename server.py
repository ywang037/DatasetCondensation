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

        # # the following attributes are used for save results
        # self.dataset = dataset
        # self.num_glob_iters = num_glob_iters
        # self.local_epochs = local_epochs
        # self.batch_size = batch_size
        # self.learning_rate = learning_rate
        # self.total_train_samples = 0
        # self.beta = beta
        # self.lamda = lamda
        # self.algorithm = algorithm
        # self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc,self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        # self.times = times


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

    def model_aggregation(self, fed_model, clients):
        """ WARNING: this method has been deprecated.
            used for aggregating local model updates, model parameters from clients will be added and saved to fed_model
            fed_model can be the global model of FedAvg; or the cluster model
        arg:
            fed_model: the aggregated model to be computed
            clients: a list of client objects whose model parameters will be added up
        """
        # assert (self.users is not None and len(self.users) > 0)
        
        # set the global model to zero weights
        for param in fed_model:
            param.data = torch.zeros_like(param.data, requires_grad=False)

        # get the number of training data
        total_train = 0
        for client in clients:
            total_train += client.n_data_train
        
        # compute the ratio and conduct model aggregation
        for client in clients:
            ratio = client.n_data_train / total_train
            # self.add_parameters(fed_model, client, ratio)
            for server_param, client_param in zip(fed_model, client.get_parameters()):
                server_param.data += client_param.data.clone() * ratio
    
    def model_aggregation2(self, selected_clients):
        self.global_W = self.add_parameter_dict(clients=selected_clients)
        self.global_model.load_state_dict(self.global_W)
        copy_parameters(target=self.global_model_param, source=self.global_model.parameters())

    
    def add_parameter_dict(self, clients):
        """ an alternative way of computing model aggregation
            the same as standard FedAvg code library
        """
        
        n_train_list = [client.n_data_train for client in clients]
        ratio = [n/sum(n_train_list) for n in n_train_list]
        w_list = [copy.deepcopy(client.model.state_dict()) for client in clients]
        w_avg = w_list[0]
        for i, w in enumerate(w_list):
            for key in w.keys():
                if i==0:
                    w_avg[key] = ratio[i]*w[key]
                else:    
                    w_avg[key]+=ratio[i]*w[key]
        return w_avg

    
    # # WY ---------------from here you might need-----------------------------
    # def send_parameters(self):
    #     assert (self.users is not None and len(self.users) > 0)
    #     for user in self.users:
    #         user.set_parameters(self.model)

    # def save_model(self):
    #     model_path = os.path.join("models", self.dataset)
    #     if not os.path.exists(model_path):
    #         os.makedirs(model_path)
    #     torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    # def load_model(self):
    #     model_path = os.path.join("models", self.dataset, "server" + ".pt")
    #     assert (os.path.exists(model_path))
    #     self.model = torch.load(model_path)

    # def model_exists(self):
    #     return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    # # WY -----------------------------------------------------------------------

    # def aggregate_grads(self):
    #     assert (self.users is not None and len(self.users) > 0)
    #     for param in self.model.parameters():
    #         param.grad = torch.zeros_like(param.data)
    #     for user in self.users:
    #         self.add_grad(user, user.train_samples / self.total_train_samples)

    # def add_grad(self, user, ratio):
    #     user_grad = user.get_grads()
    #     for idx, param in enumerate(self.model.parameters()):
    #         param.grad = param.grad + user_grad[idx].clone() * ratio

