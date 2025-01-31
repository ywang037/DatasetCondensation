# import os
from http import server
import os
import numpy as np
import random
import copy
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import copy_parameters, TensorDataset, epoch, get_time, get_daparam, get_network, evaluate_synset



class ServerDC(object):
    def __init__(self, args, net_train, clients, data_info, central_testloader=None):
        ''' whether the argument clients and attribute self.clients are needed pending further decision
        '''

        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.data_info = data_info
        self.num_classes = data_info['num_classes']
        self.channel = data_info['channel']
        self.im_size = data_info['img_size']
        self.image_syn_agg = self.server_agg_syn_data_init()
        
        self.global_model = copy.deepcopy(net_train).to(self.device)
        self.global_model_weights = copy.deepcopy(list(self.global_model.parameters()))
        self.global_model_state = copy.deepcopy(self.global_model.state_dict())
        self.clients = clients

        self.image_syn_train = None 
        self.label_syn_train = None
        self.server_syn_data = None
        self.server_trianloader = None
        self.server_testloader = central_testloader
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # self.selected_clients = []

        self.lr_net_eval_train = args.server_lr_eval_train
        self.epoch_eval_train = args.server_epoch_eval_train
        self.batch_size_eval_train = args.server_batch_eval_train
        self.save_path = os.path.join(args.save_path, 'server')

    # def select_clients(self, clients, frac=1.0):
    #     '''selects clients from possible_clients
    #     Args:
    #         clients: number of client objectes to select
        
    #     Return:
    #         list of selected clients objects
    #     '''
        
    #     if frac == 1.0:
    #         print("All users are selected")
    #         self.selected_clients = self.clients
    #         # return self.clients
    #     else:
    #         self.selected_clients = random.sample(clients, int(len(clients)*frac))
    #         # return random.sample(clients, int(len(clients)*frac)) 

    def server_agg_syn_data_init(self):
        ''' assumes that all clients have the same labels 
        '''
        image_syn_init = torch.zeros(size=(self.num_classes*self.args.ipc, self.channel, self.im_size[0], self.im_size[1]), dtype=torch.float, requires_grad=False, device=self.device)
        return image_syn_init
        # label_syn_init = torch.tensor([np.ones(self.ipc)*i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False, device='cpu').view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        # return image_syn_init, label_syn_init

    def syn_data_aggregation(self, clients):
        ''' assumes that all clients have the same labels, so that sever directly average over all the uploaded client syn data
        '''
        image_syn = self.server_agg_syn_data_init() # reset the aggregated syn data to zeros
        n_train_list = [client.num_local_data_train for client in clients]
        ratio = [n/sum(n_train_list) for n in n_train_list]
        for client in clients:
            client_syn_data = copy.deepcopy(client.image_syn.detach())
            image_syn += client_syn_data * ratio[client.id]
        self.image_syn_agg = copy.deepcopy(image_syn.detach())
        return
        
   
    def model_aggregation(self, selected_clients):
        # update the state of the global model
        global_model_state_update = self.add_net_state(clients=selected_clients)
        self.global_model.load_state_dict(global_model_state_update)
        
        # update the weight parameters of the global model
        copy_parameters(target=self.global_model_weights, source=self.global_model.parameters())
        return

    
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

    def update_server_syn_data(self, clients, server_train_batch_size=128):
        ''' this funciton concatenate the synthetic data from all the clients 
            to form the server-side synthetic data for each round of FL
        '''
        all_label = clients[0].label_syn
        all_data = clients[0].image_syn
        for id in range(1,len(clients)):
            all_label = torch.cat((all_label, clients[id].label_syn))
            all_data = torch.cat((all_data, clients[id].image_syn))
        self.image_syn_train, self.label_syn_train = copy.deepcopy(all_data.detach()), copy.deepcopy(all_label.detach()) 
        server_dataset_syn_train = TensorDataset(self.image_syn_train, self.label_syn_train)
        server_trainloader = torch.utils.data.DataLoader(server_dataset_syn_train, batch_size=server_train_batch_size, shuffle=True)
        self.server_syn_data = server_dataset_syn_train
        self.server_trianloader = server_trainloader
        return server_dataset_syn_train, server_trainloader

    def train_global_model(self, server_lr, server_train_epoch):
        ''' train server side global model using all the synthetic data uploaded from clients
        '''
        self.global_model.train()
        optimizer_server = torch.optim.SGD(self.global_model.parameters(), lr=server_lr)  # optimizer_img for synthetic data
        # criterion = nn.CrossEntropyLoss().to(self.device)
        
        # server updates the global model
        # print something to screen? 
        for r in range(server_train_epoch):
            loss_train, acc_train = epoch('train', self.server_trianloader, self.global_model, optimizer_server, self.criterion, self.args, aug = False)
        
        # # server evaluates the accuracy of the global model using the central test data
        # loss_test, acc_test = epoch('test', self.server_testloader, self.global_model, optimizer_server, self.criterion, self.args, aug = False)
        
        # TODO: server may need to check the train loss for tunining num of epochs
        # print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
        # print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))
        return

    def global_syn_data_eval_final(self, accs_server_all_exps, args):
        ''' Evaluate synthetic data after training is done
        '''
        
        for model_eval in args.model_eval_pool:
            print('-------------------------\n{} Server evaluation\nmodel_train = {}, model_eval = {}'.format(get_time(), args.model, model_eval))
            args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
            # print('DC augmentation parameters: \n', self.args.dc_aug_param)
            # self.args.epoch_eval_train = 300

            accs = []
            for it_eval in range(args.num_eval):
                # get a random model
                net_eval = get_network(model_eval, self.channel, self.num_classes, self.im_size).to(self.device) 
                
                # avoid any unaware modification
                image_syn_eval, label_syn_eval = copy.deepcopy(self.image_syn_train.detach()), copy.deepcopy(self.label_syn_train.detach()) 
                
                # trains final models using condensed/synthetic data 
                # then evaluate the accuracy of these resulting models on centralized test data
                _, loss_test, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, self.server_testloader, self.lr_net_eval_train, self.epoch_eval_train, self.batch_size_eval_train, args)
                accs.append(acc_test)
            print('{} Server evaluated %d random %s, mean accuracy = %.4f std = %.4f\n-------------------------'%(get_time(), len(accs), model_eval, np.mean(accs), np.std(accs)))

            # record the final results
            accs_server_all_exps[model_eval] += accs

        # # not sure if it is necessary to save server-side synthetic data since it is just a collection of all client synthetic data which is saved already
        # # visualize and save the server's synthtic data
        # if args.save_results:
        #     save_name = os.path.join(self.save_path, 'vis_%s_%s_%s_%dipc_exp%d_final.png'%(self.args.method, self.args.dataset, self.args.model, self.args.ipc, exp))
        #     image_syn_vis = copy.deepcopy(self.image_syn_train.detach().cpu())
        #     for ch in range(self.channel):
        #         image_syn_vis[:, ch] = image_syn_vis[:, ch]  * self.data_info['std'][ch] + self.data_info['mean'][ch]
        #     image_syn_vis[image_syn_vis<0] = 0.0
        #     image_syn_vis[image_syn_vis>1] = 1.0
        #     save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.
        return

    # def push_model_to_clients(self, clients):
    #     """ send global model parameters to a client's local cache
    #         to be used before local training steps
    #     """
    #     for client in clients:
    #         copy_parameters(target=client.model_train.parameters(), source=self.global_model_param)    

    # def add_parameters(self, model, client, ratio):
    #     """used for model aggregation, adding a client's model parameter scaled by ratio to
    #         the corresponding parameters in the input model
    #     """
    #     for server_param, client_param in zip(model, client.get_parameters()):
    #         server_param.data += client_param.data.clone() * ratio

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