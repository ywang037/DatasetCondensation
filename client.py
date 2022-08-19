import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import copy

from utils import get_network,  evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, copy_parameters
# from utils import get_loops, get_dataset, get_eval_pool, DiffAugment, ParamDiffAug
          
class ClientDC(object):
    def __init__(self, id, args, net_train, data_info, data_train, data_test, eval_it_pool, model_eval_pool, central_testloader=None):
        
        self.id = id  # integer
        self.args = args
        self.device = self.args.device
        self.ipc = self.args.ipc
        self.eval_it_pool = eval_it_pool
        self.model_eval_pool = model_eval_pool
        self.data_info = data_info
        self.num_classes = data_info['num_classes']
        self.channel = data_info['channel']
        self.im_size = data_info['img_size']

        self.model_train = copy.deepcopy(net_train).to(self.device)
        # self.model_test = self.init_model_test(model)
        # the followings are lists storing model parameters, not the trainable model
        self.model_cache = copy.deepcopy(list(self.model_train.parameters()))
        self.local_model_weights = copy.deepcopy(list(self.model_train.parameters()))
        self.local_model_state = copy.deepcopy(self.model_train.state_dict())

        self.local_data_train = data_train
        self.local_data_test = data_test
        self.num_local_data_train = len(self.local_data_train)
        self.num_local_data_test = len(self.local_data_test)
        self.batch_size_learn_data = args.client_batch_train_data
        self.batch_size_learn_model = args.client_batch_train_model
        # self.trainloader = DataLoader(self.local_data_train, batch_size=self.batch_size, shuffle=True)
        self.local_testloader = DataLoader(self.local_data_test, batch_size=256, shuffle=False)        
        
        self.images_all = []
        self.labels_all = []
        self.indices_class = [[] for c in range(self.num_classes)]
        self.image_syn = None
        self.label_syn = None
                
        self.criterion = nn.CrossEntropyLoss().to(args.device)
        self.optimizer_img = None
        self.optimizer_net = None
        # optimizer_img.zero_grad()
      
        self.accs_all_exps = dict()
        self.data_save =[]
        self.save_path = os.path.join(self.args.save_path, 'client_{}'.format(self.id))
        self.loss_avg = 0

        self.lr_net_eval_train = args.lr_net_eval_train
        self.epoch_eval_train = args.client_epoch_eval_train
        self.batch_size_eval_train = args.client_batch_eval_train
        self.central_testloader = central_testloader

   
    def organize_local_real_data(self):
        ''' organize the real dataset '''
        # images_all = []
        # labels_all = []
        # indices_class = [[] for c in range(self.num_classes)]

        self.images_all = [torch.unsqueeze(self.local_data_train[i][0], dim=0) for i in range(self.num_local_data_train)]
        self.labels_all = [self.local_data_train[i][1] for i in range(len(self.local_data_train))]
        for i, label in enumerate(self.labels_all):
            self.indices_class[label].append(i)
        self.images_all = torch.cat(self.images_all, dim=0).to(self.device)
        self.labels_all = torch.tensor(self.labels_all, dtype=torch.long, device=self.device)

        for c in range(self.num_classes):
            print('class c = %d: %d real images'%(c, len(self.indices_class[c])))
        return

    def get_images(self, c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
        return self.images_all[idx_shuffle]
   
    def syn_data_init(self):
        ''' initialize the synthetic data '''
        self.image_syn = torch.randn(size=(self.num_classes*self.ipc, self.channel, self.im_size[0], self.im_size[1]), dtype=torch.float, requires_grad=True, device=self.device)
        self.label_syn = torch.tensor([np.ones(self.ipc)*i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False, device=self.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if self.args.init == 'real':
            print('{} Client {} initialized synthetic data from random real images'.format(get_time(), self.id))
            for c in range(self.num_classes):
                self.image_syn.data[c*self.ipc:(c+1)*self.ipc] = self.get_images(c, self.ipc).detach().data
        else:
            print('{} Client {} initialized synthetic data from random noise'.format(get_time(), self.id))
        return

    def syn_data_eval(self, exp, it, accs_all_clients_all_exps):
        ''' Evaluate synthetic data for specific iterations
        '''
        if it in self.eval_it_pool:
            for model_eval in self.model_eval_pool:
                print('-------------------------\n{} Client {} evaluation\nmodel_train = {}, model_eval = {}, iteration = {}'.format(get_time(), self.id, self.args.model, model_eval, it))
                self.args.dc_aug_param = get_daparam(self.args.dataset, self.args.model, model_eval, self.args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                # print('DC augmentation parameters: \n', self.args.dc_aug_param)
                self.args.epoch_eval_train = 300

                accs = []
                for it_eval in range(self.args.num_eval):
                    # get a random model
                    net_eval = get_network(model_eval, self.channel, self.num_classes, self.im_size).to(self.device) 
                    
                    # avoid any unaware modification
                    image_syn_eval, label_syn_eval = copy.deepcopy(self.image_syn.detach()), copy.deepcopy(self.label_syn.detach()) 
                    
                    # trains new models using condensed/synthetic data then evaluate the accuracy of this resulting model
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, self.local_testloader, self.args)
                    accs.append(acc_test)
                print('{} Client {} evaluated %d random %s, mean accuracy = %.4f std = %.4f\n-------------------------'%(get_time(), self.id, len(accs), model_eval, np.mean(accs), np.std(accs)))

                if it == self.args.Iteration: # record the final results
                    accs_all_clients_all_exps[self.id][model_eval] += accs

            # visualize and save the synthtic data (of each client)
            if self.args.save_results:
                save_name = os.path.join(self.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(self.args.method, self.args.dataset, self.args.model, self.args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(self.image_syn.detach().cpu())
                for ch in range(self.channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * self.data_info['std'][ch] + self.data_info['mean'][ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=self.args.ipc) # Trying normalize = True/False may get better visual effects.
        return

    def syn_data_eval_final(self, exp, accs_all_clients_all_exps, args):
        ''' Evaluate synthetic data after training is done
        '''
        
        for model_eval in self.model_eval_pool:
            print('-------------------------\n{} Client {} evaluation\nmodel_train = {}, model_eval = {}'.format(get_time(), self.id, args.model, model_eval))
            args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
            # print('DC augmentation parameters: \n', self.args.dc_aug_param)
            # self.args.epoch_eval_train = 300

            accs = []
            for it_eval in range(args.num_eval):
                # get a random model
                net_eval = get_network(model_eval, self.channel, self.num_classes, self.im_size).to(args.device) 
                
                # avoid any unaware modification
                image_syn_eval, label_syn_eval = copy.deepcopy(self.image_syn.detach()), copy.deepcopy(self.label_syn.detach()) 
                
                # trains final models using condensed/synthetic data 
                # then evaluate the accuracy of these resulting models on centralized test data, rather than local test data
                _, loss_test, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, self.central_testloader, self.lr_net_eval_train, self.epoch_eval_train, self.batch_size_eval_train, args)
                accs.append(acc_test)
            print('{} Client {} evaluated %d random %s, mean accuracy = %.4f std = %.4f\n-------------------------'%(get_time(), self.id, len(accs), model_eval, np.mean(accs), np.std(accs)))

            # record the final results
            accs_all_clients_all_exps[self.id][model_eval] += accs

        # visualize and save the synthtic data (of each client)
        if args.save_results:
            save_name = os.path.join(self.save_path, 'vis_%s_%s_%s_%dipc_exp%d_final.png'%(args.method, args.dataset, args.model, args.ipc, exp))
            image_syn_vis = copy.deepcopy(self.image_syn.detach().cpu())
            for ch in range(self.channel):
                image_syn_vis[:, ch] = image_syn_vis[:, ch]  * self.data_info['std'][ch] + self.data_info['mean'][ch]
            image_syn_vis[image_syn_vis<0] = 0.0
            image_syn_vis[image_syn_vis>1] = 1.0
            save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.
        return

    def data_trainer_setup(self):
        ''' to setup the optimizer for the synthetic data, 
            to be called after synthetic data initialization
        '''
        self.optimizer_img = torch.optim.SGD([self.image_syn, ], lr=self.args.lr_img, momentum=0.5) # optimizer for synthetic data

    def net_trainer_setup(self, net):
        ''' sample a network initialization and set the optimizer for the network
        '''
        # net = get_network(self.args.model, self.channel, self.num_classes, self.im_size).to(self.device) # get a random model, better to rename net to net_train
        # net.train()
        # optimizer_net = torch.optim.SGD(net.parameters(), lr=self.args.lr_net)  # optimizer_img for synthetic data
        # optimizer_net.zero_grad()
        self.optimizer_net = torch.optim.SGD(net.parameters(), lr=self.args.lr_net)  # optimizer_img for synthetic data
        self.loss_avg = 0
        self.args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.
        # return optimizer_net

    def syn_data_update(self, net):
        # NOTE this loop is over labels, i.e., line 5-8 in Algorithm 1
        # NOTE that, the following synthetic data learning process is looped over only one batch of real data, i.e., one step of SGD for learning data, 
        # in actual one can use multiple steps by looping the following sections, but one may need to use non-replace method to randomly draw real data, just like a data loader
        loss = torch.tensor(0.0).to(self.device) # batch train loss
        for c in range(self.num_classes):
            # sample a batch of real data and labels
            img_real = self.get_images(c, self.batch_size_learn_data)
            lab_real = torch.ones((img_real.shape[0],), device=self.device, dtype=torch.long) * c
            
            # sample (all) synthetic data
            img_syn = self.image_syn[c*self.args.ipc:(c+1)*self.args.ipc].reshape((self.args.ipc, self.channel, self.im_size[0], self.im_size[1]))
            lab_syn = torch.ones((self.args.ipc,), device=self.device, dtype=torch.long) * c

            net_parameters = list(net.parameters())
            output_real = self.model_train(img_real)
            loss_real = self.criterion(output_real, lab_real)
            gw_real = torch.autograd.grad(loss_real, net_parameters)
            gw_real = list((_.detach().clone() for _ in gw_real))

            output_syn = self.model_train(img_syn)
            loss_syn = self.criterion(output_syn, lab_syn)
            gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

            loss += match_loss(gw_syn, gw_real, self.args)

        self.optimizer_img.zero_grad()
        loss.backward()
        self.optimizer_img.step()
        self.loss_avg += loss.item()

    def network_update(self, net, optimizer_net, server=None):
        ''' update network '''
        if (server is not None) and (not self.args.stand_alone):
            # in federated mode and use server aggregated syn data
            # assume all clients share common labels
            image_syn_train, label_syn_train = copy.deepcopy(server.image_syn_agg.detach()), copy.deepcopy(self.label_syn.detach())
        else:
            # use own syn data
            image_syn_train, label_syn_train = copy.deepcopy(self.image_syn.detach()), copy.deepcopy(self.label_syn.detach())          
        dataset_syn_train = TensorDataset(image_syn_train, label_syn_train)
        net_trainloader = torch.utils.data.DataLoader(dataset_syn_train, batch_size=self.batch_size_learn_model, shuffle=True, num_workers=0)
        
        # the following loop is iterated for local model updates by SGD, 
        # i.e., line 9 in Algorithm 1 the args.inner_loop controls number of SGD steps, 
        # because number of sythetic data is small whereas the batch size is large by default, e.g., 256
        # so that one epoch may only have one SGD update using all the available synthetic data
        # in this case, the number of client_epoch_train equals the number of local update
        for il in range(self.args.client_epoch_train):
            epoch('train', net_trainloader, net, optimizer_net, self.criterion, self.args, aug = False)
            

    '''below are client functions for federated learning'''
    def sync_with_server(self, server, method='state'):
        """ receive/copy global model of last round to a local cache
            to be used for local training steps
        """
        
        if method == 'weights': # download the weight parameters of the global model
            copy_parameters(target=self.model_train.parameters(), source=server.global_model_weights)    
        elif method == 'state': # download the state of the global model
            self.model_train.load_state_dict(server.global_model_state)

    # '''the following are copied from clientbase'''
    # def init_model_test(self, model):
    #     m = copy.deepcopy(model).to(self.device)
    #     for p in m.parameters():
    #         p.detach()
    #     return m

    # def init_data_val(self, data_val):
    #     if data_val is not None:
    #         local_data_val = data_val
    #         n_data_val = len(data_val)
    #         valloader = DataLoader(data_val, batch_size=self.batch_size, shuffle=False)
    #     else:
    #         local_data_val = None
    #         n_data_val = 0
    #         valloader = None
    #     return local_data_val, n_data_val, valloader

    # def evaluation2(self, model_weights, input_option='param'):
    #     ''' if input_option is set to 'param' then model_weights should be models.parameters() or list of parameters
    #         if input_option is set to 'state' then model_weights should be state dictionary of a model
    #     '''
    #     acc, test_loss = evaluation(
    #         test_model=self.model_test, 
    #         learned_param=model_weights, 
    #         test_loader=self.testloader, 
    #         device=self.device, 
    #         option=input_option
    #     )
    #     return acc, test_loss

    # def evaluation(self, eval_loader, model_weights, method='copy'):
    #     if method == 'copy':
    #         copy_parameters(self.model_test.parameters(), model_weights)
    #     elif method =='load':
    #         self.model.load_state_dict(model_weights)

    #     self.model_test.eval()
    #     criterion = nn.CrossEntropyLoss(reduction='sum').to(self.device)
    #     correct, loss = 0, 0.0
    #     n_test_samples = len(eval_loader.dataset)        
        
    #     with torch.no_grad():
    #         for x, y in eval_loader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             logits = self.model_test(x)
    #             pred = F.log_softmax(logits, dim=1)
    #             correct += (pred.argmax(dim=1) == y).sum().item()
    #             loss += criterion(pred,y).item()
    #     acc, loss = correct / n_test_samples, loss / n_test_samples    
    #     return acc, loss
    

    # '''the following are copied from clientcpfl as a reference'''
    # def cache_model(self, source_model_param):
    #     """ to be used after sync_with_cluster or sync_with_server
    #         save the model before update into a local cache
    #     """
    #     copy_parameters(target=self.model_cache, source=source_model_param)

    # def local_training(self, local_epochs, learning_rate, weight_decay=0):
    #     self.model.train()
    #     optimizer = torch.optim.SGD(self.model.parameters(), learning_rate, momentum=0.9, weight_decay=weight_decay)
    #     for i in range(local_epochs):
    #         train_loss, n_samples = 0.0, 0
    #         for x, y in self.trainloader: 
    #             x, y = x.to(self.device), y.to(self.device)
    #             optimizer.zero_grad()
    #             output = self.model(x)
    #             loss = self.loss(output,y)
    #             loss.backward()
    #             optimizer.step()

    #             train_loss += loss.item() * y.size(0)
    #             n_samples += y.size(0)  
        
    #     # save the newly updated local model into a local cache, as the intermediate local model
    #     copy_parameters(target=self.intermediate_local_model_param, source=self.model.parameters())
    #     self.intermediate_local_W = copy.deepcopy(self.model.state_dict())
    #     # return train_loss / n_samples
    
    # def compute_update_direction(self, model_before_update, model_after_update):
    #     """ used for computing the update direction by substracting the model after and before the local update step
    #     """
    #     target = self.local_update_direction
    #     for param_target, param_after, param_before in zip(target, model_after_update, model_before_update):
    #         param_target.data = param_after.data.clone() - param_before.data.clone()
    #     self.local_update_direction = [param.detach().cpu() for param in self.local_update_direction]

        


