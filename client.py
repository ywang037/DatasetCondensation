# import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
from utils.func_utils import evaluation

from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
from utils import copy_parameters
          
class ClientDC(object):
    def __init__(self, id, args, net_train, data_info, data_train, data_test, eval_it_pool, model_eval_pool):
        
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
        self.data_save = self.data_save_init() 

        self.model_train = copy.deepcopy(net_train).to(self.device)
        # self.model_test = self.init_model_test(model)
        # the followings are lists storing model parameters, not the trainable model
        self.model_cache = copy.deepcopy(list(self.model_train.parameters()))
        self.local_model_weights = copy.deepcopy(list(self.model_train.parameters()))
        self.local_model_state = copy.deepcopy(self.model.state_dict())

        self.local_data_train = data_train
        self.local_data_test = data_test
        # self.n_data_train = len(self.local_data_train)
        # self.n_data_test = len(data_test)
        self.batch_size_learn_data = args.batch_real
        self.batch_size_learn_model = args.batch_train
        # self.trainloader = DataLoader(self.local_data_train, batch_size=self.batch_size, shuffle=True)
        self.local_testloader = DataLoader(self.local_data_test, batch_size=256, shuffle=False)        
        
        self.images_all = []
        self.labels_all = []
        self.indices_class = [[] for c in range(self.num_classes)]
        self.image_syn = None
        self.label_syn = None
                
        self.criterion = nn.CrossEntropyLoss().to(args.device)
        


    '''data initilization'''
    # The list of iterations when we evaluate models and record results.
    # the defualt setting is to evaluate every 500 iterations, i.e., k=n*500
    def data_save_init(self):
        # record performances of all experiments
        accs_all_exps = dict() 
        for key in self.model_eval_pool:
            accs_all_exps[key] = []
        data_save = []
        return data_save

    ''' organize the real dataset '''
    def organize_local_real_data(self):
        # images_all = []
        # labels_all = []
        # indices_class = [[] for c in range(self.num_classes)]

        self.images_all = [torch.unsqueeze(self.local_data_train[i][0], dim=0) for i in range(len(self.local_data_train))]
        self.labels_all = [self.local_data_train[i][1] for i in range(len(self.local_data_train))]
        for i, lab in enumerate(self.labels_all):
            self.indices_class[lab].append(i)
        self.images_all = torch.cat(self.images_all, dim=0).to(self.device)
        self.labels_all = torch.tensor(self.labels_all, dtype=torch.long, device=self.device)

        for c in range(self.num_classes):
            print('class c = %d: %d real images'%(c, len(self.indices_class[c])))

    def get_images(self, c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
        return self.images_all[idx_shuffle]

    
        # for ch in range(self.channel):
        #     print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

        # NOTE for WY: the following lines need to be executed for every client in Federated version, see codes between marks <STARTS FROM ... ENDS HERE>
        # STARTS FROM HERE

    ''' initialize the synthetic data '''
    def syn_data_init(self):
        self.image_syn = torch.randn(size=(self.num_classes*self.ipc, self.channel, self.im_size[0], self.im_size[1]), dtype=torch.float, requires_grad=True, device=self.device)
        self.label_syn = torch.tensor([np.ones(self.ipc)*i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False, device=self.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if self.args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(self.num_classes):
                self.image_syn.data[c*self.ipc:(c+1)*self.ipc] = self.get_images(c, self.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')
        return

        # ''' training '''
        # optimizer_img = torch.optim.SGD([self.image_syn, ], lr=self.args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        # optimizer_img.zero_grad()
        # criterion = nn.CrossEntropyLoss().to(self.device)
        # print('%s training begins'%get_time())


    '''below are client functions for federated learning'''
    def sync_with_server(self, server, method='weights'):
        """ receive/copy global model of last round to a local cache
            to be used for local training steps
        """
        
        if method == 'weights': # download the weight parameters of the global model
            copy_parameters(target=self.model_train.parameters(), source=server.global_model_weights)    
        elif method == 'state': # download the state of the global model
            self.model_train.load_state_dict(server.global_model_state)


    '''the following are copied from clientbase'''
    def init_model_test(self, model):
        m = copy.deepcopy(model).to(self.device)
        for p in m.parameters():
            p.detach()
        return m

    def init_data_val(self, data_val):
        if data_val is not None:
            local_data_val = data_val
            n_data_val = len(data_val)
            valloader = DataLoader(data_val, batch_size=self.batch_size, shuffle=False)
        else:
            local_data_val = None
            n_data_val = 0
            valloader = None
        return local_data_val, n_data_val, valloader

    def evaluation2(self, model_weights, input_option='param'):
        ''' if input_option is set to 'param' then model_weights should be models.parameters() or list of parameters
            if input_option is set to 'state' then model_weights should be state dictionary of a model
        '''
        acc, test_loss = evaluation(
            test_model=self.model_test, 
            learned_param=model_weights, 
            test_loader=self.testloader, 
            device=self.device, 
            option=input_option
        )
        return acc, test_loss

    def evaluation(self, eval_loader, model_weights, method='copy'):
        if method == 'copy':
            copy_parameters(self.model_test.parameters(), model_weights)
        elif method =='load':
            self.model.load_state_dict(model_weights)

        self.model_test.eval()
        criterion = nn.CrossEntropyLoss(reduction='sum').to(self.device)
        correct, loss = 0, 0.0
        n_test_samples = len(eval_loader.dataset)        
        
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model_test(x)
                pred = F.log_softmax(logits, dim=1)
                correct += (pred.argmax(dim=1) == y).sum().item()
                loss += criterion(pred,y).item()
        acc, loss = correct / n_test_samples, loss / n_test_samples    
        return acc, loss
    

    '''the following are copied from clientcpfl as a reference'''
    def cache_model(self, source_model_param):
        """ to be used after sync_with_cluster or sync_with_server
            save the model before update into a local cache
        """
        copy_parameters(target=self.model_cache, source=source_model_param)

    def local_training(self, local_epochs, learning_rate, weight_decay=0):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), learning_rate, momentum=0.9, weight_decay=weight_decay)
        for i in range(local_epochs):
            train_loss, n_samples = 0.0, 0
            for x, y in self.trainloader: 
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output,y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * y.size(0)
                n_samples += y.size(0)  
        
        # save the newly updated local model into a local cache, as the intermediate local model
        copy_parameters(target=self.intermediate_local_model_param, source=self.model.parameters())
        self.intermediate_local_W = copy.deepcopy(self.model.state_dict())
        # return train_loss / n_samples
    
    def compute_update_direction(self, model_before_update, model_after_update):
        """ used for computing the update direction by substracting the model after and before the local update step
        """
        target = self.local_update_direction
        for param_target, param_after, param_before in zip(target, model_after_update, model_before_update):
            param_target.data = param_after.data.clone() - param_before.data.clone()
        self.local_update_direction = [param.detach().cpu() for param in self.local_update_direction]

        
    def get_parameters(self):
        """this one used for model aggregation, copy the weight of updated local model but remove grads
        """
        for param in self.intermediate_local_model_param:
            param.detach()
        return self.intermediate_local_model_param

    def get_teacher(self, clusters, server, option='per', direct_output=False, gamma=1e-2):
        # compute the teacher model
        # if option == 'per':    
        #     # weighting by the distance in parameter space between models, the nearer the distance the larger the weights 
        #     s = flatten(self.intermediate_local_model_param) # a torch tensor
            
        #     # using Eucliedan distance
        #     combination_weights = [Euclidean_dist2(s,flatten(cluster.cluster_model_param)).item() for cluster in clusters]
        #     combination_weights = np.array(combination_weights, dtype=np.float64)
        #     combination_weights = np.exp(-gamma*combination_weights)
            
        #     # # using cosine similarity
        #     # combination_weights = [cos_sim(s,flatten(cluster.cluster_model_param),eps=0) for cluster in clusters]
        #     # combination_weights = np.array(combination_weights, dtype=np.float64)
        #     # combination_weights = np.exp(100.0*combination_weights)
        # elif option == 'central':
        #     # W = torch.tensor([1.0]*server.n_cluster)
        #     combination_weights = [1.0] * server.n_cluster
        #     combination_weights = np.array(combination_weights, dtype=np.float64)
        # elif option == 'local':
        #     combination_weights = [0] * server.n_cluster
        #     combination_weights[self.cluster_id]=1
        #     combination_weights = np.array(combination_weights, dtype=np.float64)
        combination_weights = combination_weights/combination_weights.sum()

        # initialize the parameters of personlized teacher to be zero tensors
        for param in self.teacher_model_param:
            param.data = torch.zeros_like(param.data, requires_grad=False)

        # perform a model aggregation-like weighted average for compute personalized teacher
        # cluster_models = [cluster.cluster_model_param for cluster in clusters if cluster.id != self.cluster_id]
        cluster_models = [cluster.cluster_model_param for cluster in clusters]
        for cluster_id, cluster_model in enumerate(cluster_models):
            for param_teacher, param_cluster_model in zip(self.teacher_model_param, cluster_model):
                param_teacher.data += combination_weights[cluster_id] * param_cluster_model.data.clone()

        # # perform a model aggregation-like weighted average for compute personalized teacher
        # for cluster_id, cluster_model in enumerate(server.cluster_models_param):
        #     for param_teacher, param_cluster_model in zip(self.teacher_model_param, cluster_model):
        #         param_teacher.data += combination_weights[cluster_id] * param_cluster_model.data.clone()
                # param_teacher.data += W[cluster_id].item() * param_cluster_model.data.clone()
        # for cluster_id in range(server.n_cluster):
        #     for param_teacher, param_cluster_model in zip(self.teacher_model_param, server.cluster_models[cluster_id]):
        #         param_teacher.data += W[cluster_id].item() * param_cluster_model.data.clone()
        
        if direct_output:
            copy_parameters(target=self.personalized_local_model_param,source=self.teacher_model_param)

        return combination_weights


    def knowledge_transfer(
        self, 
        input_teacher_param, 
        output_student_param,
        distill_loader, 
        distill_temperature=1, 
        KD_epoch=1, 
        weight_CE = 1.0, 
        weight_KD = 1.0, 
        lr_KD=1e-2,
        weight_decay=0
    ):
        # # compute persoanlized teacher network
        # self.personalized_teacher(server)
        
        # # debug:
        # print('client {} is tranfering knowledge from {}-th cluster model'.format(self.id, server.cluster_labels[self.id]))

        # prepare teacher model: load the computed teach model parameters
        teacher_model = copy.deepcopy(self.model)
        copy_parameters(target=teacher_model.parameters(), source=input_teacher_param)
        for param in teacher_model.parameters():
            param.requires_grad = False

        # perpare student model: load the intermediate local model for KD training
        copy_parameters(target=self.model.parameters(), source=self.intermediate_local_model_param)
        
        teacher_model.eval()
        self.model.train()
        optimizer_KD = torch.optim.SGD(self.model.parameters(), lr=lr_KD, momentum=0.9, weight_decay=weight_decay)

        for k in range(KD_epoch): 
            # TO-DO: include a proper distillation set intialization
            for distill_input, distill_target in distill_loader:
                input, target = distill_input.to(self.device), distill_target.to(self.device)
                
                with torch.no_grad():
                    teacher_logits = teacher_model(input)
                soft_teacher_prediction = F.softmax(teacher_logits / distill_temperature, dim=1)   
                
                student_logits = self.model(input)
                soft_student_prediction = F.log_softmax(student_logits / distill_temperature, dim=1)
                
                CE_loss = self.loss(student_logits, target)              
                KD_loss = self.loss_kld(soft_student_prediction, soft_teacher_prediction)
                adaptation_loss = weight_CE * CE_loss + weight_KD * KD_loss
                # adaptation_loss = (1-weight_KD)*weight_CE * CE_loss + weight_KD * KD_loss
                
                # debug:
                # print('soft prob: {}'.format(soft_student_prediction))
                # print('soft target: {}'.format(soft_teacher_prediction))
                # print('CE loss: {}, KD loss: {}'.format(CE_loss, KD_loss))

                optimizer_KD.zero_grad()
                adaptation_loss.backward()
                optimizer_KD.step()

                # self.optimizer.zero_grad()
                # adaptation_loss.backward()
                # self.optimizer.step()

        # save the newly updated personalized local model into a local cache, as the intermediate local model
        copy_parameters(target=output_student_param, source=self.model.parameters())
        self.per_W = copy.deepcopy(self.model.state_dict())
        # return