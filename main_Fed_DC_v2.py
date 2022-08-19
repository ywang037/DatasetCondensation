import os
import time
import copy
import argparse
from unittest import TestLoader
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import data_preparation, gen_data_partition_iid, get_network, get_eval_pool, get_time, make_client_dataset_from_partition, ParamDiffAug 
# from utils import get_loops, get_dataset, gen_data_partition_dirichlet, evaluate_synset, get_daparam, match_loss, TensorDataset, epoch, DiffAugment

import random
from client import ClientDC
from server import ServerDC

def argparser():
    parser = argparse.ArgumentParser(description='Federated DC using gradient matching')

    # default args - data set and model
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    # default args - condensation
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')  
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric for gradient matching')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')

    # default args - evluation
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_eval', type=int, default=20, help='the number randomly initialized models for evaluating the syn data')
    parser.add_argument('--epoch_eval_train', type=int, default=50, help='epochs to train a model with synthetic data for evaluating the syn data')

    # default args - experiments
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiment runs')  
    parser.add_argument('--Iteration', type=int, default=1000, help='the number of model intializations to be used, i.e., big K in the paper gradient matching')

    # args - fed/overall
    parser.add_argument('--seed', type=int, default=3, help='set a seed for reproducability, set to 0 to activate randomness')
    parser.add_argument('--num_clients', type=int, default=5, help='number of clients')
    parser.add_argument('--client_alpha', type=float, default=100.0, help='dirichlet alpha that controls the non-iid degree')
    
    # args - fed/server
    parser.add_argument('--stand_alone', action='store_true', default=False, help='trigger non-federated local training mode')
    parser.add_argument('--server_mode', type=str, default='train', help='operation model of server train or agg')
    parser.add_argument('--server_lr', type=float, default=0.01, help='learning rate for updating global model by the server')
    parser.add_argument('--server_batch_train', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--server_epoch_train', type=int, default=10, help='epochs to train the global model with synthetic data')
    
    # args - fed/clients
    parser.add_argument('--rounds', type=int, default=10, help='epochs to train the global model with synthetic data')
    parser.add_argument('--client_epoch_train', type=int, default=10, help='epochs to train the local model with synthetic data')
    parser.add_argument('--client_batch_train_data', type=int, default=256, help='batch size for real data')
    parser.add_argument('--client_batch_train_model', type=int, default=128, help='batch size for training networks')

    # args - results
    parser.add_argument('--save_results', action='store_true', default=False, help='use this to save trained synthetic data and images')
    parser.add_argument('--save_root', type=str, default='result', help='path to save results')
    args = parser.parse_args()

    if args.stand_alone:
        save_tag = args.dataset + '_local' + time.strftime('_%y-%m-%d-%H-%M-%S') 
    elif args.server_mode == 'agg': # use model aggregation mode
        save_tag = args.dataset + '_fed_agg' + time.strftime('_%y-%m-%d-%H-%M-%S') 
    elif args.server_mode == 'train': # use model training mode
        save_tag = args.dataset + '_fed_train' + time.strftime('_%y-%m-%d-%H-%M-%S') 
    args.save_path = os.path.join(args.save_root, save_tag) 

    # the following are original arguments better to be kept unchanged
    # args.outer_loop, args.inner_loop = 10, 10
    # args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
    return args

def main(args):
    
    # set the seeds
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)    
    
    # some global setup
    print('Hyper-parameters: \n', args.__dict__)
    
    # if not os.path.exists(args.data_path):
    #     os.mkdir(args.data_path)
    if not os.path.exists(args.save_path) and args.save_results:
        os.mkdir(args.save_path)

    # The list of iterations when we evaluate models and record results.
    # the defualt setting is to evaluate every 500 iterations, i.e., k=n*500
    # eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] 
    eval_it_pool = [args.Iteration] # only evaluate at last iteration
    print('eval_it_pool: ', eval_it_pool)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    print('Evaluation model pool: ', model_eval_pool)
    data_set, data_info, server_testloader = data_preparation(args.dataset)

    accs_all_clients_all_exps = [dict() for i in range(args.num_clients)]
    for i in range(args.num_clients):
        for key in model_eval_pool:
            accs_all_clients_all_exps[i][key]=[]
    data_save_all_clients = [[] for i in range(args.num_clients)]

    # looping over multiple experiment trials
    for exp in range(args.num_exp):
        print('\n{} ================== Exp {} ==================\n'.format(get_time(), exp))       

        # split data set for each client
        # generate training data partitioning using dirichlet distribution
        # client_label_dist, client_train_data_idcs, client_train_class_dict = gen_data_partition_dirichlet(
        #     data=data_set['train_data'], 
        #     num_classes=data_info['num_classes'], 
        #     labels=data_set['train_labels'], 
        #     data_mapp=data_set['mapp'],
        #     num_clients=args.num_clients, 
        #     generator=rng, 
        #     client_alpha=args.client_alpha, 
        #     client_label_dist=None, 
        #     verbose_hist=False
        #     )

        # generate testing data partitioning
        # _, client_test_data_idcs, client_test_class_dict = gen_data_partition_dirichlet(
        #     data=data_set['test_data'], 
        #     num_classes=data_info['num_classes'], 
        #     labels=data_set['test_labels'], 
        #     data_mapp=data_set['mapp'],
        #     num_clients=args.num_clients, 
        #     generator=rng, 
        #     client_alpha=args.client_alpha, 
        #     client_label_dist=client_label_dist, 
        #     verbose_hist=False
        #     )
        
        # generate training data partitioning using IID method
        client_train_data_idcs, client_train_class_dict = gen_data_partition_iid(
            data=data_set['train_data'], 
            num_classes=data_info['num_classes'], 
            labels=data_set['train_labels'], 
            data_mapp=data_set['mapp'],
            num_clients=args.num_clients, 
            generator=rng, 
            verbose_hist=False)

        client_test_data_idcs, client_test_class_dict = gen_data_partition_iid(
            data=data_set['test_data'], 
            num_classes=data_info['num_classes'], 
            labels=data_set['test_labels'], 
            data_mapp=data_set['mapp'],
            num_clients=args.num_clients, 
            generator=rng, 
            verbose_hist=False)

        # make client data using the generated partition
        client_data_train = make_client_dataset_from_partition(data_set['train_data'], args.num_clients, client_train_data_idcs)
        client_data_test = make_client_dataset_from_partition(data_set['test_data'], args.num_clients, client_test_data_idcs)
        print('{} Data partitioning done.'.format(get_time()))

        # set the architecture for the network to be trained
        net_train = get_network(args.model, data_info['channel'], data_info['num_classes'], data_info['img_size']).to(args.device)

        # create clients and server
        clients = [ClientDC(id, args, net_train, data_info, client_data_train[i], client_data_test[i], eval_it_pool, model_eval_pool) for id in range(args.num_clients)]
        for client in clients:
            if not os.path.exists(client.save_path) and args.save_results:
                os.mkdir(client.save_path)
        server = ServerDC(args, net_train, clients, data_info, server_testloader)
        print('{} FL server created.'.format(get_time()))

        # organize the real dataset and initialize the synthetic data
        for client in clients:
            print('{} Client {} has {} training samples {} testing samples'.format(get_time(), client.id, client.num_local_data_train, client.num_local_data_test))
            client.organize_local_real_data()
            print('{} Client {} real data organization done.'.format(get_time(),client.id))
            # for ch in range(client.channel):
            #     print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(client.images_all[:, ch]), torch.std(client.images_all[:, ch])))
            client.syn_data_init()
            print('{} Client {} synthetic data initialized.'.format(get_time(),client.id))
            client.data_trainer_setup() # only to be called after syn_data_init()
            print('{} Client {} synthetic data optimizer set.\n-------------------------'.format(get_time(),client.id))
        

        # training starts from here '''
        print('%s Training session started'%get_time())
        
        # NOTE this loop is over the different model initializations, i.e., the loop indixed by K in the paper, Algorithm 1 line 4
        for it in range(args.Iteration+1): 
            # get a new random initialization of the network
            server.global_model = get_network(args.model, data_info['channel'], data_info['num_classes'], data_info['img_size']).to(args.device) 
            server.global_model_state = server.global_model.state_dict()
            # print('-------------------------\n{} {}-th model initialization sampled.'.format(get_time(), it))
            
            for client in clients:
                # ''' get a new random initialization of the network '''
                # client.model_train = get_network(args.model, client.channel, client.num_classes, client.im_size).to(args.device) 

                # fetch newly intialized server model weights '''
                client.sync_with_server(server, method='state')
                # print('{} Client {} synced initial model with server.'.format(get_time(), client.id))

                # set the optimizer for learning synthetic data '''
                optimizer_net = client.net_trainer_setup(client.model_train)
                # print('{} Client {} model optimizer set.'.format(get_time(), client.id))


            # NOTE this loop is indixed by T in the paper, Algorithm 1 line 4
            # this loop resembles the communication round in FL
            for ol in range(args.rounds): 
                # print('-------------------------\n{} {}-th round started.'.format(get_time(),ol))

                # clients perform local update of data and network '''
                for client in clients:
                    if not args.stand_alone and ol:
                        client.sync_with_server(server, method='state') # fetch server model weights
                    # print('{} Client {} synced with server.'.format(get_time(), client.id))

                    # freeze the running mu and sigma for BatchNorm layers '''
                    # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                    # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                    # This would make the training with BatchNorm layers easier.
                    BN_flag = False
                    BNSizePC = 16  # for batch normalization
                    for module in client.model_train.modules():
                        if 'BatchNorm' in module._get_name(): #BatchNorm
                            BN_flag = True
                    if BN_flag:
                        img_real = torch.cat([client.get_images(c, BNSizePC) for c in range(client.num_classes)], dim=0)
                        client.model_train.train() # for updating the mu, sigma of BatchNorm
                        output_real = client.model_train(img_real) # get running mu, sigma
                        for module in client.model_train.modules():
                            if 'BatchNorm' in module._get_name():  #BatchNorm
                                module.eval() # fix mu and sigma of every BatchNorm layer

                    # local update of synthetic data '''
                    # one step of SGD, can be repeated for multiple steps
                    # update only once but over T iterations equivalent to T steps of SGD for learning the data
                    client.syn_data_update(client.model_train)
                    # print('{} Client {} syntheic data updated.'.format(get_time(), client.id)) 

                    # local update of network (using synthetic data) '''
                    client.network_update(client.model_train, optimizer_net) 
                    client.local_model_state = copy.deepcopy(client.model_train.state_dict()) # copy the updated local model weights to another iterables to avoid any unaware modification
                    # print('{} Client {} model updated.'.format(get_time(), client.id))    

                # server side operation - update global model
                if not args.stand_alone:
                    if args.server_mode == 'train': # Server perform aggregation-free global model update by training on client-uploaded synthetic data
                        server.update_server_syn_data(clients, server_train_batch_size=args.server_batch_train) # server first update its synthetic data set by receiving synthetic data from every clients
                        server.train_global_model(server_lr=args.server_lr, server_train_epoch=args.server_epoch_train) # server then update the global model by training on its server synthetic data set
                        # print('{} Server global model updated.'.format(get_time()))
                    else: # Server perform model aggregation for synthetic updated model uploaded by clients
                        server.model_aggregation(clients) 
                        # print('{} Clients models aggregated.'.format(get_time()))

            # monitor training loss of synthetic data
            for client in clients:
                client.loss_avg /= (client.num_classes*args.rounds) 
                if it%10 == 0:               
                    print('%s Client %d iter = %04d, syn data train loss = %.4f' % (get_time(), client.id, it, client.loss_avg))

            # Evaluate synthetic data trained in last iteration, i.e., summary for client data condensation for this exp trial
            # print('{} Synthetic data evaluation started.'.format(get_time()))
            for client in clients:
                client.syn_data_eval(exp, it, accs_all_clients_all_exps)          
                if it == args.Iteration and args.save_results: # only record the final results
                    data_save_all_clients[client.id].append([copy.deepcopy(client.image_syn.detach().cpu()), copy.deepcopy(client.label_syn.detach().cpu())])
                    torch.save({'data': data_save_all_clients[client.id], 'accs_all_exps': accs_all_clients_all_exps[client.id], }, os.path.join(client.save_path, 'res_%s_%s_%s_%dipc.pt'%(client.args.method, client.args.dataset, client.args.model, client.args.ipc)))

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        acc_overall = []
        for i in range(args.num_clients):
            accs = accs_all_clients_all_exps[i][key]
            print('Client %d run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(i, args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))
            acc_overall += accs
        print('\n-------------------------\nAverage performance after {:d} experiments, evaluted on {:2d} random {}, mean = {:.2f}%, std = {:.2f}%'.format(args.num_exp, len(acc_overall), args.model, np.mean(acc_overall)*100, np.std(accs)*100))


if __name__ == '__main__':
    time_start = time.time()
    
    main(args=argparser())
    
    time_end = time.time()
    sesseion_time = np.around((time_end-time_start)/3600, 2)
    print('Session time: {} hrs. That\'s all folks.'.format(sesseion_time))
    time_end_stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    # time_end_stamp = time.strftime('%y-%m-%d-%H-%M-%S')
    print(f'Session completed at {time_end_stamp}')

