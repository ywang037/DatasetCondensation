import os
import time
import copy
import argparse
from unittest import TestLoader
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
from utils import data_preparation, gen_data_partition_iid, gen_data_partition_dirichlet, make_client_dataset_from_partition

import random
from client import ClientDC
from server import ServerDC

def argparser():
    parser = argparse.ArgumentParser(description='Federated DC using gradient matching')

    # default args
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    # addon args
    parser.add_argument('--num_clients', type=int, default=5, help='number of clients')
    parser.add_argument('--seed', type=int, default=3, help='set a seed for reproducability, set to 0 to activate randomness')
    parser.add_argument('--client_alpha', type=float, default=100.0, help='dirichlet alpha for intra-cluster non-iid degree')

    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args

def main(args):
    # if not os.path.exists(args.data_path):
    #     os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    '''do some global settings'''
    # The list of iterations when we evaluate models and record results.
    # the defualt setting is to evaluate every 500 iterations, i.e., k=n*500
    # eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] 
    eval_it_pool = [args.Iteration] # only evaluate at last iteration
    print('eval_it_pool: ', eval_it_pool)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    data_set, data_info, testloader_server = data_preparation(args.dataset)
    
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    



    # NOTE for WY: the author run multiple trial of experiments using the for loop over exp and args.num_exp is the number of trials
    # if you want to control the reproducibility, then freeze the randomness outside the following loop 
    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        '''split data set for each client'''
        # generate training data partitioning
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
        client_train_data_idcs, client_train_class_dict = gen_data_partition_iid(
            data=data_set['train_data'], 
            num_classes=data_info['num_classes'], 
            labels=data_set['train_labels'], 
            data_mapp=data_set['mapp'],
            num_clients=args.num_clients, 
            generator=rng, 
            verbose_hist=False)

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
        client_test_data_idcs, client_test_class_dict = gen_data_partition_iid(
            data=data_set['test_data'], 
            num_classes=data_info['num_classes'], 
            labels=data_set['test_labels'], 
            data_mapp=data_set['mapp'],
            num_clients=args.num_clients, 
            generator=rng, 
            verbose_hist=False)

        # make client data using the generated partition
        client_data_train = make_client_dataset_from_partition(data_set['train_data'], args.num_clients, client_test_data_idcs, data_set['transform'])
        client_data_test = make_client_dataset_from_partition(data_set['test_data'], args.num_clients, client_test_data_idcs, data_set['transform'])

        '''set the architecture for the model to be trained'''
        net_train = get_network(args.model, data_info['channel'], data_info['num_classes'], data_info['img_size']).to(args.device)

        '''create clients and server'''
        clients = [ClientDC(id, args, net_train, data_info, client_data_train[i], client_data_test[i], eval_it_pool, model_eval_pool) for i in range(args.num_clients)]
        for client in clients:
            print('Client {} has {} training samples {} testing samples'.format(client.id, client.n_data_train, client.n_data_test))
        server = ServerDC(args, net_train, clients)
        print('FL server created.')

        ''' organize the real dataset '''
        # to be repeated for every client
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

        # NOTE for WY: the following lines need to be executed for every client in Federated version, see codes between marks <STARTS FROM ... ENDS HERE>
        # STARTS FROM HERE
        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins'%get_time())
        # ENDS HERE

        # NOTE this iteration is over the different model initializations, 
        # i.e., the loop indixed by K in the paper, Algorithm 1 line 4
        for it in range(args.Iteration+1): 

            # NOTE for WY: the following loop needs to be executed for every client in Federated version
            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                    print('DC augmentation parameters: \n', args.dc_aug_param)
                    args.epoch_eval_train = 300

                    accs = []
                    for it_eval in range(args.num_eval):
                        # get a random model
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) 
                        
                        # avoid any unaware modification
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) 
                        
                        # trains new models using condensed/synthetic data then evaluate the accuracy of this resulting model
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

            # NOTE the following will be modified to federated rounds
            # NOTE for WY: the following lines need to be executed for every client in Federated version, see codes between marks <start from ... ends here>
            # start from here
            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model, better to rename net to net_train
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.

            
            # NOTE this outer_loop is not the loop over different model initialization
            # this loop is indixed by T in the paper, Algorithm 1 line 4
            for ol in range(args.outer_loop): 

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer


                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(args.device)
                
                # NOTE this loop is over labels, i.e., line 5-8 in Algorithm 1
                # NOTE that, the following synthetic data learning process is looped over only one batch of real data, i.e., one step of SGD for learning data, 
                # in actual one can use multiple steps by looping the following sections, but one may need to use non-replace method to randomly draw real data, just like a data loader
                # update only once but over K iterations equivalent to K steps of SGD for learning the data
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss += match_loss(gw_syn, gw_real, args)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                if ol == args.outer_loop - 1:
                    break
                
                # avoid any unaware modification
                # the following line copy the synthetic data from the running/working variables to another variable for saving purpose
                # so that variables image_syn_train, label_syn_train needs to be made distributed for each client to avoid overlapping
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  
                

                ''' update network '''
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                
                # NOTE the following loop is iterated for local model updates by SGD, 
                # i.e., line 9 in Algorithm 1 the args.inner_loop controls number of SGD steps, 
                # the number of args.inner_loop will be the number of local updates, 
                # note that because number of sythetic data is small, however the batch size is large by default, e.g., 256
                # so that one epoch may only have one SGD update using all the available synthetic data
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug = False)
                
            # ends here
                
            # NOTE FOR WY: you may wish to modify the above loop to make it a federated version
            # E.g., before the above loop, the net.parameters can be replaced by a aggregated version
            # the aggregation can be calculated here after the above "outer loop", i.e., the loop over t
            # <Place holder for model parameter aggregation>

            loss_avg /= (num_classes*args.outer_loop)

            if it%10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()


