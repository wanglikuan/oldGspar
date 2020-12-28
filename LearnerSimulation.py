# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import numpy as np


import torch
# import torch.distributed.deprecated as dist
from cjltest.divide_data import partition_dataset, select_dataset
from cjltest.models import MnistCNN, AlexNetForCIFAR, LeNetForMNIST
from cjltest.utils_data import get_data_transform
from cjltest.utils_model import MySGD
from torch.autograd import Variable
from torch.multiprocessing import Process as TorchProcess
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import ResNetOnCifar10
import vgg

parser = argparse.ArgumentParser()
# 集群信息
parser.add_argument('--ps-ip', type=str, default='127.0.0.1')
parser.add_argument('--ps-port', type=str, default='29500')
parser.add_argument('--this-rank', type=int, default=1)
parser.add_argument('--workers', type=int, default=2)

# 模型与数据集
parser.add_argument('--data-dir', type=str, default='~/dataset')
parser.add_argument('--data-name', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='LROnMnist')
parser.add_argument('--save-path', type=str, default='./')

# 参数信息
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--train-bsz', type=int, default=100)
parser.add_argument('--stale-threshold', type=int, default=0)
parser.add_argument('--ratio', type=float, default=0.01)
parser.add_argument('--isCompensate', type=bool, default=False)
parser.add_argument('--loops', type=int, default=1)
parser.add_argument('--title', type=str, default='GSpar')
parser.add_argument('--method', type=str, default='Mean')

parser.add_argument('--byzantine', type=int, default=0)
parser.add_argument('--V', type=float, default=0)

args = parser.parse_args()

def greedy(g_new):
    kappa, j, c = args.ratio, 0, 2
    param_num = 0.0
    prob = []
    total = 0.0
    # print("g_new:", g_new)
    for idx, g_layer in enumerate(g_new):
        param_num += torch.numel(g_layer)
        prob.append(torch.abs(g_layer))
        total += torch.sum(prob[idx])
    # print("Initial:", prob)
    for idx, _ in enumerate(prob):
        prob[idx] = torch.clamp(kappa * param_num * prob[idx] / total, max=1)
    while(c > 1):
        # calculate the value of c 
        num_nonones, num_ones, tot = 0.0, 0.0, 0.0
        for idx, _ in enumerate(prob):
            num_nonones += (prob[idx] != 1).sum().float()
            num_ones += (prob[idx] == 1).sum().float()
            tot += torch.sum(prob[idx]).float()
        c = (kappa * param_num - param_num + num_nonones) / (tot - num_ones)
        # print(total, kappa, param_num, num_nonones, tot, num_ones, tot - num_ones, c)
        for idx, _ in enumerate(prob):
            prob[idx] = torch.clamp(c * prob[idx], max=1)
        j += 1
    return prob

def byzantine_func(g_change, dev):
    g_mem = []
    g_layer_vector = torch.empty(0).cuda(dev)
    for idx, g_layer in enumerate(g_change):
        g_mem.append(g_layer)
        g_layer_reshape = g_layer.reshape(torch.numel(g_layer))
        g_layer_vector = torch.cat((g_layer_vector, g_layer_reshape),dim=0)  # merge two vectors into one vector

    tot_num = len(torch.nonzero(g_layer_vector))
    tot_val = torch.sum(g_layer_vector)
    tot_val /= tot_num

    g_new = []
    for idx, g_layer in enumerate(g_mem):
        mask = g_layer >= 0
        g_new_layer = torch.zeros_like(g_layer).cuda(dev)
        g_new_layer[mask] = tot_val
        g_new.append(g_new_layer)
    return g_new

def get_upload(g_new):
    # print("Receiving:", g_new)
    upload_gradient = []
    upload_num = 0
    param_num = 0
    prob = greedy(g_new)
    for idx, g_layer in enumerate(g_new):
        z = torch.bernoulli(prob[idx])
        temp = z * g_layer / prob[idx]
        temp [temp != temp] = 0
        upload_gradient.append(temp)
        upload_num += (z == 1).sum().data
        param_num += torch.numel(g_layer)
    return upload_gradient, int(upload_num) / param_num


def test_model(rank, model, test_data, dev):
    correct = 0
    total = 0
    # model.eval()
    with torch.no_grad():
        for data, target in test_data:
            data, target = Variable(data).cuda(dev), Variable(target).cuda(dev)
            output = model(data)
            # get the index of the max log-probability
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            # pred = output.data.max(1)[1]
            # correct += pred.eq(target.data).sum().item()

    acc = format(correct / total, '.4%')
    # print('Rank {}: Test set: Accuracy: {}/{} ({})'
    #       .format(rank, correct, len(test_data.dataset), acc))
    return acc

# input: gradient list
# output: element-wise median of all gradients
def median_defense(g_list, workers, dev):
    median_g = []
    for p_idx, g_layer in enumerate(g_list[0]):
        g_layer_list = []
        for w in workers:
            g_layer_list.append(g_list[w - 1][p_idx])
        data_dim = g_layer_list[0].dim()
        # 取中位数
        tensor = torch.zeros_like(g_layer.data).cuda(dev) + torch.median(torch.stack(g_layer_list, data_dim), data_dim)[0]
        median_g.append(tensor)
    return median_g

# output: element-wise trimmed mean of all gradients
def trimmed_mean(g_list, workers, trimK, dev):
    workers_num = len(workers)
    g_trimmed_mean = []
    for p_idx, g_layer in enumerate(g_list[0]):
        g_trimmed_mean_layer = torch.zeros_like(g_layer.data).cuda(dev)
        g_layer_list = []
        for w in workers:
            g_layer_list.append(g_list[w - 1][p_idx])
        data_dim = g_layer_list[0].dim()
        tensor_max = torch.min(torch.topk(torch.stack(g_layer_list, data_dim), trimK)[0], -1)[0]
        tensor_min = -torch.min(torch.topk(-torch.stack(g_layer_list, data_dim), trimK)[0], -1)[0]

        for w in workers:
            max_mask = g_list[w - 1][p_idx].data >= tensor_max
            min_mask = g_list[w - 1][p_idx].data <= tensor_min

            tmp_layer = g_list[w - 1][p_idx].data + torch.zeros_like(g_list[w - 1][p_idx].data).cuda(dev)
            tmp_layer[max_mask] = 0
            tmp_layer[min_mask] = 0

            g_list[w - 1][p_idx] = tmp_layer

            g_trimmed_mean_layer.data += g_list[w-1][p_idx].data / (workers_num - 2 * trimK)
        g_trimmed_mean.append(g_trimmed_mean_layer)
    return g_trimmed_mean

# output: the mean of the applicable gradients
def FABA(g_list, workers, byzantine, dev):
    workers_num = len(workers)
    for k in range(byzantine):
        g_zero = []
        for p_idx, g_layer in enumerate(g_list[0]):
            global_update_layer = torch.zeros_like(g_layer.data).cuda(dev)
            for w in workers:
                global_update_layer += g_list[w-1][p_idx]
            g_zero.append(global_update_layer / (workers_num - k))
        max_differ, max_idx = 0, 0
        for w in workers:
            # g_differ = [0] * len(g_zero)
            total_differ = 0
            for p_idx, _ in enumerate(g_zero):
                # g_differ[p_idx] = g_list[w-1][p_idx] - g_zero[p_idx]
                total_differ += torch.norm(g_list[w - 1][p_idx] - g_zero[p_idx]).pow(2)
            # total_differ = torch.norm(g_differ)
            # print(max_differ)
            # print(total_differ)
            if max_differ < total_differ.data:
                max_differ, max_idx = total_differ.data, w
        for p_idx, g_layer in enumerate(g_list[max_idx - 1]):
            g_list[max_idx - 1][p_idx] = torch.zeros_like(g_layer.data).cuda(dev)
    g_zero = []
    print(max_idx)
    for p_idx, g_layer in enumerate(g_list[0]):
        global_update_layer = torch.zeros_like(g_layer.data).cuda(dev)
        for w in workers:
            global_update_layer += g_list[w - 1][p_idx]
        g_zero.append(global_update_layer / (workers_num - byzantine))
    return g_zero

# noinspection PyTypeChecker
def run(workers, models, save_path, train_data_list, test_data, iterations_epoch):
    dev = torch.device('cuda')
    cpu = torch.device('cpu')
    
    start_time = time.time()
    models[0] = models[0].cuda(dev)
    for i in workers:
        models[i] = models[i].cuda(dev)

    workers_num = len(workers)
    print('Model recved successfully!')
    optimizers_list = []
    for i in workers:
        optimizer = MySGD(models[i].parameters(), lr=args.lr)
        # if args.model in ['MnistCNN', 'AlexNet', 'ResNet18OnCifar10', 'VGG11']:
        #     optimizer = MySGD(models[i].parameters(), lr=0.1)
        # else:
        #     optimizer = MySGD(models[i].parameters(), lr=0.1)
        optimizers_list.append(optimizer)

    if args.model in ['MnistCNN', 'AlexNet']:
        criterion = torch.nn.NLLLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.model in ['AlexNet', 'ResNet18OnCifar10']:
        decay_period = 17500
    else:
        decay_period = 17500

    print('Begin!')

    g_old_num = args.loops
    # cache g_old_num old gradients
    g_old_list = [] # cache old gradient
    for i in workers:
        worker_g_old_list = [[torch.zeros_like(param.data).cuda(dev) for param in models[i].parameters()] for _ in range(g_old_num)]
        g_old_list.append(worker_g_old_list)
    g_old_count = 0

    global_g = [torch.zeros_like(param.data).cuda(dev) for param in models[0].parameters()]
    
    byzantine_workers_list = [w+1 for w in range(args.byzantine)]   # the several workers in the front of the rank list

    # store (train loss, energy, iterations)
    # naming rules: title + model_name + number_of_workers
    trainloss_file = './../result/' \
        + args.title \
        + '_' + args.method \
        + '_' + args.model \
        + '_lr' + str(args.lr) \
        + '_bsz' + str(args.train_bsz) \            
        + '_B' + str(args.byzantine) \
        + '_V' + str(int(args.V)) \
        + '_E' + str(args.loops) \
        + '_R' + str(int(args.ratio * 1000)) \        
        + '_W' + str(args.workers) + '.txt'
    
    if(os.path.isfile(trainloss_file)):
        os.remove(trainloss_file)
    f_trainloss = open(trainloss_file, 'a')

    train_data_iter_list = []
    for i in workers:
        train_data_iter_list.append(iter(train_data_list[i-1]))

    # epoch_train_loss = 0.0
    global_clock = 0
    g_remain_list = []
    ratio = args.ratio
    threshold = 0.
    for epoch in range(args.epochs):
        iteration_loss = 0.0

        g_change_average = [torch.zeros_like(param.data).cuda(dev) for param in models[0].parameters()]
        global_clock += 1
        g_list = [[] for _ in range(workers_num)]

        for i in workers:
            try:
                data, target = next(train_data_iter_list[i-1])
            except StopIteration:
                train_data_iter_list[i-1] = iter(train_data_list[i - 1])
                data, target = next(train_data_iter_list[i-1])
            data, target = Variable(data).cuda(dev), Variable(target).cuda(dev)
            optimizers_list[i-1].zero_grad()
            output = models[i](data)
            loss = criterion(output, target)
            loss.backward()
            delta_ws = optimizers_list[i-1].get_delta_w()
            iteration_loss += loss.data.item() / workers_num

            g_change, sparsification_ratio = get_upload(delta_ws)

            # if i in byzantine_workers_list:
                # print(i)
                # g_change = byzantine_func(g_change, dev)
                # for g_change_layer in g_change:
                #     g_change_layer.data += args.V * torch.randn_like(g_change_layer.data).cuda(dev)
            
            for g_change_layer in g_change:
                g_list[i - 1].append(g_change_layer)
            
            # for g_change_layer_idx, g_change_layer in enumerate(g_change_average):
            #     g_change_layer.data += g_change[g_change_layer_idx].data/workers_num
        
            # print(i, "Changing result:", g_change_average)

        # 同步操作
        g_quare_sum = 0.0   # for threshold
        if args.method == "Mean":
            for p_idx, param in enumerate(models[0].parameters()):
                global_update_layer = torch.zeros_like(param.data).cuda(dev)
                for w in workers:
                    global_update_layer += g_list[w-1][p_idx]
                tensor = global_update_layer / workers_num
                param.data -= tensor
                for w in workers:
                    list(models[w].parameters())[p_idx].data = param.data
        else:
            # aggregation
            if args.method == "TrimmedMean":
                g_median = trimmed_mean(g_list, workers, args.byzantine, dev)
            elif args.method == "Median":
                g_median = median_defense(g_list, workers, dev)
            elif args.method == "FABA":
                g_median = FABA(g_list, workers, args.byzantine, dev)

            # print(g_median, len(g_median))
            for p_idx, param in enumerate(models[0].parameters()):
                param.data -= g_median[p_idx].data
                for w in workers:
                    list(models[w].parameters())[p_idx].data = param.data + torch.zeros_like(param.data).cuda(dev)

        # epoch_train_loss += iteration_loss
        # epoch = int(iteration / iterations_epoch)
        current_time = time.time() - start_time
        test_acc = 0
        if epoch % 50 >= 45:
            test_acc = test_model(0, models[1], test_data, dev)
        print('Epoch {}, Time:{}, Loss:{}, Accuracy:{}'.format(epoch, current_time, iteration_loss, test_acc))
        f_trainloss.write(str(epoch) +
                            '\t' + str(current_time) +
                            '\t' + str(iteration_loss) + 
                            '\t' + str(sparsification_ratio) + 
                            # '\t' + str(test_loss) + 
                            '\t' + str(test_acc) +
                            '\n')
        f_trainloss.flush()
        # epoch_train_loss = 0.0
        # 在指定epochs (iterations) 减少缩放因子
        if (epoch + 1) in [0, 250000]:
            ratio = ratio * 0.1
            print('--------------------------------')
            print(ratio)

        # for i in workers:
        #     models[i].train()
        #     if (epoch + 1) % decay_period == 0:
        #         for param_group in optimizers_list[i - 1].param_groups:
        #             param_group['lr'] *= 0.1
        #             print('LR Decreased! Now: {}'.format(param_group['lr']))

    f_trainloss.close()


def init_processes(workers,
                   models, save_path,
                   train_dataset_list, test_dataset,iterations_epoch,
                   fn, backend='tcp'):
    fn(workers, models, save_path, train_dataset_list, test_dataset, iterations_epoch)


if __name__ == '__main__':

    torch.manual_seed(1)
    workers_num = args.workers
    workers = [v+1 for v in range(workers_num)]
    models = []

    for i in range(workers_num + 1):
        if args.model == 'MnistCNN':
            model = MnistCNN()

            train_transform, test_transform = get_data_transform('mnist')

            train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                           transform=train_transform)
            test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                          transform=test_transform)
        elif args.model == 'LeNet':
            model = LeNetForMNIST()

            train_transform, test_transform = get_data_transform('mnist')

            train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                           transform=train_transform)
            test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                          transform=test_transform)
        elif args.model == 'LROnMnist':
            model = ResNetOnCifar10.LROnMnist()
            train_transform, test_transform = get_data_transform('mnist')

            train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                           transform=train_transform)
            test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                          transform=test_transform)
        elif args.model == 'LROnCifar10':
            model = ResNetOnCifar10.LROnCifar10()
            train_transform, test_transform = get_data_transform('cifar')

            train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                           transform=train_transform)
            test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                          transform=test_transform)
        elif args.model == 'AlexNet':

            train_transform, test_transform = get_data_transform('cifar')

            if args.data_name == 'cifar10':
                model = AlexNetForCIFAR()
                train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                                 transform=train_transform)
                test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                                transform=test_transform)
            else:
                model = AlexNetForCIFAR(num_classes=100)
                train_dataset = datasets.CIFAR100(args.data_dir, train=True, download=False,
                                                  transform=train_transform)
                test_dataset = datasets.CIFAR100(args.data_dir, train=False, download=False,
                                                 transform=test_transform)
        elif args.model == 'ResNet18OnCifar10':
            model = ResNetOnCifar10.ResNet18()
            
            train_transform, test_transform = get_data_transform('cifar')
            train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                             transform=train_transform)
            test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                            transform=test_transform)
        elif args.model == 'ResNet34':
            model = models.resnet34(pretrained=False)
            
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            test_transform = train_transform
            train_dataset = datasets.ImageFolder(args.data_dir, train=True, download=False,
                                             transform=train_transform)
            test_dataset = datasets.ImageFolder(args.data_dir, train=False, download=False,
                                            transform=test_transform)
        elif args.model == 'VGG11':
            model = vgg.vgg11()

            train_transform, test_transform = get_data_transform('cifar')
            train_dataset = datasets.CIFAR100(args.data_dir, train=True, download=False,
                                             transform=train_transform)
            test_dataset = datasets.CIFAR100(args.data_dir, train=False, download=False,
                                            transform=test_transform)
        else:
            print('Model must be {} or {}!'.format('MnistCNN', 'AlexNet'))
            sys.exit(-1)
        models.append(model)
    train_bsz = args.train_bsz
    train_bsz /= len(workers)
    train_bsz = int(train_bsz)

    train_data = partition_dataset(train_dataset, workers)
    train_data_list = []
    for i in workers:
        train_data_sub = select_dataset(workers, i, train_data, batch_size=train_bsz)
        train_data_list.append(train_data_sub)

    test_bsz = 400
    # 用所有的测试数据测试
    test_data = DataLoader(test_dataset, batch_size=test_bsz, shuffle = False)

    iterations_epoch = int(len(train_dataset) / args.train_bsz)

    save_path = str(args.save_path)
    save_path = save_path.rstrip('/')

    p = TorchProcess(target=init_processes, args=(workers,
                                                  models, save_path,
                                                  train_data_list, test_data,iterations_epoch,
                                                  run))
    p.start()
    p.join()
