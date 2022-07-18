'''Train CIFAR10 with PyTorch.'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys
import time
import random

sys.path.append("../cifarformat")
sys.path.append(".")
from models import *

from cifarformat import in_mem_bismarck, in_mem_block, in_mem_block_only, in_mem_no_shuffle, in_mem_sliding_window, in_mem_once_fully_shuffle, in_mem_always_fully_shuffle



# Training
def train(epoch, net, trainloader, device, optimizer, criterion):
    #print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


       

def test(epoch, net, testloader, device, criterion):
    #global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

     

    # Save checkpoint.
    acc = 100.*correct/total

    return (acc, test_loss)



def main_worker(args):
    log_file = args['log_file']
    writer = open(log_file, 'w')

    for k in args:
        writer.write("[params] " + str(k) + " = " + str(args[k]) + '\n')
    
    writer.flush()

    writer.write('[%s] Start iteration' % get_current_time())
    writer.write('\n')


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    data_dir = args['data_dir']
    download = args['download']
    num_workers = args['num_workers']
    iter_num = args['iter_num']
    learning_rate = args['learning_rate']
    saving = args['saving']
    shuffle_mode = args['shuffle_mode']
    model_name = args['model_name']
    batch_size = args['batch_size']

    use_train_accuracy = args['use_train_accuracy']
    use_sgd = args['use_sgd']


    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    sliding_window_size_ratio = args['sliding_window_size_ratio']
    bismarck_buffer_size_ratio = args['bismarck_buffer_size_ratio']

    if (shuffle_mode == 'bismarck_mrs'):
        select_ratio_from_old_buffer = args['select_ratio_from_old_buffer']
    block_num = args['block_num']
    buffer_size_ratio = args['buffer_size_ratio']
    use_clustered_data = args['use_clustered_data']

    if (shuffle_mode == 'once_fully'):
        trainset = in_mem_once_fully_shuffle.InMemOnceFullyShuffleCifarDataset(
            base_dir=data_dir, 
            use_clustered_data=use_clustered_data, 
            train=True, transform=transform_train)

    elif (shuffle_mode == 'no_shuffle'):
        trainset = in_mem_no_shuffle.InMemNoShuffleCifarDataset(
            base_dir=data_dir, 
            use_clustered_data=use_clustered_data, 
            train=True, transform=transform_train)

    elif (shuffle_mode == 'always_fully'):
        trainset = in_mem_always_fully_shuffle.InMemAlwaysFullyShuffleCifarDataset(
            base_dir=data_dir, 
            use_clustered_data=use_clustered_data, 
            train=True, transform=transform_train)


    elif (shuffle_mode == 'bismarck_mrs'):
        old_buffer_checkpoint_dir = args['old_buffer_checkpoint_dir']
        trainset = in_mem_bismarck.InMemBismarckCifarDataset(
            base_dir=data_dir, 
            use_clustered_data=use_clustered_data, 
            bismarck_buffer_size_ratio=bismarck_buffer_size_ratio,
            select_ratio_from_old_buffer=select_ratio_from_old_buffer,
            old_buffer_checkpoint_dir=old_buffer_checkpoint_dir,
            train=True, transform=transform_train)
    
    elif (shuffle_mode == 'block'):
        trainset = in_mem_block.InMemBlockCifarDataset(
            base_dir=data_dir, 
            use_clustered_data=use_clustered_data, 
            block_num=block_num,
            buffer_size_ratio=buffer_size_ratio,
            drop_last=False,
            train=True, transform=transform_train)
    
    elif (shuffle_mode == 'block_only'):
        trainset = in_mem_block_only.InMemBlockOnlyCifarDataset(
            base_dir=data_dir, 
            use_clustered_data=use_clustered_data, 
            block_num=block_num,
            buffer_size_ratio=buffer_size_ratio,
            drop_last=False,
            train=True, transform=transform_train)

    elif (shuffle_mode == 'sliding_window'):
        trainset = in_mem_sliding_window.InMemSlidingWindowCifarDataset(
            base_dir=data_dir, 
            use_clustered_data=use_clustered_data, 
            sliding_window_size_ratio=sliding_window_size_ratio, 
            train=True, transform=transform_train)
    
    else:
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=download, transform=transform_train)
     

    trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
   

    if (use_train_accuracy):
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=download, transform=transform_train)
    else:
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=download, transform=transform_test)
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    writer.write('==> Building model..\n')

    if (model_name == 'ResNet18'):
        net = ResNet18()
    elif (model_name == 'VGG19'):
        net = VGG('VGG19')
 

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args['resume']:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()

    if (use_sgd):
        optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                        momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate,
                        weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_num)


    avg_exec_t = 0.0
    avg_grad_t = 0.0
    avg_loss_t = 0.0

    first_exec_t = 0.0
    first_grad_t = 0.0
    first_loss_t = 0.0

    second_exec_t = 0.0
    second_grad_t = 0.0
    second_loss_t = 0.0

    max_accuracy = 0.0

    print('[%s] Start training' % get_current_time())

    for epoch in range(start_epoch, start_epoch + iter_num):
        start = time.time()
        train(epoch, net, trainloader, device, optimizer, criterion)
        grad_end = time.time()
        (acc, test_loss) = test(epoch, net, testloader, device, criterion)
        loss_end = time.time()
        
        exec_t = loss_end - start
        grad_t = grad_end - start
        loss_t = exec_t - grad_t

      
        if saving == True and acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc

        scheduler.step()

        i = epoch
        avg_exec_t += exec_t
        avg_grad_t += grad_t
        avg_loss_t += loss_t

        if i == 0:
            first_exec_t = exec_t
            first_grad_t = grad_t
            first_loss_t = loss_t
        elif i == 1:
            second_exec_t = exec_t
            second_grad_t = grad_t
            second_loss_t = loss_t

        writer.write('[%s] [Epoch %2d] Loss = %.2f, acc = %.2f, exec_t = %.2fs, grad_t = %.2fs, loss_t = %.2fs' % 
            (get_current_time(), i + 1, test_loss, acc, round(exec_t, 2),
			round(grad_t, 2), round(loss_t, 2)))
        writer.write('\n')
        writer.flush()

        if acc > max_accuracy:
            max_accuracy = acc



    writer.write('[%s] [Finish] avg_exec_t = %.2fs, avg_grad_t = %.2fs, avg_loss_t = %.2fs' % 
            (get_current_time(), avg_exec_t / iter_num,
            avg_grad_t / iter_num, avg_loss_t / iter_num))
    writer.write('\n')

    if iter_num > 2:
        avg_exec_t -= first_exec_t
        avg_grad_t -= first_grad_t
        avg_loss_t -= first_loss_t

        writer.write('[%s] [-first] avg_exec_t = %.2fs, avg_grad_t = %.2fs, avg_loss_t = %.2fs' % 
                (get_current_time(), avg_exec_t / (iter_num - 1),
				avg_grad_t / (iter_num - 1), avg_loss_t / (iter_num - 1)))
        writer.write('\n')
		
        avg_exec_t -= second_exec_t
        avg_grad_t -= second_grad_t
        avg_loss_t -= second_loss_t

        writer.write('[%s] [-1 & 2] avg_exec_t = %.2fs, avg_grad_t = %.2fs, avg_loss_t = %.2fs' % 
                (get_current_time(), avg_exec_t / (iter_num - 2),
                avg_grad_t / (iter_num - 2), avg_loss_t / (iter_num - 2)))
        writer.write('\n')
        writer.write('[%s] [MaxAcc] max_accuracy = %.2f' % 
				(get_current_time(), max_accuracy))
        writer.write('\n')

def get_current_time() :
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_current_time_filename():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def main():

    log_base_dir = '/home/username/code/CorgiPile-PyTorch'
    data_dir = '/home/username/data/'
    
    log_dir = 'train_log_cifar10_sgd'
    
    model_name = 'ResNet18'
    #model_name = 'VGG19'
    
    data_name = 'cifar10'
    
    use_clustered_data = True

    use_train_accuracy = True # If False, it will compute and output test accuracy instead of train accuracy
    use_sgd = True # If false, it will use Adam instead of SGD

    batch_size = 128
    iter_num = 10
    num_workers = 1
    lr_decay = 0.95

    shuffle_modes = ['once_shuffle']
    #shuffle_modes = ['once_shuffle', 'block', 'block', 'sliding_window', 'bismarck_mrs', 'no_shuffle', 'block_only']
    #shuffle_modes = ['block', 'sliding_window', 'bismarck_mrs', 'no_shuffle', 'block_only']
    #shuffle_modes = ['block']

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'


    n_records = 0
    block_num = 500
    
    if (model_name == 'ResNet18'):
        learning_rate = 0.001
            
    elif (model_name == 'VGG19'):
        learning_rate = 0.001
    


    args = {}
    args['use_clustered_data'] = use_clustered_data
    args['use_train_accuracy'] = use_train_accuracy
    args['use_sgd'] = use_sgd
    
    args['model_name'] = model_name
    args['batch_size'] = batch_size
    args['iter_num'] = iter_num
    args['n_records'] = n_records
    args['learning_rate'] = learning_rate
    args['num_workers'] = num_workers
    args['data_name'] = data_name
    args['lr_decay'] = lr_decay


   
    args['resume'] = False
    args['data_dir'] = data_dir
    args['download'] = False
    args['saving'] = False
    # for our-block-based sgd
    buffer_size_ratio = 0.1  

    # for sliding_window
    sliding_window_size_ratio = 0.1

    # for bismarck_mrs
    bismarck_buffer_size_ratio = 0.1
    select_ratio_from_old_buffers = [0.4, 0.5]

    args['block_num'] = block_num
    args['buffer_size_ratio'] = buffer_size_ratio
    args['sliding_window_size_ratio'] = sliding_window_size_ratio
    args['bismarck_buffer_size_ratio'] = bismarck_buffer_size_ratio
    args['old_buffer_checkpoint_dir'] = log_base_dir + '/checkpoint/' + get_current_time_filename() + str(random.randint(1,100))

    
    for shuffle_mode in shuffle_modes:
        args['shuffle_mode'] = shuffle_mode

        if (shuffle_mode == 'bismarck_mrs'):
            for ratio in select_ratio_from_old_buffers:
                args['select_ratio_from_old_buffer'] = ratio
                log_txt = shuffle_mode + '_' + data_name + '_lr' + str(learning_rate) + '_ratio_' + str(ratio) + '_' + get_current_time_filename() + '.txt' 
                outdir = os.path.join(log_base_dir, log_dir, data_name, model_name, 'sgd-bs' + str(batch_size), shuffle_mode)
                log_file = os.path.join(outdir, log_txt)
                args['log_file'] = log_file
                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                main_worker(args)

        else:
            log_txt = shuffle_mode + '_' + data_name + '_lr' + str(learning_rate) + '_' + get_current_time_filename() + '.txt'
            outdir = os.path.join(log_base_dir, log_dir, data_name, model_name, 'sgd-bs' + str(batch_size), shuffle_mode)
    
            log_file = os.path.join(outdir, log_txt)
            args['log_file'] = log_file
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            main_worker(args)

   
if __name__ == '__main__':
    main()

