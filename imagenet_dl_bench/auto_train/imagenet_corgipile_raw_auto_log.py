import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import sys
import numpy as np
import io

from torch.distributed.algorithms.join import Join

from PIL import Image

sys.path.append("../shuffleformat")
sys.path.append(".")
import shuffleformat.tfrecord as tfrecord
import shuffleformat.corgipile as corgipile


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

best_acc1 = 0

'''
    # Single node, multiple GPUs: 
    #         python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' 
    #         --dist-backend 'nccl' --multiprocessing-distributed 
    #         --world-size 1 --rank 0 [imagenet-folder with train and val folders]

    # Multiple nodes:
    # Node 0: python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT'
    #         --dist-backend 'nccl' --multiprocessing-distributed 
    #         --world-size 2 --rank 0 
    #         [imagenet-folder with train and val folders]
    # Node 1: python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' 
    #         --dist-backend 'nccl' --multiprocessing-distributed
    #         --world-size 2 --rank 1 
    #         [imagenet-folder with train and val folders]
'''

def get_data_path(image_type, data_name):
    if image_type == 'raw':
        if data_name == 'imagenette':
            data_path = "/mnt/ds3lab-scratch/xuliji/corgipile_data/imagenette2-raw-tfrecords"
        elif data_name == 'ImageNet':
            data_path = "/mnt/ds3lab-scratch/xuliji/corgipile_data/ImageNet-all-raw-tfrecords"
        else:
            raise ValueError('This dataset is not supported currently!')
    elif image_type == 'RGB':
        if data_name == 'imagenette':
            data_path = "/mnt/ds3lab-scratch/xuliji/corgipile_data/imagenette2-tfrecords"
        elif data_name == 'ImageNet':
            data_path = "/mnt/ds3lab-scratch/xuliji/corgipile_data/ImageNet-all-tfrecords"
        else:
            raise ValueError('This dataset is not supported currently!')
    else:
        raise ValueError('This data type is not supported currently!')

    return data_path

def main():

    image_type = 'raw' # or 'RGB'
    data_name = 'imagenette'
    # data_name = 'ImageNet'
    data_path = get_data_path(image_type, data_name)
    
    log_base_dir = '/mnt/ds3lab-scratch/xuliji/code/corgipile-pytorch'
    log_dir = 'train_log_' + data_name

    model_name = "resnet18"
    #model_name = "resnet50"

    
    data_loading_workers_num = 1
    epoch_num = 3
    start_epoch = 0
    batch_sizes = [256]
    learning_rates = [0.1]
    momentum = 0.9
    weight_decay = 1e-4
    print_freq = 5
    world_size = 1 # the number of nodes
    rank = 0
    dist_url = 'tcp://127.0.0.1:23456'
    dist_backend = 'nccl'
    multiprocessing_distributed = True

    # once_shuffle: the image data is shufffled => tfrecords.
    # block: the image data is clustered.
    # no_shuffle: the image data is clustered.
    # epoch_shuffle: randomly access each tuple in each epoch.
    # shuffle_modes = ['once_shuffle', 'block', 'no_shuffle', 'epoch_shuffle']
    shuffle_modes = ['block']


    block_num = 112 #200 #563 #112 #14000
    buffer_size_ratio = 0.1

    seed = None
    gpu = None
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3, 4'
    os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6'

    args = {}
    args['image_type'] = image_type
    args['data'] = data_path
    args['model_name'] = model_name
    args['arch'] = model_name
    args['epochs'] = epoch_num
    args['workers'] = data_loading_workers_num
    args['data_name'] = data_name
    args['momentum'] = momentum
    args['weight_decay'] = weight_decay
    args['print_freq'] = print_freq
    args['world_size'] = world_size 
    args['rank'] = rank
    args['dist_backend'] = dist_backend
    args['dist_url'] = dist_url

    args['block_num'] = block_num
    args['buffer_size_ratio'] = buffer_size_ratio
    args['start_epoch'] = start_epoch

    args['seed'] = seed
    args['gpu'] = gpu
    args['multiprocessing_distributed'] = multiprocessing_distributed
    args['pretrained'] = False
    args['resume'] = False
    args['evaluate'] = False
    
    for batch_size in batch_sizes:
        args['batch_size'] = batch_size
        for learning_rate in learning_rates:
            args['lr'] = learning_rate
            for shuffle_mode in shuffle_modes:
                args['shuffle_mode'] = shuffle_mode
                
                acc_log_txt = shuffle_mode + '_' + data_name + '_lr' + str(learning_rate) + '_' + get_current_time_filename() #+ '.txt'
                batch_run_log = shuffle_mode + '_' + data_name + '_lr' + str(learning_rate) + '_' + get_current_time_filename() #+ '.log'
                outdir = os.path.join(log_base_dir, log_dir, data_name, model_name, 'sgd-bs' + str(batch_size), shuffle_mode)
            
                acc_log_file = os.path.join(outdir, acc_log_txt)
                batch_log_file = os.path.join(outdir, batch_run_log)
                args['acc_log_file'] = acc_log_file
                args['batch_log_file'] = batch_log_file
                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                if args['seed'] is not None:
                    random.seed(args['seed'])
                    torch.manual_seed(args['seed'])
                    cudnn.deterministic = True
                    warnings.warn('You have chosen to seed training. '
                                'This will turn on the CUDNN deterministic setting, '
                                'which can slow down your training considerably! '
                                'You may see unexpected behavior when restarting '
                                'from checkpoints.')

                if args['gpu'] is not None:
                    warnings.warn('You have chosen a specific GPU. This will completely '
                                'disable data parallelism.')

                if args['dist_url'] == "env://" and args['world_size'] == -1:
                    args['world_size'] = int(os.environ["WORLD_SIZE"])

                # Single node, multiple GPUs: world_size = 1, Multiple nodes: e.g., world_size = 2
                args['distributed'] = args['world_size'] > 1 or args['multiprocessing_distributed']

                ngpus_per_node = torch.cuda.device_count()
                args['ngpus_per_node'] = ngpus_per_node
        
                if args['multiprocessing_distributed']:
                    # Since we have ngpus_per_node processes per node, the total world_size
                    # needs to be adjusted accordingly
                    # world_size = 8 GPUs * 2
                    args['world_size'] = ngpus_per_node * args['world_size']
                    # Use torch.multiprocessing.spawn to launch distributed processes: the
                    # main_worker process function
                    # spawn nprocs
                    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
                else:
                    # Simply call main_worker function
                    main_worker(args['gpu'], ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args, join=True):
    global best_acc1

    args['gpu'] = gpu
    args['ngpus_per_node'] = ngpus_per_node

    acc_log_file = args['acc_log_file'] + '-gpu' + str(gpu) + '.txt'
    batch_log_file = args['batch_log_file'] + '-gpu' + str(gpu) + '.log'


    writer = sys.stdout
    batch_log_writer = sys.stdout

    for k in args:
        writer.write("[params] " + str(k) + " = " + str(args[k]) + '\n')
        batch_log_writer.write("[params] " + str(k) + " = " + str(args[k]) + '\n')
    
    writer.flush()
    batch_log_writer.flush()

    writer.write('[%s] Start iteration\n' % get_current_time())
    batch_log_writer.write('[%s] Start iteration\n' % get_current_time())

    if gpu is not None:
        writer.write("Use GPU: {} for training\n".format(gpu))
        batch_log_writer.write("Use GPU: {} for training\n".format(gpu))
        # print("Use GPU: {} for training".format(args['gpu']))

    if args['distributed']:
        if args['dist_url'] == "env://" and args.rank == -1:
            args['rank'] = int(os.environ["RANK"])
        if args['multiprocessing_distributed']:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            # rank = 1 * 8 + gpu (global rank)
            args['rank'] = args['rank'] * ngpus_per_node + gpu
            
        # here, world_size = global_procs_num, rank = global_current_procs_num
        dist.init_process_group(backend=args['dist_backend'], init_method=args['dist_url'],
                                world_size=args['world_size'], rank=args['rank'])
    # create model
    if args['pretrained']:
        arch = args['arch']
        writer.write("=> using pre-trained model '{}'\n".format(arch))
        batch_log_writer.write("=> using pre-trained model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=True)
    else:
        arch = args['arch']
        writer.write("=> creating model '{}'\n".format(arch))
        batch_log_writer.write("=> creating model '{}'\n".format(arch))
        model = models.__dict__[arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args['distributed']:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if gpu is not None:
            torch.cuda.set_device(gpu)
            model.cuda(gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            ngpus_per_node = args['ngpus_per_node']
            args['batch_size'] = int(args['batch_size'] / ngpus_per_node)
            args['workers'] = int((args['workers'] + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        arch = args['arch']
        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    optimizer = torch.optim.SGD(model.parameters(), args['lr'],
                                momentum=args['momentum'],
                                weight_decay=args['weight_decay'])
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    #scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    
    # optionally resume from a checkpoint
    if args['resume']:
        if os.path.isfile(args['resume']):
            
            writer.write("=> loading checkpoint '{}'\n".format(args['resume']))
            batch_log_writer.write("=> loading checkpoint '{}'".format(args['resume']))
            if gpu is None:
                checkpoint = torch.load(args['resume'])
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(gpu)
                checkpoint = torch.load(args['resume'], map_location=loc)
            args['start_epoch'] = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            
            writer.write("=> loaded checkpoint '{}' (epoch {})\n"
                  .format(args.resume, checkpoint['epoch']))
            batch_log_writer.write("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.resume, checkpoint['epoch']))
        else:
            writer.write("=> no checkpoint found at '{}'\n".format(args.resume))
            batch_log_writer.write("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args['data'], 'train')
    valdir = os.path.join(args['data'], 'val')

    train_tfrecord_file = os.path.join(traindir, "train_clustered.tfrecord")
    train_index_file = os.path.join(traindir, "train_clustered.index")

    val_tfrecord_file = os.path.join(valdir, "val_clustered.tfrecord")
    val_index_file = os.path.join(valdir, "val_clustered.index")

    if args['image_type'] == 'RGB':
        description = {"image": "byte", "label": "int", "index": "int", "width": "int", "height": "int"}
    else:
        description = {"image": "byte", "label": "int", "index": "int"}

    shuffle_mode = args['shuffle_mode']


    if shuffle_mode == 'block':
        train_dataset = corgipile.dataset.CorgiPileTFRecordDataset(
                                train_tfrecord_file, 
                                train_index_file,
                                block_num=args['block_num'],
                                buffer_size_ratio=args['buffer_size_ratio'],
                                description=description,
                                transform=decode_train_raw_image,
                                trans_after_buffered=True,
                                distributed=args['distributed'])   
    elif shuffle_mode == 'no_shuffle':
        train_dataset = corgipile.dataset.SeqTFRecordDataset(
                                train_tfrecord_file, 
                                train_index_file,
                                num_workers = args['workers'],
                                description=description, 
                                transform=decode_train_raw_image,
                                trans_after_buffered=True,
                                distributed=args['distributed'],
                                data_partition=True)    

    elif shuffle_mode == 'epoch_shuffle':
        train_dataset = corgipile.dataset.RandomAccessTFRecordDataset(
                                train_tfrecord_file, 
                                train_index_file,
                                description=description, 
                                transform=decode_train_raw_image,
                                trans_after_buffered=True,
                                distributed=args['distributed'])    
                
    val_dataset = corgipile.dataset.SeqTFRecordDataset(
                                val_tfrecord_file, 
                                val_index_file,
                                num_workers = args['workers'],
                                description=description, 
                                transform=decode_val_raw_image,
                                trans_after_buffered=True,
                                distributed=args['distributed'],
                                data_partition=True)    
  
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=False,
        num_workers=args['workers'], pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args['batch_size'], shuffle=False,
        num_workers=args['workers'], pin_memory=True)

    if args['evaluate']:
        validate(batch_log_writer, val_loader, model, criterion, args)
        return
    
    avg_exec_t = 0.0
    avg_grad_t = 0.0
    avg_loss_t = 0.0

    first_exec_t = 0.0
    first_grad_t = 0.0
    first_loss_t = 0.0

    second_exec_t = 0.0
    second_grad_t = 0.0
    second_loss_t = 0.0

    max_acc1 = 0.0
    max_acc5 = 0.0

    batch_log_writer.write('[%s] Start training\n' % get_current_time())

    for epoch in range(args['start_epoch'], args['epochs']):
        start = time.time()

        if shuffle_mode == 'block' or shuffle_mode == 'epoch_shuffle':
            train_dataset.set_epoch(epoch)

        with Join([model]):
            # train for one epoch
            train(batch_log_writer, train_loader, model, criterion, optimizer, epoch, args)

        grad_end = time.time()
        # evaluate on validation set
        acc1, acc5, num_val_records = validate(batch_log_writer, val_loader, model, criterion, args)
        loss_end = time.time()
        
        exec_t = loss_end - start
        grad_t = grad_end - start
        loss_t = exec_t - grad_t

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

        writer.write('[%s] [Epoch %2d] acc1 = %.2f, acc5 = %.2f, train_t = %.2fs, val_t = %.2fs, num_record = %d\n' % 
            (get_current_time(), i + 1, acc1, acc5, round(exec_t, 2), round(grad_t, 2), num_val_records))
        writer.flush()

        if acc1 > max_acc1:
            max_acc1 = acc1
        
        if acc5 > max_acc5:
            max_acc5 = acc5


        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args['multiprocessing_distributed'] or (args['multiprocessing_distributed']
                and args['rank'] % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args['arch'],
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best)

    epoch_num = args['epochs'] - args['start_epoch']
    writer.write('[%s] [Finish] avg_exec_t = %.2fs, avg_train_t = %.2fs, avg_val_t = %.2fs\n' % 
            (get_current_time(), avg_exec_t / epoch_num,
            avg_grad_t / epoch_num, avg_loss_t / epoch_num))
    writer.write('\n')

    if epoch_num > 2:
        avg_exec_t -= first_exec_t
        avg_grad_t -= first_grad_t
        avg_loss_t -= first_loss_t

        writer.write('[%s] [-first] avg_exec_t = %.2fs, avg_train_t = %.2fs, avg_val_t = %.2fs\n' % 
                (get_current_time(), avg_exec_t / (epoch_num - 1),
				avg_grad_t / (epoch_num - 1), avg_loss_t / (epoch_num - 1)))

		
        avg_exec_t -= second_exec_t
        avg_grad_t -= second_grad_t
        avg_loss_t -= second_loss_t

        writer.write('[%s] [-1 & 2] avg_exec_t = %.2fs, avg_train_t = %.2fs, avg_val_t = %.2fs\n' % 
                (get_current_time(), avg_exec_t / (epoch_num - 2),
                avg_grad_t / (epoch_num - 2), avg_loss_t / (epoch_num - 2)))
  
        writer.write('[%s] [MaxAcc] max_acc1 = %.2f, max_acc5 = %.2f\n' % 
				(get_current_time(), max_acc1, max_acc5))

    writer.close()
    batch_log_writer.close()


def train(writer, train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        writer,
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    gpu = args['gpu']

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if gpu is not None:
            images = images.cuda(gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args['print_freq'] == 0:
            progress.display(i)


def validate(writer, val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        writer,
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    gpu = args['gpu']
    num_records = 0

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args['print_freq'] == 0:
                progress.display(i)

            num_records += images.size(0)
        progress.display_summary()

    return (top1.avg, top5.avg, num_records)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, writer, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.writer = writer

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        self.writer.write('[' + get_current_time() + '] ' + '\t'.join(entries) + '\n')
        self.writer.flush()
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        # print(' '.join(entries))
        self.writer.write('[' + get_current_time() + '] ' + ' '.join(entries) + '\n')
        self.writer.flush()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# decode image
def decode_train_RGB_image(features):
    width = features["width"][0]
    height = features["height"][0]

    img_numpy_array = features["image"].reshape((height, width, 3))
    features["image"] = Image.fromarray(np.uint8(img_numpy_array))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    read_trans = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    features["image"] = read_trans(features["image"])

    features["label"] = features["label"][0]

    return (features["image"], features["label"])



def decode_train_raw_image(features):

    img_bytes = features["image"]
    features["image"] = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    read_trans = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    features["image"] = read_trans(features["image"])

    features["label"] = features["label"][0]

    return (features["image"], features["label"])



def decode_val_raw_image(features):

    img_bytes = features["image"]
    features["image"] = Image.open(io.BytesIO(img_bytes)).convert("RGB")


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    read_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    features["image"] = read_trans(features["image"])

  

    features["label"] = features["label"][0]

    return (features["image"], features["label"])




def decode_val_RGB_image(features):
    width = features["width"][0]
    height = features["height"][0]

    img_numpy_array = features["image"].reshape((height, width, 3))
    features["image"] = Image.fromarray(np.uint8(img_numpy_array))
   

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    read_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    features["image"] = read_trans(features["image"])

  

    features["label"] = features["label"][0]

    return (features["image"], features["label"])

def get_current_time() :
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_current_time_filename():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

if __name__ == '__main__':
    main()