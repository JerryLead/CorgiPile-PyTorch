import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.backends.cudnn as cudnn
from torch import optim, nn
import time


import models
from trainer import Trainer
from datasets import load_data
from utils import load_embeddings, load_checkpoint, parse_opt




def set_trainer(config, args):

  
    model_name = args['model_name']
    batch_size = args['batch_size']
    data_name = args['data_name']
    learning_rate = args['learning_rate']
    iter_num = args['iter_num']
    lr_decay = args['lr_decay']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    
    # load a checkpoint
    if config.checkpoint is not None:
        # load data
        train_loader = load_data(config, 'train', False)
        model, optimizer, word_map, start_epoch = load_checkpoint(config.checkpoint, device)
        print('\nLoaded checkpoint from epoch %d.\n' % (start_epoch - 1))

    # or initialize model
    else:
        start_epoch = 0

        # load data
        train_loader, test_loader, embeddings, emb_size, word_map, n_classes, vocab_size, train_tuple_num = load_data(args, config, 'train', True)

        model = models.make(
            config = config,
            n_classes = n_classes,
            vocab_size = vocab_size,
            embeddings = embeddings,
            emb_size = emb_size
        )

        # optimizer = optim.Adam(
        #     params = filter(lambda p: p.requires_grad, model.parameters()),
        #     lr = learning_rate

        # )

        # optimizer = optim.SGD(
        #     params = filter(lambda p: p.requires_grad, model.parameters()),
        #     lr = learning_rate,
        #     momentum=0.9, 
        #     weight_decay=5e-4
        # )

        optimizer = optim.SGD(
            params = model.parameters(),
            lr = learning_rate,
            weight_decay=5e-4
        )

    # loss functions
    loss_function = nn.CrossEntropyLoss()

    # move to device
    model = model.to(device)
    loss_function = loss_function.to(device)

    trainer = Trainer(
        num_epochs = iter_num,
        start_epoch = start_epoch,
        train_loader = train_loader,
        test_loader = test_loader,
        train_tuple_num = train_tuple_num,
        model = model,
        model_name = model_name,
        loss_function = loss_function,
        optimizer = optimizer,
        lr_decay = lr_decay,
        dataset_name = data_name,
        word_map = word_map,
        grad_clip = config.grad_clip,
        print_freq = config.print_freq,
        checkpoint_path = config.checkpoint_path,
        checkpoint_basename = config.checkpoint_basename,
        tensorboard = config.tensorboard,
        log_dir = config.log_dir
    )

    return trainer

def get_current_time_filename():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

if __name__ == '__main__':
    # batch_size: 128  # batch size
    # lr: 0.001  # learning rate
    # lr_decay: 0.3  # a factor to multiply learning rate with (0, 1)
    # workers: 1  # number of workers for loading data in the DataLoader
    # num_epochs: 5  # number of epochs to run
    base_dir = '/mnt/ds3lab-scratch/xuliji/code/corgipile-pytorch'
    log_dir = 'train_log_nlp_bench_50_sgd_mo9_wd4'

    # model_name = 'han', 'textcnn'
    #model_name = 'han'
    model_name = 'textcnn'

    data_name = 'yelp_review_full'

    
    use_clustered_data = True

    batch_size = 256
    #batch_size = 128
    iter_num = 50
    num_workers = 1
    lr_decay = 0.95

    shuffle_modes = ['once_shuffle', 'block', 'sliding_window', 'bismarck_mrs', 'no_shuffle', 'block_only', 'block', 'block']

    
    n_records = 0
    
    if (data_name == 'yelp_review_full'):
        if (model_name == 'han'):
            learning_rate = 0.1
        elif (model_name == 'textcnn'):
            learning_rate = 0.01

        block_num = 650

    else:
        print ('Error in the data_name')

    args = {}
    args['use_clustered_data'] = use_clustered_data
    
    args['model_name'] = model_name
    args['batch_size'] = batch_size
    args['iter_num'] = iter_num
    args['n_records'] = n_records
    args['learning_rate'] = learning_rate
    args['num_workers'] = num_workers
    args['data_name'] = data_name
    args['lr_decay'] = lr_decay

    # for our-block-based sgd
    buffer_size_ratio = 0.1  

    # for sliding_window
    sliding_window_size_ratio = 0.1

    # for bismarck_mrs
    bismarck_buffer_size_ratio = 0.1
    select_ratio_from_old_buffers = [0.5, 0.4]

    args['block_num'] = block_num
    args['buffer_size_ratio'] = buffer_size_ratio
    args['sliding_window_size_ratio'] = sliding_window_size_ratio
    args['bismarck_buffer_size_ratio'] = bismarck_buffer_size_ratio
    args['select_ratio_from_old_buffer'] = 0.5
    args['old_buffer_checkpoint_dir'] = '/mnt/ds3lab-scratch/xuliji/code/corgipile-pytorch/checkpoint/' + get_current_time_filename() + '0'

    
    for shuffle_mode in shuffle_modes:
        args['shuffle_mode'] = shuffle_mode

        if (shuffle_mode == 'bismarck_mrs'):
            for ratio in select_ratio_from_old_buffers:
                args['select_ratio_from_old_buffer'] = ratio
                log_txt = shuffle_mode + '_' + data_name + '_lr' + str(learning_rate) + '_ratio_' + str(ratio) + '_' + get_current_time_filename() + '.txt' 
                outdir = os.path.join(base_dir, log_dir, data_name, model_name, 'sgd-bs' + str(batch_size), shuffle_mode)
                log_file = os.path.join(outdir, log_txt)
                args['log_file'] = log_file
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                
                config = parse_opt(data_name, model_name)
    
                trainer = set_trainer(config, args)
                trainer.run_train(args)

        else:
            log_txt = shuffle_mode + '_' + data_name + '_lr' + str(learning_rate) + '_' + get_current_time_filename() + '.txt'
            outdir = os.path.join(base_dir, log_dir, data_name, model_name, 'sgd-bs' + str(batch_size), shuffle_mode)
    
            log_file = os.path.join(outdir, log_txt)
            args['log_file'] = log_file
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            config = parse_opt(data_name, model_name)
    
            trainer = set_trainer(config, args)
            trainer.run_train(args)

   

  



    
