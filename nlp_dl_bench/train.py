import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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

        optimizer = optim.Adam(
            params = filter(lambda p: p.requires_grad, model.parameters()),
            lr = learning_rate

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
    # model_name = 'han', 'attbilstm', 'fasttext', 'textcnn', 'transformer'
    model_name = 'han'
    #model_name = 'transformer'
    # data_name = 'ag_news', 'dbpedia', 'yahoo_answers'
    data_name = 'ag_news'
    
    use_clustered_data = True

    #batch_size = 256
    batch_size = 128
    iter_num = 2
    num_workers = 1
    lr_decay = 0.95

    #shuffle_mode = ['once_shuffle', 'block', 'sliding_window', 'bismarck_mrs']
    #shuffle_mode = 'once_shuffle'
    #shuffle_mode = 'sliding_window'
    #shuffle_mode = 'bismarck_mrs'
    #shuffle_mode = 'block'
    shuffle_mode = 'block_only'
    #shuffle_mode = 'no_shuffle'


    
    n_records = 0
    
    if (data_name == 'ag_news'):
        if (model_name == 'han'):
            learning_rate = 0.001
            
        elif (model_name == 'attbilstm'):
            learning_rate = 0.001
        elif (model_name == 'fasttext'):
            learning_rate = 0.001
        elif (model_name == 'textcnn'):
            learning_rate = 0.001
        elif (model_name == 'transformer'):
            learning_rate = 0.001

        block_num = 240

    else:
        print ('Error in the data_name')

    args = {}
    args['use_clustered_data'] = use_clustered_data
    args['shuffle_mode'] = shuffle_mode
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
    select_ratio_from_old_buffer = 0.5

    args['block_num'] = block_num
    args['buffer_size_ratio'] = buffer_size_ratio
    args['sliding_window_size_ratio'] = sliding_window_size_ratio
    args['bismarck_buffer_size_ratio'] = bismarck_buffer_size_ratio
    args['select_ratio_from_old_buffer'] = select_ratio_from_old_buffer
    args['old_buffer_checkpoint_dir'] = '/mnt/ds3lab-scratch/xuliji/code/corgipile-pytorch/checkpoint/'


    if (shuffle_mode == 'bismarck_mrs'):
        ratio = args['select_ratio_from_old_buffer']
        log_txt = shuffle_mode + '_' + data_name + '_lr' + str(learning_rate) + '_ratio_' + str(ratio) + '_' + get_current_time_filename() + '.txt' 
    elif (shuffle_mode.endswith('feature_order')):
        selected_feature_index = args['selected_feature_index']
        log_txt = shuffle_mode + '_' + data_name + '_lr' + str(learning_rate) + '_feature_' + str(selected_feature_index) + '_' + get_current_time_filename() + '.txt'
    else:
        log_txt = shuffle_mode + '_' + data_name + '_lr' + str(learning_rate) + '_' + get_current_time_filename() + '.txt'

    outdir = os.path.join(base_dir, 'train_log_nlp_bench', data_name, model_name, 'sgd-bs' + str(batch_size), shuffle_mode)
   
    log_file = os.path.join(outdir, log_txt)

    args['log_file'] = log_file

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # writer = open(log_file, 'w')


    # for k in args:
    #     writer.write("[params] " + str(k) + " = " + str(args[k]) + "\n")
    # writer.flush()

    config = parse_opt(data_name, model_name)
    
    trainer = set_trainer(config, args)
    trainer.run_train(args)


'''
[lijie@db4ai-1 nlp_dl_bench]$ /usr/bin/python3 /datadisk/lijie/code/corgipile-pytorch/nlp_dl_bench/train.py
Loading embeddings from /ssddisk/data/text_data/outputs/ag_news/docs/glove.6B.300d.txt.pth.tar
Epoch: [0][0/1875]      Batch Time 0.332 (0.332)        Data Load Time 0.105 (0.105)    Loss 1.3754 (1.3754) Accuracy 0.359 (0.359)
Epoch: [0] finished, time consumed: 275.498

DECAYING learning rate.
The new learning rate is 0.000300

Epoch: [1][0/1875]      Batch Time 0.447 (0.447)        Data Load Time 0.115 (0.115)    Loss 0.2521 (0.2521) Accuracy 0.938 (0.938)
Epoch: [1] finished, time consumed: 287.535

DECAYING learning rate.
The new learning rate is 0.000090

Epoch: [2][0/1875]      Batch Time 0.491 (0.491)        Data Load Time 0.113 (0.113)    Loss 0.0871 (0.0871)    Accuracy 0.969 (0.969)
Epoch: [2] finished, time consumed: 287.233

DECAYING learning rate.
The new learning rate is 0.000027

Epoch: [3][0/1875]      Batch Time 0.454 (0.454)        Data Load Time 0.115 (0.115)    Loss 0.2018 (0.2018)    Accuracy 0.891 (0.891)
'''


'''
no_shuffle
[lijie@db4ai-1 nlp_dl_bench]$  /usr/bin/python3 /datadisk/lijie/code/corgipile-pytorch/nlp_dl_bench/train.py
[2022-01-14 05:19:33] Start loading data into memory
[2022-01-14 05:19:37] data_load_time = 3.87s
Loading embeddings from /ssddisk/data/text_data/outputs/ag_news/docs/glove.6B.300d.txt.pth.tar
Epoch: [0][0/1875]      Batch Time 0.358 (0.358)        Data Load Time 0.074 (0.074)    Loss 1.3381 (1.3381)    Accuracy 0.891 (0.891)
Epoch: [0] finished, time consumed: 285.021

DECAYING learning rate.
The new learning rate is 0.000300

Epoch: [1][0/1875]      Batch Time 0.405 (0.405)        Data Load Time 0.042 (0.042)    Loss 8.4698 (8.4698)    Accuracy 0.000 (0.000)
Epoch: [1] finished, time consumed: 304.336

DECAYING learning rate.
The new learning rate is 0.000090

Epoch: [2][0/1875]      Batch Time 0.397 (0.397)        Data Load Time 0.047 (0.047)    Loss 5.9393 (5.9393)    Accuracy 0.000 (0.000)
Epoch: [2] finished, time consumed: 303.222

DECAYING learning rate.
The new learning rate is 0.000027

Epoch: [3][0/1875]      Batch Time 0.394 (0.394)        Data Load Time 0.045 (0.045)    Loss 3.9617 (3.9617)    Accuracy 0.000 (0.000)
Epoch: [3] finished, time consumed: 301.601

DECAYING learning rate.
The new learning rate is 0.000008

Epoch: [4][0/1875]      Batch Time 0.391 (0.391)        Data Load Time 0.043 (0.043)    Loss 2.6433 (2.6433)    Accuracy 0.016 (0.016)
Epoch: [4] finished, time consumed: 303.184

DECAYING learning rate.
The new learning rate is 0.000002
'''