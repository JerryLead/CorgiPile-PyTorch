"""
Load data from manually preprocessed data (see ``datasets/prepocess/``).
"""

import os
import json
from typing import Dict, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader

from utils import load_embeddings
from utils.opts import Config
from .info import get_label_map

import sys

sys.path.append("../../nlpformat")
sys.path.append("../")
sys.path.append(".")


from nlpformat import in_mem_bismarck, in_mem_block, in_mem_block_only, in_mem_no_shuffle, in_mem_sliding_window, in_mem_once_fully_shuffle


'''
class DocDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches
    (for document classification).

    Parameters
    ----------
    data_folder : str
        Path to folder where data files are stored

    split : str
        Split, one of 'TRAIN' or 'TEST'
    """
    def __init__(self, data_folder: str, split: str) -> None:
        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        # load data
        self.data = torch.load(os.path.join(data_folder, split + '_data.pth.tar'))

    def __getitem__(self, i: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        return torch.LongTensor(self.data['docs'][i]), \
               torch.LongTensor([self.data['sentences_per_document'][i]]), \
               torch.LongTensor(self.data['words_per_sentence'][i]), \
               torch.LongTensor([self.data['labels'][i]])

    def __len__(self) -> int:
        return len(self.data['labels'])


class SentDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches
    (for sentence classification).

    Parameters
    ----------
    data_folder : str
        Path to folder where data files are stored

    split : str
        Split, one of 'TRAIN' or 'TEST'
    """
    def __init__(self, data_folder: str, split: str) -> None:
        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        # load data
        self.data = torch.load(os.path.join(data_folder, split + '_data.pth.tar'))

    def __getitem__(self, i: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        return torch.LongTensor(self.data['sents'][i]), \
               torch.LongTensor([self.data['words_per_sentence'][i]]), \
               torch.LongTensor([self.data['labels'][i]])

    def __len__(self) -> int:
        return len(self.data['labels'])
'''

def load_data(args,
    config: Config, split: str, build_vocab: bool = True
) -> Union[DataLoader, Tuple[DataLoader, torch.Tensor, int, Dict[str, int], int, int]]:
    """
    Load data from files output by ``prepocess.py``.

    Parameters
    ----------
    config : Config
        Configuration settings

    split : str
        'trian' / 'test'

    build_vocab : bool
        Build vocabulary or not. Only makes sense when split = 'train'.

    Returns
    -------
    split = 'test':
        test_loader : DataLoader
            Dataloader for test data

    split = 'train':
        build_vocab = Flase:
            train_loader : DataLoader
                Dataloader for train data

        build_vocab = True:
            train_loader : DataLoader
                Dataloader for train data

            embeddings : torch.Tensor
                Pre-trained word embeddings (None if config.emb_pretrain = False)

            emb_size : int
                Embedding size (config.emb_size if config.emb_pretrain = False)

            word_map : Dict[str, int]
                Word2ix map

            n_classes : int
                Number of classes

            vocab_size : int
                Size of vocabulary
    """
    split = split.lower()
    assert split in {'train', 'test'}

    data_folder = config.output_path
    num_workers = args['num_workers']
    shuffle_mode = args['shuffle_mode']
    model_name = args['model_name']
    batch_size = args['batch_size']
    data_name = args['data_name']


    sliding_window_size_ratio = args['sliding_window_size_ratio']
    bismarck_buffer_size_ratio = args['bismarck_buffer_size_ratio']

    if (shuffle_mode == 'bismarck_mrs'):
        select_ratio_from_old_buffer = args['select_ratio_from_old_buffer']
    block_num = args['block_num']
    buffer_size_ratio = args['buffer_size_ratio']
    use_clustered_data = args['use_clustered_data']




    # test
    if split == 'test':

        if (model_name == 'han'):
            testset = in_mem_no_shuffle.InMemNoShuffleDocDataset(
                    data_folder=data_folder, 
                    split=split,
                    use_clustered_data=use_clustered_data
                    )
        else:
            testset = in_mem_no_shuffle.InMemNoShuffleSentDataset(
                    data_folder=data_folder, 
                    split=split,
                    use_clustered_data=use_clustered_data
                    )

        
        test_tuple_num = len(testset)
     

        test_loader = DataLoader(
            testset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = True
        )
        return test_loader, test_tuple_num

    # train
    else:

        if (model_name == 'han'):
            if (shuffle_mode == 'once_shuffle' or shuffle_mode == 'once_fully'):
                trainset = in_mem_once_fully_shuffle.InMemOnceFullyShuffleDocDataset(
                    data_folder=data_folder, 
                    split=split,
                    use_clustered_data=use_clustered_data
                    )
    
                testset = trainset
            elif (shuffle_mode == 'no_shuffle'):
                trainset = in_mem_no_shuffle.InMemNoShuffleDocDataset(
                    data_folder=data_folder, 
                    split=split,
                    use_clustered_data=use_clustered_data
                    )
                testset = trainset

            elif (shuffle_mode == 'bismarck_mrs'):
                old_buffer_checkpoint_dir = args['old_buffer_checkpoint_dir']
                trainset = in_mem_bismarck.InMemBismarckDocDataset(
                    data_folder=data_folder, 
                    split=split,
                    use_clustered_data=use_clustered_data, 
                    bismarck_buffer_size_ratio=bismarck_buffer_size_ratio,
                    select_ratio_from_old_buffer=select_ratio_from_old_buffer,
                    old_buffer_checkpoint_dir=old_buffer_checkpoint_dir)

                testset = in_mem_no_shuffle.InMemNoShuffleDocDataset(
                    data_folder=data_folder, 
                    split=split,
                    use_clustered_data=use_clustered_data
                    )
           

            elif (shuffle_mode == 'block'):
                trainset = in_mem_block.InMemBlockDocDataset(
                    data_folder=data_folder,  
                    split=split,
                    use_clustered_data=use_clustered_data, 
                    block_num=block_num,
                    buffer_size_ratio=buffer_size_ratio,
                    drop_last=False
                    )
                testset = trainset
            elif (shuffle_mode == 'block_only'):
                trainset = in_mem_block_only.InMemBlockOnlyDocDataset(
                    data_folder=data_folder,  
                    split=split,
                    use_clustered_data=use_clustered_data, 
                    block_num=block_num,
                    buffer_size_ratio=buffer_size_ratio,
                    drop_last=False)
                testset = trainset

            elif (shuffle_mode == 'sliding_window'):
                trainset = in_mem_sliding_window.InMemSlidingWindowDocDataset(
                    data_folder=data_folder,  
                    split=split,
                    use_clustered_data=use_clustered_data, 
                    sliding_window_size_ratio=sliding_window_size_ratio, 
                    )
                testset = trainset
        else:
            if (shuffle_mode == 'once_shuffle' or shuffle_mode == 'once_fully'):
                trainset = in_mem_once_fully_shuffle.InMemOnceFullyShuffleSentDataset(
                    data_folder=data_folder, 
                    split=split,
                    use_clustered_data=use_clustered_data
                    )
                testset = trainset
            elif (shuffle_mode == 'no_shuffle'):
                trainset = in_mem_no_shuffle.InMemNoShuffleSentDataset(
                    data_folder=data_folder,  
                    split=split,
                    use_clustered_data=use_clustered_data
                    )
                testset = trainset

            elif (shuffle_mode == 'bismarck_mrs'):
                old_buffer_checkpoint_dir = args['old_buffer_checkpoint_dir']
                trainset = in_mem_bismarck.InMemBismarckSentDataset(
                    data_folder=data_folder, 
                    split=split,
                    use_clustered_data=use_clustered_data, 
                    bismarck_buffer_size_ratio=bismarck_buffer_size_ratio,
                    select_ratio_from_old_buffer=select_ratio_from_old_buffer,
                    old_buffer_checkpoint_dir=old_buffer_checkpoint_dir)

                testset = in_mem_no_shuffle.InMemNoShuffleSentDataset(
                    data_folder=data_folder,  
                    split=split,
                    use_clustered_data=use_clustered_data
                    )
        
            elif (shuffle_mode == 'block'):
                trainset = in_mem_block.InMemBlockSentDataset(
                    data_folder=data_folder, 
                    split=split,
                    use_clustered_data=use_clustered_data, 
                    block_num=block_num,
                    buffer_size_ratio=buffer_size_ratio,
                    drop_last=False
                    )
                testset = trainset
            elif (shuffle_mode == 'block_only'):
                trainset = in_mem_block_only.InMemBlockOnlySentDataset(
                    data_folder=data_folder, 
                    split=split,
                    use_clustered_data=use_clustered_data, 
                    block_num=block_num,
                    buffer_size_ratio=buffer_size_ratio,
                    drop_last=False)
                testset = trainset

            elif (shuffle_mode == 'sliding_window'):
                trainset = in_mem_sliding_window.InMemSlidingWindowSentDataset(
                    data_folder=data_folder, 
                    split=split,
                    use_clustered_data=use_clustered_data, 
                    sliding_window_size_ratio=sliding_window_size_ratio, 
                    )
                testset = trainset


        train_tuple_num = len(trainset)
        
        # writer.write('[Computed] train_tuple_num = %d' % (train_tuple_num))
        # writer.write('\n')

        # dataloaders
        train_loader = DataLoader(
            trainset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = True
        )

        test_loader = DataLoader(
            testset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = True
        )

        if build_vocab == False:
            return train_loader

        else:
            # load word2ix map
            with open(os.path.join(data_folder, 'word_map.json'), 'r') as j:
                word_map = json.load(j)
            # size of vocabulary
            vocab_size = len(word_map)

            # number of classes
            label_map, _ = get_label_map(data_name)
            n_classes = len(label_map)

            # word embeddings
            if config.emb_pretrain == True:
                # load Glove as pre-trained word embeddings for words in the word map
                emb_path = os.path.join(config.emb_folder, config.emb_filename)
                embeddings, emb_size = load_embeddings(
                    emb_file = os.path.join(config.emb_folder, config.emb_filename),
                    word_map = word_map,
                    output_folder = config.output_path
                )
            # or initialize embedding weights randomly
            else:
                embeddings = None
                emb_size = config.emb_size

            return train_loader, test_loader, embeddings, emb_size, word_map, n_classes, vocab_size, train_tuple_num
