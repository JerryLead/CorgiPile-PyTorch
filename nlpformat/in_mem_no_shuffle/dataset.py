import numpy as np
import random
import time

import torch.utils.data

from nlpformat.loader import nlp_format_dataloader


"""
Load data from manually preprocessed data (see ``datasets/prepocess/``).
"""

import os
from typing import Tuple
import torch
from torch.utils.data import Dataset



class InMemNoShuffleDocDataset(torch.utils.data.Dataset):

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

    def __init__(self,
                 data_folder: str, split: str,
                 use_clustered_data: bool = True
                 ) -> None:
        super(InMemNoShuffleDocDataset, self).__init__()
        self.data_folder = data_folder
        self.use_clustered_data = use_clustered_data

        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        self.data_buffer = []
        self.__load_data__()
        # load data
       
     
    def __getitem__(self, idx):
        return torch.LongTensor(self.data_buffer['docs'][idx]), \
               torch.LongTensor([self.data_buffer['sentences_per_document'][idx]]), \
               torch.LongTensor(self.data_buffer['words_per_sentence'][idx]), \
               torch.LongTensor([self.data_buffer['labels'][idx]])

    def __load_data__(self):
        print('[%s] Start loading data into memory' % nlp_format_dataloader.get_current_time())
        load_start_time = time.time()
        self.data_buffer = torch.load(os.path.join(self.data_folder, self.split + '_data.pth.tar'))
        self.num_records = len(self.data_buffer['labels'])

        if (self.use_clustered_data):
            zipped = zip(self.data_buffer['docs'], self.data_buffer['sentences_per_document'], 
                         self.data_buffer['words_per_sentence'], self.data_buffer['labels'])
            sort_zipped = sorted(zipped, key=lambda x:(x[3]))
            self.data_buffer['docs'], self.data_buffer['sentences_per_document'], self.data_buffer['words_per_sentence'], self.data_buffer['labels'] = zip(*sort_zipped)
   
        
        load_end_time = time.time()
        print("[%s] data_load_time = %.2fs" % (nlp_format_dataloader.get_current_time(), (load_end_time - load_start_time)))

    def __len__(self):
        return self.num_records


    



class InMemNoShuffleSentDataset(torch.utils.data.Dataset):
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
    def __init__(self,
                 data_folder: str, 
                 split: str,
                 use_clustered_data: bool = True
                 ) -> None:
        super(InMemNoShuffleSentDataset, self).__init__()
        self.data_folder = data_folder
        self.use_clustered_data = use_clustered_data

        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        self.data_buffer = []
        self.__load_data__()



    def __getitem__(self, i: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        return torch.LongTensor(self.data_buffer['sents'][i]), \
               torch.LongTensor([self.data_buffer['words_per_sentence'][i]]), \
               torch.LongTensor([self.data_buffer['labels'][i]])


    def __load_data__(self):
        print('[%s] Start loading data into memory' % nlp_format_dataloader.get_current_time())
        load_start_time = time.time()
        self.data_buffer = torch.load(os.path.join(self.data_folder, self.split + '_data.pth.tar'))
        self.num_records = len(self.data_buffer['labels'])

        if (self.use_clustered_data):
            zipped = zip(self.data_buffer['sents'], 
                         self.data_buffer['words_per_sentence'], self.data_buffer['labels'])
            sort_zipped = sorted(zipped, key=lambda x:(x[2]))
            result = zip(*sort_zipped)
            self.data_buffer['sents'], self.data_buffer['words_per_sentence'], self.data_buffer['labels'] = [list(x) for x in result]
        
        load_end_time = time.time()
        print("[%s] data_load_time = %.2fs" % (nlp_format_dataloader.get_current_time(), (load_end_time - load_start_time)))

    def __len__(self):
        return self.num_records