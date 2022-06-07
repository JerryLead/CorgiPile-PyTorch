import numpy as np
import random
import time
import os

import torch.utils.data

from nlpformat.loader import nlp_format_dataloader


class InMemOnceFullyShuffleDocDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 data_folder: str, split: str,
                 use_clustered_data: bool
                 ) -> None:
        super(InMemOnceFullyShuffleDocDataset, self).__init__()
        self.data_folder = data_folder
        self.use_clustered_data = use_clustered_data

        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        self.data_buffer = []
        self.__load_data__()
        self.num_records = len(self.data_buffer['labels'])



    def __getitem__(self, idx):
        return torch.LongTensor(self.data_buffer['docs'][idx]), \
               torch.LongTensor([self.data_buffer['sentences_per_document'][idx]]), \
               torch.LongTensor(self.data_buffer['words_per_sentence'][idx]), \
               torch.LongTensor([self.data_buffer['labels'][idx]])

    def __load_data__(self):
        print('[%s] Start loading data into memory' % nlp_format_dataloader.get_current_time())
        load_start_time = time.time()
        self.data_buffer = torch.load(os.path.join(self.data_folder, self.split + '_data.pth.tar'))
    
        
        zipped = zip(self.data_buffer['docs'], self.data_buffer['sentences_per_document'], 
                         self.data_buffer['words_per_sentence'], self.data_buffer['labels'])
        temp = list(zipped)
        random.shuffle(temp)
        self.data_buffer['docs'], self.data_buffer['sentences_per_document'], self.data_buffer['words_per_sentence'], self.data_buffer['labels'] = zip(*temp)
        
        
        load_end_time = time.time()
        print("[%s] data_load_time = %.2fs" % (nlp_format_dataloader.get_current_time(), (load_end_time - load_start_time)))

    def __len__(self):
        return self.num_records



    
   
class InMemOnceFullyShuffleSentDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 data_folder: str, split: str,
                 use_clustered_data: bool
                 ) -> None:
        super(InMemOnceFullyShuffleSentDataset, self).__init__()
        self.data_folder = data_folder
        self.use_clustered_data = use_clustered_data

        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        self.data_buffer = []
        self.__load_data__()
        self.num_records = len(self.data_buffer['labels'])



    def __getitem__(self, i):
        return torch.LongTensor(self.data_buffer['sents'][i]), \
               torch.LongTensor([self.data_buffer['words_per_sentence'][i]]), \
               torch.LongTensor([self.data_buffer['labels'][i]])

    def __load_data__(self):
        print('[%s] Start loading data into memory' % nlp_format_dataloader.get_current_time())
        load_start_time = time.time()
        self.data_buffer = torch.load(os.path.join(self.data_folder, self.split + '_data.pth.tar'))
    
        
        zipped = zip(self.data_buffer['sents'], 
                         self.data_buffer['words_per_sentence'], self.data_buffer['labels'])
        temp = list(zipped)
        random.shuffle(temp)
        self.data_buffer['sents'], self.data_buffer['words_per_sentence'], self.data_buffer['labels'] = zip(*temp)
        
        
        load_end_time = time.time()
        print("[%s] data_load_time = %.2fs" % (nlp_format_dataloader.get_current_time(), (load_end_time - load_start_time)))

    def __len__(self):
        return self.num_records


    