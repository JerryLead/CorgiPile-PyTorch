"""Load tfrecord files into torch datasets."""

import typing
import numpy as np
import datetime
import random
import time
import os

import torch.utils.data

from nlpformat.loader import nlp_format_dataloader
from nlpformat.in_mem_block import block_iterator_utils


class InMemBlockDocDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 data_folder: str, 
                 split: str,
                 use_clustered_data: bool,
                 block_num: int,
                 buffer_size_ratio: float,
                 drop_last=False
                 ) -> None:
        super(InMemBlockDocDataset, self).__init__()
        self.data_folder = data_folder
        self.use_clustered_data = use_clustered_data

        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        self.data_buffer = []
           
        self.block_num = block_num

        self.__load_data__()
       
        self.buffer_size = int(self.num_records * buffer_size_ratio)

        self.drop_last = drop_last
        self.block_index_list = self.split_index_to_blocks()
       
        # print("[Tip] Using datast.set_epoch(epoch) to shuffle the blocks before each epoch.")

    def __load_data__(self):
        print('[%s] Start loading data into memory' % nlp_format_dataloader.get_current_time())
        load_start_time = time.time()
        self.data = torch.load(os.path.join(self.data_folder, self.split + '_data.pth.tar'))
        self.num_records = len(self.data['labels'])


        if (self.use_clustered_data):
            zipped = zip(self.data['docs'], self.data['sentences_per_document'], 
                         self.data['words_per_sentence'], self.data['labels'])
            sort_zipped = sorted(zipped, key=lambda x:(x[3]))
            self.data['docs'], self.data['sentences_per_document'], self.data['words_per_sentence'], self.data['labels'] = zip(*sort_zipped)

        
        for i in range(0, self.num_records):
            self.data_buffer.append((self.data['docs'][i], self.data['sentences_per_document'][i], self.data['words_per_sentence'][i], self.data['labels'][i]))

   
        load_end_time = time.time()
        print("[%s] data_load_time = %.2fs" % (nlp_format_dataloader.get_current_time(), (load_end_time - load_start_time)))
  

    def split_index_to_blocks(self):
        # e.g., [0, 71, 142, 213, 284, 355, 426, 497, 568, 639]
        assert(self.block_num < self.num_records)
        block_tuple_num = int(self.num_records / self.block_num)
        
        print ('[param] block_num = %d, block_tuple_num = %d' % (self.block_num, block_tuple_num) )
       
        # store index_id instead of file_offset
        block_index_list = []

        for idx in range(0, self.block_num):
            start_index = block_tuple_num * idx
            end_index = block_tuple_num * (idx + 1)

            if end_index > self.num_records:
                end_index = self.num_records
            
            if start_index < self.num_records:
                block_index_list.append((start_index, end_index))
        
        #print(block_index_list)
        return block_index_list

    def __iter__(self):
        it = self.in_mem_block_iterator()
        it = block_iterator_utils.shuffle_iterator(it, self.buffer_size)

        it = map(self.transform_doc_item, it)

        return it

    
    def in_mem_block_iterator(self):
        random.shuffle(self.block_index_list)
        # [(0, 71), (213, 284), (142, 213), (71, 142), (284, 355)]
        for block_index in self.block_index_list:
            for data_index in range(block_index[0], block_index[1]):
                yield self.data_buffer[data_index]


    def __len__(self):
        return self.num_records

    def transform_doc_item(self, doc_item):
        return torch.LongTensor(doc_item[0]), \
               torch.LongTensor([doc_item[1]]), \
               torch.LongTensor(doc_item[2]), \
               torch.LongTensor([doc_item[3]])
    


class InMemBlockSentDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 data_folder: str, 
                 split: str,
                 use_clustered_data: bool,
                 block_num: int,
                 buffer_size_ratio: float,
                 drop_last=False
                 ) -> None:
        super(InMemBlockSentDataset, self).__init__()
        self.data_folder = data_folder
        self.use_clustered_data = use_clustered_data

        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        self.data_buffer = []
           
        self.block_num = block_num

        self.__load_data__()
       
        self.buffer_size = int(self.num_records * buffer_size_ratio)

        self.drop_last = drop_last
        self.block_index_list = self.split_index_to_blocks()
       
        # print("[Tip] Using datast.set_epoch(epoch) to shuffle the blocks before each epoch.")

    def __load_data__(self):
        print('[%s] Start loading data into memory' % nlp_format_dataloader.get_current_time())
        load_start_time = time.time()
        self.data = torch.load(os.path.join(self.data_folder, self.split + '_data.pth.tar'))
        self.num_records = len(self.data['labels'])


        if (self.use_clustered_data):
            zipped = zip(self.data['sents'], 
                         self.data['words_per_sentence'], self.data['labels'])
            sort_zipped = sorted(zipped, key=lambda x:(x[2]))
            self.data['sents'], self.data['words_per_sentence'], self.data['labels'] = zip(*sort_zipped)

        
        for i in range(0, self.num_records):
            self.data_buffer.append((self.data['sents'][i], self.data['words_per_sentence'][i], self.data['labels'][i]))

   
        load_end_time = time.time()
        print("[%s] data_load_time = %.2fs" % (nlp_format_dataloader.get_current_time(), (load_end_time - load_start_time)))
  

    def split_index_to_blocks(self):
        # e.g., [0, 71, 142, 213, 284, 355, 426, 497, 568, 639]
        assert(self.block_num < self.num_records)
        block_tuple_num = int(self.num_records / self.block_num)
        
        print ('[param] block_num = %d, block_tuple_num = %d' % (self.block_num, block_tuple_num) )
       
        # store index_id instead of file_offset
        block_index_list = []

        for idx in range(0, self.block_num):
            start_index = block_tuple_num * idx
            end_index = block_tuple_num * (idx + 1)

            if end_index > self.num_records:
                end_index = self.num_records
            
            if start_index < self.num_records:
                block_index_list.append((start_index, end_index))
        
        #print(block_index_list)
        return block_index_list

    def __iter__(self):
        it = self.in_mem_block_iterator()
        it = block_iterator_utils.shuffle_iterator(it, self.buffer_size)

        it = map(self.transform_sent_item, it)

        return it

    
    def in_mem_block_iterator(self):
        random.shuffle(self.block_index_list)
        # [(0, 71), (213, 284), (142, 213), (71, 142), (284, 355)]
        for block_index in self.block_index_list:
            for data_index in range(block_index[0], block_index[1]):
                yield self.data_buffer[data_index]


    def __len__(self):
        return self.num_records

    def transform_sent_item(self, doc_item):
        return torch.LongTensor(doc_item[0]), \
               torch.LongTensor([doc_item[1]]), \
               torch.LongTensor([doc_item[2]])
    