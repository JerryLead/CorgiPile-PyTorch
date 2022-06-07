"""Load tfrecord files into torch datasets."""

import typing
import numpy as np
import datetime
import random
import time

import torch.utils.data

from cifarformat.loader import cifar_format_dataloader
from cifarformat.in_mem_block_only import block_only_iterator_utils


class InMemBlockOnlyCifarDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 base_dir: str,
                 use_clustered_data: bool,
                 block_num: int,
                 buffer_size_ratio: float,
                 drop_last=False,
                 train=True, transform=None, target_transform=None,
                 data_name='cifar10'
                 ) -> None:
        super(InMemBlockOnlyCifarDataset, self).__init__()
        self.data_buffer = []
        self.base_dir = base_dir
        self.block_num = block_num
        self.data_name = data_name

        self.use_clustered_data = use_clustered_data
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.__load_data__()
        self.num_records = len(self.data_buffer)
        self.buffer_size = int(self.num_records * buffer_size_ratio)

        self.drop_last = drop_last
        self.block_index_list = self.split_index_to_blocks()
       
        # print("[Tip] Using datast.set_epoch(epoch) to shuffle the blocks before each epoch.")

    # shuffle the block_index_list before each epoch
   
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
        it = block_only_iterator_utils.shuffle_iterator(it, self.buffer_size)

        if self.transform or self.transform_target:
            it = map(self.transform_item, it)

        return it

    def transform_item(self, img_targe_item):
        img = img_targe_item[0]
        target = img_targe_item[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def in_mem_block_iterator(self):
        random.shuffle(self.block_index_list)
        # [(0, 71), (213, 284), (142, 213), (71, 142), (284, 355)]
        for block_index in self.block_index_list:
            for data_index in range(block_index[0], block_index[1]):
                yield self.data_buffer[data_index]


    def __load_data__(self):
        print('[%s] Start loading data into memory' % cifar_format_dataloader.get_current_time())

        load_start_time = time.time()
        it = cifar_format_dataloader.reader_iterator(self.base_dir, self.train, self.use_clustered_data, self.data_name)
        self.data_buffer.extend(it)
        load_end_time = time.time()
        print("[%s] data_load_time = %.2fs" % (cifar_format_dataloader.get_current_time(), (load_end_time - load_start_time)))
 

    def __len__(self):
        return self.num_records
