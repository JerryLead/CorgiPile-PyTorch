"""Load tfrecord files into torch datasets."""
import numpy as np
import os
import pickle
import time

import torch.utils.data

from cifarformat.loader import cifar_format_dataloader
from cifarformat.in_mem_bismarck import iterator_utils


class InMemBismarckCifarDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 base_dir: str,
                 use_clustered_data: bool,
                 bismarck_buffer_size_ratio: float,
                 select_ratio_from_old_buffer: float,
                 old_buffer_checkpoint_dir: str,
                 train=True, transform=None, target_transform=None,
                 data_name='cifar10'
                 ) -> None:
        super(InMemBismarckCifarDataset, self).__init__()
        self.base_dir = base_dir
        self.data_name = data_name

        self.bismarck_buffer_size_ratio = bismarck_buffer_size_ratio
        self.select_ratio_from_old_buffer = select_ratio_from_old_buffer

        self.use_clustered_data = use_clustered_data
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.data_buffer = []
        self.__load_data__()
    
        self.total_records_num = len(self.data_buffer)
        #self.num_records_from_old_buffer = self.total_records_num * self.select_ratio_on_old_buffer
        
        self.io_buffer_size = int(self.total_records_num * bismarck_buffer_size_ratio)
        assert(self.io_buffer_size > 0)

        # buffer in the memory worker in Bismarck's SIGMOD paper
        self.old_buffer = []
        self.io_buffer = []

        self.old_buffer_checkpoint_dir = old_buffer_checkpoint_dir
        if self.old_buffer_checkpoint_dir:
            self.delete_old_buffer()
       

    def __iter__(self):
        it = self.in_mem_bismarck_loader()
        id = 1
        self.load_old_buffer(id)
        file_writer = self.old_buffer_writer(id)

        it = iterator_utils.shuffle_iterator(it, 
                            self.io_buffer, self.io_buffer_size,
                            self.total_records_num,
                            self.old_buffer, 
                            self.select_ratio_from_old_buffer,
                            file_writer)
        
        if self.transform or self.target_transform:
            it = map(self.transform_item, it)
        return it
    
    def load_old_buffer(self, id):  
        old_buffer_file = os.path.join(self.old_buffer_checkpoint_dir, '_old_buffer_' + str(id) +'.dat')
        if os.path.exists(old_buffer_file):
            file = open(old_buffer_file, 'rb')
            self.old_buffer = pickle.load(file)
            #self.io_buffer = pickle.load(file)
            file.close()
        
    def old_buffer_writer(self, id):
        if not os.path.exists(self.old_buffer_checkpoint_dir):
            os.makedirs(self.old_buffer_checkpoint_dir)
        old_buffer_file = os.path.join(self.old_buffer_checkpoint_dir, '_old_buffer_' + str(id) +'.dat')
        
        file_writer = open(old_buffer_file, 'wb')
        return file_writer


    def delete_old_buffer(self):
        if os.path.exists(self.old_buffer_checkpoint_dir):
            files = os.listdir(self.old_buffer_checkpoint_dir)
            for file in files:
                if file.startswith('_old_buffer_') and file.endswith('.dat'):
                    os.remove(os.path.join(self.old_buffer_checkpoint_dir, file))

    def transform_item(self, img_targe_item):
        img = img_targe_item[0]
        target = img_targe_item[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.total_records_num
    
    def __load_data__(self):
        print('[%s] Start loading data into memory' % cifar_format_dataloader.get_current_time())

        load_start_time = time.time()
        it = cifar_format_dataloader.reader_iterator(self.base_dir, self.train, self.use_clustered_data, self.data_name)
        self.data_buffer.extend(it)
        load_end_time = time.time()
        print("[%s] data_load_time = %.2fs" % (cifar_format_dataloader.get_current_time(), (load_end_time - load_start_time)))
        
    def in_mem_bismarck_loader(self):
        for data_index in range(0, self.total_records_num):
            yield self.data_buffer[data_index] 