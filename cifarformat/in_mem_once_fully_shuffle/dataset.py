import numpy as np
import random
import time

import torch.utils.data

from cifarformat.loader import cifar_format_dataloader


class InMemOnceFullyShuffleCifarDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 base_dir: str,
                 use_clustered_data: bool,
                 train=True, transform=None, target_transform=None,
                 data_name='cifar10'
                 ) -> None:
        super(InMemOnceFullyShuffleCifarDataset, self).__init__()
        self.base_dir = base_dir
        self.data_name = data_name

        self.use_clustered_data = use_clustered_data
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.data_buffer = []
        self.__load_data__()
    
        self.buffer_len = len(self.data_buffer)

    def transform_item(self, img_targe_item):
        img = img_targe_item[0]
        target = img_targe_item[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __getitem__(self, idx):
        return self.transform_item(self.data_buffer[idx])

    def __load_data__(self):
        print('[%s] Start loading data into memory' % cifar_format_dataloader.get_current_time())

        load_start_time = time.time()
        it = cifar_format_dataloader.reader_iterator(self.base_dir, self.train, self.use_clustered_data, self.data_name)
        self.data_buffer.extend(it)
        load_end_time = time.time()
        print("[%s] data_load_time = %.2fs" % (cifar_format_dataloader.get_current_time(), (load_end_time - load_start_time)))
        
        random.shuffle(self.data_buffer)
        #print(self.data_buffer[0])
        sort_end_time = time.time()
        print("[%s] data_sort_time = %.2fs" % (cifar_format_dataloader.get_current_time(), (sort_end_time - load_end_time)))
        
    def __len__(self):
        return self.buffer_len

   


    