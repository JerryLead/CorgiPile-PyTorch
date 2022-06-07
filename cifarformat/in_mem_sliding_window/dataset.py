import numpy as np
import warnings
import random
import time

import torch.utils.data

from cifarformat.loader import cifar_format_dataloader

class InMemSlidingWindowCifarDataset(torch.utils.data.IterableDataset):

    def __init__(self, 
                base_dir: str,
                use_clustered_data: bool,
                sliding_window_size_ratio: float, 
                train=True, transform=None, target_transform=None,
                data_name='cifar10'
                ) -> None:
        
        super(InMemSlidingWindowCifarDataset, self).__init__()

        self.base_dir = base_dir
        self.data_name = data_name
        self.train = train
        self.use_clustered_data = use_clustered_data
        self.transform = transform
        self.target_transform = target_transform
        self.data_buffer = []
        self.sliding_window_size_ratio = sliding_window_size_ratio

        self.__load_data__()
        self.num_records = len(self.data_buffer)
        self.shuffle_window_size = int(self.sliding_window_size_ratio * self.num_records)

    def __load_data__(self):
        print('[%s] Start loading data into memory' % cifar_format_dataloader.get_current_time())

        load_start_time = time.time()
        it = cifar_format_dataloader.reader_iterator(self.base_dir, self.train, self.use_clustered_data, self.data_name)
        self.data_buffer.extend(it)
        load_end_time = time.time()
        print("[%s] data_load_time = %.2fs" % (cifar_format_dataloader.get_current_time(), (load_end_time - load_start_time)))
      
    def transform_item(self, img_targe_item):
        img = img_targe_item[0]
        target = img_targe_item[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



    def __iter__(self):
        it = self.in_mem_sliding_window_iterator()

        if self.transform or self.transform_target:
            it = map(self.transform_item, it)

        return it

    def __len__(self):
        return self.num_records

    def in_mem_sliding_window_iterator(self):
        buffer = []
        try:
            for i in range(0, self.shuffle_window_size):
                buffer.append(self.data_buffer[i])
        except StopIteration:
            warnings.warn("Number of elements in the iterator is less than the "
                      f"queue size (N={self.shuffle_window_size}).")
        
        buffer_size = self.shuffle_window_size
        
        for i in range(buffer_size, self.num_records):
            index = random.randint(0, buffer_size - 1)
            
            item = buffer[index]
            buffer[index] = self.data_buffer[i]
            yield item
            
        random.shuffle(buffer)
        for i in range(0, len(buffer)):
            yield buffer[i]

 
