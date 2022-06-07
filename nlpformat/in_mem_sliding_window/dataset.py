import numpy as np
import warnings
import random
import time
import os

import torch.utils.data

from nlpformat.loader import nlp_format_dataloader

class InMemSlidingWindowDocDataset(torch.utils.data.IterableDataset):

    def __init__(self, 
                data_folder: str, 
                split: str,
                use_clustered_data: bool,
                sliding_window_size_ratio: float
                ) -> None:
        
        super(InMemSlidingWindowDocDataset, self).__init__()
        self.data_folder = data_folder
        self.use_clustered_data = use_clustered_data

        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        self.data_buffer = []
      
        self.sliding_window_size_ratio = sliding_window_size_ratio

        self.__load_data__()
        
        self.shuffle_window_size = int(self.sliding_window_size_ratio * self.num_records)

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
  


    def __iter__(self):
        it = self.in_mem_sliding_window_iterator()
        it = map(self.transform_doc, it)
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

 
    def transform_doc(self, doc_item):
        return torch.LongTensor(doc_item[0]), \
               torch.LongTensor([doc_item[1]]), \
               torch.LongTensor(doc_item[2]), \
               torch.LongTensor([doc_item[3]])
    


class InMemSlidingWindowSentDataset(torch.utils.data.IterableDataset):

    def __init__(self, 
                data_folder: str, 
                split: str,
                use_clustered_data: bool,
                sliding_window_size_ratio: float
                ) -> None:
        
        super(InMemSlidingWindowSentDataset, self).__init__()
        self.data_folder = data_folder
        self.use_clustered_data = use_clustered_data

        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        self.data_buffer = []
      
        self.sliding_window_size_ratio = sliding_window_size_ratio

        self.__load_data__()
        
        self.shuffle_window_size = int(self.sliding_window_size_ratio * self.num_records)

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
  


    def __iter__(self):
        it = self.in_mem_sliding_window_iterator()
        it = map(self.transform_sent, it)
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

 
     
    def transform_sent(self, doc_item):
        return torch.LongTensor(doc_item[0]), \
               torch.LongTensor([doc_item[1]]), \
               torch.LongTensor([doc_item[2]])
    
    