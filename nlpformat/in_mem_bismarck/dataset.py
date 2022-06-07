"""Load tfrecord files into torch datasets."""
import numpy as np
import os
import pickle
import time

import torch.utils.data

from nlpformat.loader import nlp_format_dataloader
from nlpformat.in_mem_bismarck import iterator_utils


class InMemBismarckDocDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 data_folder: str, 
                 split: str,
                 use_clustered_data: bool,
                 bismarck_buffer_size_ratio: float,
                 select_ratio_from_old_buffer: float,
                 old_buffer_checkpoint_dir: str
                 ) -> None:
        super(InMemBismarckDocDataset, self).__init__()
        self.data_folder = data_folder
        self.use_clustered_data = use_clustered_data

        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        self.data_buffer = []
        self.bismarck_buffer_size_ratio = bismarck_buffer_size_ratio
        self.select_ratio_from_old_buffer = select_ratio_from_old_buffer

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
        
        it = map(self.transform_doc, it)
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

    def transform_doc(self, doc_item):
        return torch.LongTensor(doc_item[0]), \
               torch.LongTensor([doc_item[1]]), \
               torch.LongTensor(doc_item[2]), \
               torch.LongTensor([doc_item[3]])
    

    def __len__(self):
        return self.total_records_num
    
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
  

        
    def in_mem_bismarck_loader(self):
        for data_index in range(0, self.total_records_num):
            yield self.data_buffer[data_index] 



class InMemBismarckSentDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 data_folder: str, 
                 split: str,
                 use_clustered_data: bool,
                 bismarck_buffer_size_ratio: float,
                 select_ratio_from_old_buffer: float,
                 old_buffer_checkpoint_dir: str
                 ) -> None:
        super(InMemBismarckSentDataset, self).__init__()
        self.data_folder = data_folder
        self.use_clustered_data = use_clustered_data

        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        self.data_buffer = []
        self.bismarck_buffer_size_ratio = bismarck_buffer_size_ratio
        self.select_ratio_from_old_buffer = select_ratio_from_old_buffer

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
        
        it = map(self.transform_sent, it)
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

    def transform_sent(self, doc_item):
        return torch.LongTensor(doc_item[0]), \
               torch.LongTensor([doc_item[1]]), \
               torch.LongTensor([doc_item[2]])
    
    

    def __len__(self):
        return self.total_records_num
    
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
  

        
    def in_mem_bismarck_loader(self):
        for data_index in range(0, self.total_records_num):
            yield self.data_buffer[data_index] 