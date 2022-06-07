import typing
import numpy as np
import datetime
import random
import time
import math

import torch.utils.data
import torch.distributed as dist


from shuffleformat.corgipile import block_reader_tfrecord
from shuffleformat.corgipile import block_iterator_utils
from shuffleformat.corgipile import seq_reader_tfrecord


class CorgiPileTFRecordDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 data_path: str,
                 index_path: str,
                 block_num: int,
                 buffer_size_ratio: float,
                 description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 drop_last=False,
                 transform: typing.Callable[[dict], typing.Any] = None,
                 trans_after_buffered=True,
                 distributed=False
                 ) -> None:
        super(CorgiPileTFRecordDataset, self).__init__()

        self.data_path = data_path
        self.index_path = index_path
        # note that this block_num is an indication, the real number is more or less than it.
        self.block_num = block_num
        self.buffer_size_ratio = buffer_size_ratio

        self.description = description
        self.drop_last = drop_last
        self.transform = transform
        self.trans_after_buffered = trans_after_buffered

        self.epoch = 0


        # e.g., [0, 71, 142, 213, 284, 355, 426, 497, 568, 639, 710, 781, 852, 923, 994, 1065, 1136]
        self.block_index_list = self.split_index_to_blocks()
        # num_blocks = 17
        num_blocks = len(self.block_index_list) 
        
       
        if dist.is_available() and distributed:
            # e.g., world_size = 4
            world_size = dist.get_world_size()
            # rank = 0 or 1 or 2 or 3
            rank = dist.get_rank()

            print('world_size = ', world_size, ', rank = ', rank)
          
            # rank = 0/1/2/3, world_size = 4

            # start_index = 
            #   17 * 0 // 4 = 0
            #   17 * 1 // 4 = 4
            #   17 * 2 // 4 = 8
            #   17 * 3 // 4 = 12
            self.start_block_index = (num_blocks * rank) // world_size
            # end_index = 
            #   17 * 1 // 4 = 4
            #   17 * 2 // 4 = 8
            #   17 * 3 // 4 = 12
            #   17 * 4 // 4 = 17
            # block[0, 4), block[4, 8), block[8, 12), block[12, 17)
            self.end_block_index = (num_blocks * (rank + 1)) // world_size
            if self.end_block_index >= num_blocks:
                self.end_block_index = num_blocks
            # self.buffer_size = int(self.num_records / world_size * self.buffer_size_ratio)
            self.buffer_size = int(self.num_records * self.buffer_size_ratio)
            self.length = int(self.num_records / world_size)

        else:
            self.start_block_index = 0
            self.end_block_index = num_blocks
            self.buffer_size = int(self.num_records * self.buffer_size_ratio)
            self.length = self.num_records
       
       
        print("[Tip] Using datast.set_epoch(epoch) to shuffle the blocks before each epoch.")

    # shuffle the block_index_list before each epoch
    def set_epoch(self, epoch):
        # shuffle the blocks' indexes
        random.seed(epoch)
        random.shuffle(self.block_index_list)



    def split_index_to_blocks(self):
        # e.g., [0, 71, 142, 213, 284, 355, 426, 497, 568, 639]
        start_time = datetime.datetime.now()
        index = np.loadtxt(self.index_path, dtype=np.int64)[:, 0]
        end_time = datetime.datetime.now()
        str_time = '[index loading time] %dms' % ((end_time - start_time).seconds * 1000 + (end_time - start_time).microseconds / 1000)
        print(str_time)
  
        self.num_records = len(index)
        assert(self.block_num <= self.num_records)

        block_tuple_num = int(self.num_records / self.block_num)

        # re-compute the real block_num
        self.block_num = math.ceil(self.num_records / block_tuple_num)
    
        block_index_list = []

        for idx in range(0, self.block_num):
            start_index = block_tuple_num * idx
            end_index = block_tuple_num * (idx + 1)

            if end_index < self.num_records:
                end_byte = index[end_index]  
            else:
                end_byte = None
            if start_index < self.num_records:
                start_byte = index[start_index]
                # TODO: start_byte and end_type should be Long for very large dataset
                block_index_list.append((start_byte, end_byte))
            else:
                print('ERROR: start_index >= self.num_records!')
        
        print ('[param] num_records = %d, real_block_num = %d, block_tuple_num = %d' % (self.num_records, self.block_num, block_tuple_num) )
     
        return block_index_list



    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            shard = worker_info.id, worker_info.num_workers
        else:
            shard = None

      
        if self.trans_after_buffered:
            dataset_block_iter = block_reader_tfrecord.tfrecord_block_loader(
                                self.data_path, 
                                self.block_index_list,
                                self.start_block_index,
                                self.end_block_index,
                                self.description, transform=None, shard=shard)
        else:
            dataset_block_iter = block_reader_tfrecord.tfrecord_block_loader(
                                self.data_path, 
                                self.block_index_list,
                                self.start_block_index,
                                self.end_block_index,
                                self.description, self.transform, shard)

        if self.buffer_size > 0:
            it = block_iterator_utils.shuffle_iterator(dataset_block_iter, self.buffer_size)
        else:
            it = dataset_block_iter

        if self.trans_after_buffered:
            it = map(self.transform, it)

        return it


    def __len__(self):
        return self.length 

'''

For multi processes (nodes), 
    (1) specify `num_workers` and `distributed = True` and `partition = True`. Each process will load one part of the dataset.
        block_num = num_workers * world_size, start_block_index = block_num * rank // world_size, end_block_index = block_num * (rank + 1) // world_size
    (2) specify `num_workers` and `distributed = True` and `partition = False`. Each process will load the whole dataset.
        block_num = num_workers, start_block_index = 0, end_block_index = num_workers
For single process (node), 
    (1) Specify `num_workers` and `distributed = False`, each data loading thread will load one part of the dataset.
        block_num = num_workers, start_block_index = 0, end_block_index = num_workers
'''


class SeqTFRecordDataset(torch.utils.data.IterableDataset):
    
    # distributed = true: partition the dataset into differnt splits for each process
    # num_workers = the number of data loading thread for each process
    def __init__(self,
                 data_path: str,
                 index_path: str,
                 num_workers: int,
                 description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 drop_last=False,
                 transform: typing.Callable[[dict], typing.Any] = None,
                 trans_after_buffered=True,
                 distributed = False,
                 data_partition=True
                 ) -> None:
        super(SeqTFRecordDataset, self).__init__()

        self.data_path = data_path
        self.index_path = index_path
        self.description = description
        self.drop_last = drop_last
        self.transform = transform
        self.trans_after_buffered = trans_after_buffered

        self.epoch = 0

        if dist.is_available() and distributed:
            # e.g., world_size = 4
            world_size = dist.get_world_size()
            # rank = 0 or 1 or 2 or 3
            rank = dist.get_rank()

            print('world_size = ', world_size, ', rank = ', rank)
          
            if data_partition:
                # suppose rank = 0/1/2/3, world_size = 4, num_workers = 2
                self.block_num = world_size * num_workers
                # start_index = 
                #   8 * 0 // 4 = 0
                #   8 * 1 // 4 = 2
                #   8 * 2 // 4 = 4
                #   8 * 3 // 4 = 6
                self.block_index_list = self.split_index_to_blocks()
                self.start_block_index = (self.block_num * rank) // world_size
                # end_index = 
                #   8 * 1 // 4 = 2
                #   8 * 2 // 4 = 4
                #   8 * 3 // 4 = 6
                #   8 * 4 // 4 = 8
                # block[0, 2), block[2, 4), block[4, 6), block[6, 8)
                self.end_block_index = (self.block_num * (rank + 1)) // world_size
                if self.end_block_index >= self.block_num:
                    self.end_block_index = self.block_num

                
                self.length = int(self.num_records / world_size)
                
                if self.num_records % world_size != 0:
                    print('[Warning] the number of records', self.num_records, 'cannot be euqally divided by the world_size', world_size, '!')
            else:
                self.block_num = num_workers
                self.start_block_index = 0
                self.end_block_index = self.block_num
                self.block_index_list = self.split_index_to_blocks()
                self.length = self.num_records

        else:
            self.block_num = num_workers
            self.start_block_index = 0
            self.end_block_index = self.block_num

            self.block_index_list = self.split_index_to_blocks()
            self.length = self.num_records

        assert(len(self.block_index_list) == self.block_num)


    def split_index_to_blocks(self):
        # e.g., [0, 71, 142, 213, 284, 355, 426, 497, 568, 639]
        start_time = datetime.datetime.now()
        index = np.loadtxt(self.index_path, dtype=np.int64)[:, 0]
        end_time = datetime.datetime.now()
        str_time = '[index loading time] %dms' % ((end_time - start_time).seconds * 1000 + (end_time - start_time).microseconds / 1000)
        print(str_time)
  
        self.num_records = len(index)
        assert(self.block_num <= self.num_records)

        block_tuple_num = int(self.num_records / self.block_num)

        # re-compute the real block_num
        self.block_num = math.ceil(self.num_records / block_tuple_num)
    
        block_index_list = []

        for idx in range(0, self.block_num):
            start_index = block_tuple_num * idx
            end_index = block_tuple_num * (idx + 1)

            if end_index < self.num_records:
                end_byte = index[end_index]  
            else:
                end_byte = None
            if start_index < self.num_records:
                start_byte = index[start_index]
                # TODO: start_byte and end_type should be Long for very large dataset
                block_index_list.append((start_byte, end_byte))
            else:
                print('ERROR: start_index >= self.num_records!')
        
        print ('[param] num_records = %d, real_block_num = %d, block_tuple_num = %d' % (self.num_records, self.block_num, block_tuple_num) )
     
        return block_index_list



    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            shard = worker_info.id, worker_info.num_workers
        else:
            shard = None

      
        if self.trans_after_buffered:
            it = block_reader_tfrecord.tfrecord_block_loader(
                                self.data_path, 
                                self.block_index_list,
                                self.start_block_index,
                                self.end_block_index,
                                self.description, transform=None, shard=shard)
        else:
            it = block_reader_tfrecord.tfrecord_block_loader(
                                self.data_path, 
                                self.block_index_list,
                                self.start_block_index,
                                self.end_block_index,
                                self.description, self.transform, shard)


        if self.trans_after_buffered:
            it = map(self.transform, it)

        return it


    def __len__(self):
        return self.length 





class RandomAccessTFRecordDataset(torch.utils.data.IterableDataset):
    
    def __init__(self,
                 data_path: str,
                 index_path: str,
                 description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 drop_last=False,
                 transform: typing.Callable[[dict], typing.Any] = None,
                 trans_after_buffered=True,
                 distributed=False
                 ) -> None:
        super(RandomAccessTFRecordDataset, self).__init__()

        self.data_path = data_path
        self.index_path = index_path
        self.description = description
        self.drop_last = drop_last
        self.transform = transform
        self.trans_after_buffered = trans_after_buffered

        self.epoch = 0

         # e.g., [0, 71, 142, 213, 284, 355, 426, 497, 568, 639, 710, 781, 852, 923, 994, 1065, 1136]
        self.block_index_list = self.split_index_to_blocks()
        # num_blocks = 17
        num_blocks = len(self.block_index_list) 
        
       
        if dist.is_available() and distributed:
            # e.g., world_size = 4
            world_size = dist.get_world_size()
            # rank = 0 or 1 or 2 or 3
            rank = dist.get_rank()

            print('world_size = ', world_size, ', rank = ', rank)
          
            # rank = 0/1/2/3, world_size = 4

           
            # start_index = 
            #   17 * 0 // 4 = 0
            #   17 * 1 // 4 = 4
            #   17 * 2 // 4 = 8
            #   17 * 3 // 4 = 12
            self.start_block_index = (num_blocks * rank) // world_size
            # end_index = 
            #   17 * 1 // 4 = 4
            #   17 * 2 // 4 = 8
            #   17 * 3 // 4 = 12
            #   17 * 4 // 4 = 17
            # block[0, 4), block[4, 8), block[8, 12), block[12, 17)
            self.end_block_index = (num_blocks * (rank + 1)) // world_size
            if self.end_block_index >= num_blocks:
                self.end_block_index = num_blocks
            
            self.length = int(self.num_records / world_size)

        else:
            self.start_block_index = 0
            self.end_block_index = num_blocks
           
            self.length = self.num_records
       

        print("[Tip] Using datast.set_epoch(epoch) to shuffle the blocks before each epoch.")

    # shuffle the block_index_list before each epoch
    def set_epoch(self, epoch):
        # shuffle the blocks' indexes
        random.seed(epoch)
        random.shuffle(self.block_index_list)
        # print(self.block_index_list[0])
        # print(self.block_index_list[1])
        # print(self.block_index_list[2])
        # print(self.block_index_list[3])
        # print(self.block_index_list[4])


    def split_index_to_blocks(self):
        # e.g., [0, 71, 142, 213, 284, 355, 426, 497, 568, 639]
        start_time = datetime.datetime.now()
        index = np.loadtxt(self.index_path, dtype=np.int64)[:, 0]
        end_time = datetime.datetime.now()
        str_time = '[index loading time] %dms' % ((end_time - start_time).seconds * 1000 + (end_time - start_time).microseconds / 1000)
        print(str_time)
        #print(index)
        self.num_records = len(index)

        self.block_num = self.num_records

        block_tuple_num = 1
    
        print ('[param] block_num = %d, block_tuple_num = %d' % (self.block_num, block_tuple_num) )
        block_index_list = []

        for idx in range(0, self.block_num):
            start_index = block_tuple_num * idx
            end_index = block_tuple_num * (idx + 1)

            if end_index < self.num_records:
                end_byte = index[end_index]  
            else:
                end_byte = None
            if start_index < self.num_records:
                start_byte = index[start_index]
                # TODO: start_byte and end_type should be Long for very large dataset
                block_index_list.append((start_byte, end_byte))
        
        #print(block_index_list)
        return block_index_list



    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            shard = worker_info.id, worker_info.num_workers
        else:
            shard = None

      
        if self.trans_after_buffered:
            it = block_reader_tfrecord.tfrecord_block_loader(
                                self.data_path, 
                                self.block_index_list,
                                self.start_block_index,
                                self.end_block_index,
                                self.description, transform=None, shard=shard)
        else:
            it = block_reader_tfrecord.tfrecord_block_loader(
                                self.data_path, 
                                self.block_index_list,
                                self.start_block_index,
                                self.end_block_index,
                                self.description, self.transform, shard)
                                
        if self.trans_after_buffered:
            it = map(self.transform, it)

        return it


    def __len__(self):
        return self.length 



