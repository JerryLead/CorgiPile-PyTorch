"""Reader utils"""
import os
import time
import numpy as np

import torchvision
import random

class MY_CIFAR10(torchvision.datasets.CIFAR10):
    
    def __init__(self, root, train=True, use_clustered_data=True):

        super(MY_CIFAR10, self).__init__(root, train=train, transform=None,
                                      target_transform=None, download=False)
  
        if (use_clustered_data):
            # sort the in-memory data of the original CIFAR10
            zipped = zip(self.data, self.targets)
            sort_zipped = sorted(zipped, key=lambda x:(x[1]))

            result = zip(*sort_zipped)
            self.data, self.targets = [list(x) for x in result]


class MY_CIFAR100(torchvision.datasets.CIFAR100):
    
    def __init__(self, root, train=True, use_clustered_data=True):

        super(MY_CIFAR100, self).__init__(root, train=train, transform=None,
                                      target_transform=None, download=False)
  
        if (use_clustered_data):
            # sort the in-memory data of the original CIFAR10
            zipped = zip(self.data, self.targets)
            sort_zipped = sorted(zipped, key=lambda x:(x[1]))

            result = zip(*sort_zipped)
            self.data, self.targets = [list(x) for x in result]

def reader_iterator(base_dir, train, use_clustered_data=True, data_name='cifar10'):
    if (data_name == 'cifar10'):
        trainset = MY_CIFAR10(root=base_dir, train=train, use_clustered_data=use_clustered_data)
    elif (data_name == 'cifar100'):
        trainset = MY_CIFAR100(root=base_dir, train=train, use_clustered_data=use_clustered_data)
    for i in range(0, len(trainset)):
        yield trainset[i]


if __name__ == '__main__' :
    base_dir = '/mnt/ds3lab-scratch/xuliji/data/'
    data_buffer = []
    data_buffer.extend(reader_iterator(base_dir, True, True, 'cifar100'))
    random.shuffle(data_buffer)
    print('len = ', len(data_buffer))
    # for i in range(0, len(data_buffer)):
    #     print(data_buffer[i][1], end=" ")
    print(data_buffer[0])


def get_current_time() :
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
