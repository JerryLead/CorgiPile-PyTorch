# CorgiPile-PyTorch
The implementation of CorgiPile in PyTorch.

# Introduction

CorgiPile is a novel two-level hierarchical data shuffle mechanism for efficient SGD computation in both deep learning and in-DB ML systems. The main idea is to first sample and shuffle the data at block level, and then shuffle data at tuple level within the sampled data blocks, i.e., first sampling data blocks, then merging the sampled blocks in a buffer, and finally shuffling the tuples in the buffer for SGD. Compared with existing mechanisms, CorgiPile can avoid the full shuffle while maintaining comparable convergence rate as if a full shuffle were performed.

We have implemented CorgiPile inside PyTorch, by designing new parallel/distributed shuffle operators as well as a new DataSet API. Extensive experimental results show that our CorgiPile in PyTorch can achieve comparable convergence rate with the full-shuffle based SGD, and 1.5X faster than PyTorch PyTorch with full data shuffle on ImageNet dataset.


# How to run CorgiPile-PyTorch? 

The following steps describe how to use CorgiPile-PyTorch to train deep learning models on ImageNet, cifar-10, and yelp-review-full datasets.

## Install Python packages
Install necessary Python packages such as `torch`, `torchvision`, `torchtext`, `numpy`, `nltk` and `pandas`, through `pip` or `conda`.

## Download the datasets

Download the following datasets. Decompress them into a directory like 'corgipile_data'.
1. [ImageNet dataset](https://www.image-net.org/download.php)
2. [cifar-10 dataset](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
3. [yelp-review-full dataset](https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0)


## Data preprocessing

### 1. ImageNet data preprocessing

ImageNet dataset contains 1.2 million raw images. It is usually not feasible to load all the images into memory, and it is slow to (randomly) access each image on block-based parallel/distributed file systems such as [Lustre](https://www.lustre.org/) and HDFS. Therefore, in a parallel/distributed cluster, we usually transform the ImageNet dataset into binary format like TFRecord used by TensorFlow (https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset) and PyTorch (https://github.com/vahidk/tfrecord).

Run [images_raw_to_tfrecord.py](imagenet_dl_bench/pre_process/images_raw_to_tfrecord.py) to transform the ImageNet dataset.
The code contains annotations for some important configurations. The transformed data contains four files, including `train/train_clustered.tfrecord`, `train/train_clustered.index`, `val/val_clustered.tfrecord`, and `val/val_clustered.index`.


### 2. cifar-10 data preprocessing

We do not need to preprocess cifar-10, which can be loaded into memory.


### 2. yelp-review-full data preprocessing

We train HAN and TextCNN models on this NLP dataset based on the code from https://github.com/Renovamen/Text-Classification. To perform these models, we need to transform this dataset into docs and sentences in advance. 
Feel free to use the [nlp_dl_bench/preprocess.py](nlp_dl_bench/preprocess.py) to perform this transformation. You can also refer to this [guide](https://github.com/Renovamen/Text-Classification) for details. 



# Train deep learning models on these datasets


## Train ResNet50 on ImageNet dataset

Use [imagenet_dl_bench/normal_node/imagenet_corgipile_raw_train.py](imagenet_dl_bench/normal_node/imagenet_corgipile_raw_train.py) to train the ResNet50 model on the ImageNet dataset. This code is similar to that of the official [PyTorch-ImageNet code](https://github.com/pytorch/examples/blob/main/imagenet/main.py). The main difference is that our code contains multiple data shuffling modes such as 'no_shuffle', 'corgipile_shuffle (block)', and 'once_shuffle'. This code can run on multiple GPUs.

## Train VGG19 and ResNet18 on cifar-10 dataset

Use [cifar_dl_bench/cifar_dl_bench_train.py](cifar_dl_bench/cifar_dl_bench_train.py) to train the VGG19 or ResNet18 models on the cifar-10 dataset. This code runs on 1 GPU.

## Train HAN and TextCNN on yelp-review-full dataset

Use [nlp_dl_bench/nlp_dl_bench_train.py](nlp_dl_bench/nlp_dl_bench_train.py) to train the HAN or TextCNN models on the  yelp-review-full dataset. This code runs on 1 GPU.


## Training logs

The training logs are automatically stored in the specified log files and there is a demo:

```
[params] use_clustered_data = True
[params] use_train_accuracy = True
[params] use_sgd = True
[params] model_name = textcnn
[params] batch_size = 128
[params] iter_num = 3
[params] learning_rate = 0.001
[params] num_workers = 1
[params] data_name = yelp_review_full
[params] lr_decay = 0.95
[params] block_num = 650
[params] buffer_size_ratio = 0.1
[params] sliding_window_size_ratio = 0.1
[params] bismarck_buffer_size_ratio = 0.1
[params] shuffle_mode = once_shuffle
[params] log_file = XXX/CorgiPile-PyTorch/train_log_nlp_sgd/yelp_review_full/textcnn/sgd-bs128/once_shuffle/once_shuffle_yelp_review_full_lr0.001_2022-06-08-16-21-42.txt
[Computed] train_tuple_num = 650000
[2022-06-08 16:22:08] Start iteration
[2022-06-08 16:25:36] [Iter  1] Loss = 7336.05, acc = 35.96, exec_t = 207.69s, grad_t = 151.62s, loss_t = 56.07s
[2022-06-08 16:28:56] [Iter  2] Loss = 6767.29, acc = 41.84, exec_t = 199.47s, grad_t = 143.40s, loss_t = 56.07s
[2022-06-08 16:32:15] [Iter  3] Loss = 6420.20, acc = 45.06, exec_t = 199.79s, grad_t = 143.55s, loss_t = 56.24s
[2022-06-08 16:32:15] [Finish] avg_exec_t = 202.32s, avg_grad_t = 146.19s, avg_loss_t = 56.13s
[2022-06-08 16:32:15] [-first] avg_exec_t = 199.63s, avg_grad_t = 143.47s, avg_loss_t = 56.15s
[2022-06-08 16:32:15] [-1 & 2] avg_exec_t = 199.79s, avg_grad_t = 143.55s, avg_loss_t = 56.24s
[2022-06-08 16:32:15] [MaxAcc] max_accuracy = 45.06
```