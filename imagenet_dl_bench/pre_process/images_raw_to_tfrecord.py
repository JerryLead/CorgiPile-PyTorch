import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import sys
import os
import datetime

sys.path.append("../shuffleformat/tfrecord")
sys.path.append("../shuffleformat/corgipile")
sys.path.append(".")

import shuffleformat.tfrecord as tfrecord
import shuffleformat.corgipile as corgipile
import random
import io

def test(img_path):
    img = Image.open(img_path)
    img = np.array(img)
    print(img)
    print(img.shape)
    print(img.dtype)
    plt.imshow(img)
    plt.show()

def write_to_TFRecord(image_dir, image_file_list, output_record_file,
                      trans=None, shuffle=False):
    
    writer = tfrecord.writer.TFRecordWriter(output_record_file)
    if shuffle:
        random.shuffle(image_file_list)

    i = 0
    for image in image_file_list:
        img_path = image[0]
        label = image[1]
        index = image[2]
        image_bytes = open(os.path.join(image_dir, img_path), "rb").read()
        
        #print(img)
        # tensor([[[0.7137, 0.7059, 0.7059,  ..., 0.2000, 0.2353, 0.2549],
        # [0.7059, 0.7216, 0.6863,  ..., 0.1569, 0.1922, 0.2118],
        # [0.7098, 0.7216, 0.7137,  ..., 0.1176, 0.1647, 0.1843],
        #print(trans(img))
        # print(img)
        # plt.imshow(img)
        # plt.show()
    
        writer.write({
            "image": (image_bytes, "byte"),
            "label": (label, "int"),
            "index": (index, "int")
        })

        i += 1
        if (i % 1000 == 0):
            print('Has written', i, 'images', flush=True)
    
    writer.close()

# decode image
def decode_image(features):

    img_bytes = features["image"]
    features["image"] = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    
    read_trans = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    features["image"] = read_trans(features["image"])

    features["label"] = features["label"][0]

    return features


def read_image(tfrecord_path, index_path, mode = 'seq'):
    description = {"image": "byte", "label": "int", "index": "int"}


    if mode == 'block':
        dataset = corgipile.dataset.CorgiPileTFRecordDataset(
                                tfrecord_path, 
                                index_path,
                                # block_num=75700,
                                # buffer_size_ratio=0.00002,
                                block_num=500,
                                buffer_size_ratio=0.1,
                                description=description,
                                transform=decode_image)   
    else:
        dataset = tfrecord.torch.dataset.TFRecordDataset(
                                tfrecord_path, 
                                index_path,
                                description, 
                                transform=decode_image)                                          
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=1)

   
    epochs = 1
    for epoch in range(0, epochs):
        if mode == 'block':
            dataset.set_epoch(epoch)
     
        start_time = datetime.datetime.now()
        for i, images in enumerate(loader):
          
            for label in images['label']:
                print(label)
            j = 0
        
            for img in images['image']:
                img
                j = j + 1
            #print(images['label'])
            end_time = datetime.datetime.now()
            strTime = 'read time = %dms' % ((end_time - start_time).seconds * 1000 + (end_time - start_time).microseconds / 1000)
            print(strTime, "[", j, " images]")
           
            start_time = end_time
            if i == 1200:
                break
     
def build_index(output_record_file, index_file):
    tfrecord.tools.tfrecord2idx.create_index(output_record_file, index_file)

def get_class_labels(image_dir):
    classes = [d.name for d in os.scandir(image_dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return class_to_idx

def fetch_all_images(image_absolute_dir, class_labels, max_num=0):
    image_file_list = []
    index = 0
    for img_dir in os.listdir(image_absolute_dir): 
        if not img_dir.startswith('.'):
            class_label = class_labels[img_dir]
            for img_file in os.listdir(os.path.join(image_absolute_dir, img_dir)):
                if not img_file.startswith('.'):
                    path = os.path.join(img_dir, img_file)
                    index += 1
                    image_file_list.append((path, class_label, index))

                    if len(image_file_list) == max_num:
                        return image_file_list
    return image_file_list


def generate_tfRecords_with_index(image_dir, class_labels, 
                                  output_record_file, index_file, max_num=0, shuffle=False):
    image_file_list = fetch_all_images(image_dir, class_labels, max_num)
    write_to_TFRecord(image_dir, image_file_list, output_record_file,
                      trans=None, shuffle=shuffle)
    
    build_index(output_record_file, index_file)

def main():

    base_dir = "/home/username/"
    raw_images_base_dir = base_dir + "ImageNet/"
    tfrecord_output_base_dir = os.path.join("/home/username/corgipile_data/", "ImageNet-all-raw-tfrecords")
    shuffle = False # True if you would like to get shuffled images in the generated TFRecords
    generate_tfRecords = True
    read_tfRecords = False
    
    images_num = 0 # the number of images included in the gerated TFRecords, 0 for all the images

    image_train_dir = os.path.join(raw_images_base_dir, "train")
    train_output_dir = os.path.join(tfrecord_output_base_dir, "train")
    train_tfrecord_file = os.path.join(train_output_dir, "train_clustered.tfrecord")
    train_index_file = os.path.join(train_output_dir, "train_clustered.index")

    image_val_dir = os.path.join(raw_images_base_dir, "val")
    val_output_dir = os.path.join(tfrecord_output_base_dir, "val")
    val_tfrecord_file = os.path.join(val_output_dir, "val_clustered.tfrecord")
    val_index_file = os.path.join(val_output_dir, "val_clustered.index")


    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
    if not os.path.exists(val_output_dir):
        os.makedirs(val_output_dir)

    if generate_tfRecords:
        # assert the train and val images have the same class labels
        print("[Parse] Parsing class labels of training images.")
        train_class_labels = get_class_labels(image_train_dir)
        print("[Parse] Parsing class labels of validation images.")
        val_class_labels = get_class_labels(image_val_dir)
        assert(train_class_labels == val_class_labels)
        
        print("[TFRecords] Generating tfrecords of training images.")
        generate_tfRecords_with_index(image_train_dir, train_class_labels, 
                                      train_tfrecord_file, train_index_file, max_num=images_num, shuffle=shuffle)
        
        print("[TFRecords] Generating tfrecords of validation images.")
        generate_tfRecords_with_index(image_val_dir, val_class_labels, 
                                      val_tfrecord_file, val_index_file, max_num=images_num)

    if read_tfRecords:
        print("[Test] Reading tfrecords of training images.")
        start_time = datetime.datetime.now()
        mode = 'seq'
        read_image(train_tfrecord_file, train_index_file, mode)
        print("read val")
        read_image(val_tfrecord_file, val_index_file, mode)
        end_time = datetime.datetime.now()
        strTime = '[Test] data read time = %d ms' % ((end_time - start_time).seconds * 1000 + (end_time - start_time).microseconds / 1000)
        print(strTime)


if __name__ == '__main__':
    main()