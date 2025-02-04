import os
import json
import re

from transform import transform

## 05

voc_labels = ('bicycle', 'car', 'motorbike', 'person')
# we do not use 0 as a `voc_label` as we reserve that for the background label
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0

# pre (only once)

def parse_text_files(annotation_path, obj_type):
    """
    Function to parser the dataset text files. You can find 
    more details about the datset here:
    http://host.robots.ox.ac.uk/pascal/VOC/voc2005/chapter.pdf
    """
    boxes = list()
    labels = list()
    with open(annotation_path) as file:
        lines = file.readlines()

    for line in lines:
        if 'Bounding box' in line:
            label = obj_type 
            
            colon_split = line.split(':')[-1] # split by colon sign, :
            space_split = re.split('\s|(?<!\d)[,.]|[,.](?!\d)|\(|\)', colon_split)
            # print('LINE', space_split[2], space_split[4], space_split[8], space_split[10])
            xmin = int(space_split[2]) - 1
            ymin = int(space_split[4]) - 1
            xmax = int(space_split[8]) - 1
            ymax = int(space_split[10]) - 1

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_map[label])

    return {'boxes': boxes, 'labels': labels}

def create_data(voc2005_1_path, voc2005_2_path, output_folder):
    """
    This function creates lists of images, bounding boxes, and  the 
    corresponding labels and saves them as JSON files.

    :param voc07_path: path to `voc2005_1` folder
    :param voc12_path: path to `voc2005_2` folder
    :param output_folder: path to save the JSON files
    """
    voc2005_1_path = os.path.abspath(voc2005_1_path)
    voc2005_2_path = os.path.abspath(voc2005_2_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    # training data
    for path in [voc2005_1_path, voc2005_2_path]:

        data_folders = os.listdir(path+'/Annotations')

        for data_folder in data_folders:
            # ignore the test images folder
            if 'test' not in data_folder.lower():
                print(data_folder)
                if 'car' in data_folder.lower():
                    print('GOT CAR')
                    obj_type = 'car'
                elif 'voiture' in data_folder.lower():
                    print('GOT CAR')
                    obj_type = 'car'
                elif 'moto' in data_folder.lower():
                    print('GOT MOTORBIKE')
                    obj_type = 'motorbike'
                elif 'person' in data_folder.lower():
                    print('GOT PERSON')
                    obj_type = 'person'
                elif 'pieton' in data_folder.lower():
                    print('GOT PERSON')
                    obj_type = 'person'
                elif 'pedestrian' in data_folder.lower():
                    print('GOT PERSON')
                    obj_type = 'person'
                elif 'bike' in data_folder.lower():
                    print('GOT BICYLE')
                    obj_type = 'bicycle'
                elif 'velo' in data_folder.lower():
                    print('GOT BICYLE')
                    obj_type = 'bicycle'
                elif 'bicycle' in data_folder.lower():
                    print('GOT BICYLE')
                    obj_type = 'bicycle'
                

                text_files = os.listdir(os.path.join(path+'/Annotations', data_folder))
                for file in text_files:
                    # parse text files
                    objects = parse_text_files(os.path.join(path+'/Annotations', 
                                                    data_folder, file.split('.')[0] + '.txt'), 
                                                    obj_type)
                    if len(objects) == 0:
                        continue
                    n_objects += len(objects)
                    train_objects.append(objects)
                    train_images.append(os.path.join(path, 'PNGImages', data_folder, file.split('.')[0] + '.png'))

    assert len(train_objects) == len(train_images)

    # save training JSON files and label map
    if output_folder is None:
        output_folder_train = voc2005_1_path
    else:
        output_folder_train = output_folder
    with open(os.path.join(output_folder_train, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder_train, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder_train, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  

    print(f"Total training images: {len(train_images)}")
    print(f"Total training objects: {n_objects}")
    print(f"File save path: {os.path.abspath(output_folder_train)}")


    # test data
    test_images = list()
    test_objects = list()
    n_objects = 0

    for path in [voc2005_1_path, voc2005_2_path]:

        data_folders = os.listdir(path+'/Annotations')

        for data_folder in data_folders:
            # ignore the test images folder
            if 'test' in data_folder.lower():
                print(data_folder)
                if 'car' in data_folder.lower():
                    print('GOT CAR')
                    obj_type = 'car'
                elif 'voiture' in data_folder.lower():
                    print('GOT CAR')
                    obj_type = 'car'
                elif 'moto' in data_folder.lower():
                    print('GOT MOTORBIKE')
                    obj_type = 'motorbike'
                elif 'person' in data_folder.lower():
                    print('GOT PERSON')
                    obj_type = 'person'
                elif 'pieton' in data_folder.lower():
                    print('GOT PERSON')
                    obj_type = 'person'
                elif 'pedestrian' in data_folder.lower():
                    print('GOT PERSON')
                    obj_type = 'person'
                elif 'bike' in data_folder.lower():
                    print('GOT BICYLE')
                    obj_type = 'bicycle'
                elif 'velo' in data_folder.lower():
                    print('GOT BICYLE')
                    obj_type = 'bicycle'
                elif 'bicycle' in data_folder.lower():
                    print('GOT BICYLE')
                    obj_type = 'bicycle'
                

                text_files = os.listdir(os.path.join(path+'/Annotations', data_folder))
                for file in text_files:
                    # parse text files
                    objects = parse_text_files(os.path.join(path+'/Annotations', 
                                                    data_folder, file.split('.')[0] + '.txt'), 
                                                    obj_type)
                    if len(objects) == 0:
                        continue
                    n_objects += len(objects)
                    test_objects.append(objects)
                    test_images.append(os.path.join(path, 'PNGImages', data_folder, file.split('.')[0] + '.png'))

    assert len(test_objects) == len(test_images)

    # save test JSON files
    if output_folder is None:
        output_folder_test = voc2005_2_path
    else:
        output_folder_test = output_folder
    with open(os.path.join(output_folder_test, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder_test, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print(f"Total test images: {len(test_images)}")
    print(f"Total test objects: {n_objects}")
    print(f"File save path: {os.path.abspath(output_folder_test)}")

## 

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class PascalVOCDataset05(Dataset):
    """
    Custom dataset to load PascalVOC data as batches
    """
    def __init__(self, data_folder, split, img_size=300, size=None):
        """
        :param data_folder: folder path of the data files
        :param split: either `TRAIN` or `TEST`
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder

        # read the data files
        with open(os.path.join(data_folder, 
                               self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, 
                               self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)
        
        if size is not None:
            self.images = self.images[:size]
            self.objects = self.objects[:size]

        assert len(self.images) == len(self.objects)

        self.img_size = img_size
        self.dataset_shape = (len(self.images), img_size, img_size, 3)
        self.n_attributes = len(voc_labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        # read image
        image = Image.open(self.images[i])
        image = image.convert('RGB')

        # get bounding boxes, labels, diffculties for the corresponding image
        # all of them are objects
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)

        # apply transforms
        image, boxes, labels = transform(image, boxes, labels, split=self.split, new_dims=(self.img_size,self.img_size))
        # return image, boxes, labels
        return image, labels

    def collate_fn(self, batch):
        """
        Each batch can have different number of objects.
        We will pass this collate function to the DataLoader.
        You can define this function outside the class as well.

        :param batch: iterable items from __getitem(), size equal to batch size
        :return: a tensor of images, lists of varying-size tensors of 
                 bounding boxes, labels, and difficulties
        """
        images = list()
        # boxes = list()
        labels = list()
        K = len(voc_labels)
        binary_labels = list()

        for b in batch:
            images.append(b[0])
            # boxes.append(b[1])
            # labels.append(b[2])
            y = torch.zeros(K)
            y[b[1]-1] = 1 # do not treat background as attribute
            binary_labels.append(y)

        images = torch.stack(images, dim=0)
        labels = torch.stack(binary_labels, dim=0).to(torch.double)

        # return images, boxes, labels
        return images, labels
