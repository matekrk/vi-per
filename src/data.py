import collections
import os
import sys
import json
import copy
import random
import logging
from PIL import Image
import torch.utils.data
from torchvision import transforms

from utils import load_label, rmdir

def get_transformer(opt):
    transform_list = []
    
    # resize  
    osize = [opt.load_size, opt.load_size]
    transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    
    # grayscale
    if opt.input_channel == 1:
        transform_list.append(transforms.Grayscale())

    # crop
    if opt.crop == "RandomCrop":
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.crop == "CenterCrop":
        transform_list.append(transforms.CenterCrop(opt.input_size))
    elif opt.crop == "FiveCrop":
        transform_list.append(transforms.FiveCrop(opt.input_size))
    elif opt.crop == "TenCrop":
        transform_list.append(transforms.TenCrop(opt.input_size))
    
    # flip
    if opt.mode == "Train" and opt.flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    # to tensor
    transform_list.append(transforms.ToTensor())
    
    # If you make changes here, you should also modified 
    # function `tensor2im` in util/util.py accordingly
    transform_list.append(transforms.Normalize(opt.mean, opt.std))

    return transforms.Compose(transform_list)

def fix_box(box, width, height, ratio=-1, scale=1.0):
    if scale < 0:
        scale = 1.0
    box = copy.deepcopy(box)
    w = box["w"]
    h = box["h"]
    x = box["x"] + w / 2
    y = box["y"] + h / 2
    mw = 2 * min(x, width - x)
    mh = 2 * min(y, height - y)
    w = max(1, min(int(w * scale), mw))
    h = max(1, min(int(h * scale), mh))
    if ratio > 0:
      if 1.0 * w / h > ratio:
          h = int(w / ratio)
          h = min(h, mh)
          w = int(h * ratio)
      else:
          w = int(h * ratio)
          w = min(w, mw)
          h = int(w / ratio)
    box["x"] = x - w / 2
    box["y"] = y - h / 2
    box["w"] = w
    box["h"] = h
    return box

def load_image(image_file, box, opt, transformer):
    img = Image.open(image_file)
    if opt.input_channel == 3:
        img = img.convert('RGB')
    width, height = img.size
    # box crop
    if box is not None and opt.region == True:
        box = fix_box(box, width, height, opt.box_ratio, opt.box_scale)
        area = (box['x'], box['y'], box['x']+box['w'], box['y']+box['h'])
        img = img.crop(area)
    # transform
    input = transformer(img)
    return input

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, opt, data_type, id2rid):
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.data_type = data_type
        self.dataset = self._load_data(opt.data_dir+ '/' + data_type + '/data.txt')
        self.id2rid = id2rid
        self.data_size = len(self.dataset)
        self.image_type = not opt.notimage_type
        if self.image_type: 
            self.transformer = get_transformer(opt)

    def __getitem__(self, index):
        
        if self.image_type:
            image_file, box, attr_ids = self.dataset[index % self.data_size]
            input = load_image(image_file, box, self.opt, self.transformer)
        else:
            data, box, attr_ids = self.dataset[index % self.data_size]
            input = torch.tensor(data)

        # label
        labels = list()
        for index, attr_id in enumerate(attr_ids):
            labels.append(self.id2rid[index][attr_id])

        return input, labels

    def __len__(self):
        return self.data_size

    def _load_data(self, data_file):
        dataset = list()
        if not os.path.exists(data_file):
            return dataset
        with open(data_file) as d:
            for line in d.readlines():
                line = json.loads(line)
                dataset.append(self.readline(line))
        if self.opt.shuffle:
            logging.info("Shuffle %s Data" %(self.data_type))
            random.shuffle(dataset)
        else:
            logging.info("Not Shuffle %s Data" %(self.data_type))
        return dataset
    
    def readline(self, line):
        data = [None, None, None]
        if 'image_file' in line:
            data[0] = line["image_file"]
        elif 'data' in line:
            data[0] = line["data"]
        if 'box' in line:
            data[1] = line["box"]
        if 'id' in line:
            data[2] = line["id"]
        return data

class MultiLabelDataLoader():
    def __init__(self, opt):
        self.opt = opt
        assert os.path.exists(opt.dir + "/data.txt"), "No data.txt found in specified dir"
        assert os.path.exists(opt.dir + "/label.txt"), "No label.txt found in specified dir"
        
        train_dir = opt.data_dir + "/TrainSet/"
        val_dir = opt.data_dir + "/ValidateSet/"
        test_dir = opt.data_dir + "/TestSet/"
         
        # split data
        if not all([os.path.exists(train_dir), os.path.exists(val_dir), os.path.exists(test_dir)]):
            # rm existing directories
            rmdir(train_dir)
            rmdir(val_dir)
            rmdir(test_dir)

            # split data to Train, Val, Test
            logging.info("Split raw data to Train, Val and Test")
            ratios = opt.ratio
            dataset = collections.defaultdict(list)
            with open(opt.dir + '/data.txt') as d:
                lines = d.readlines()
                for i_line, line in enumerate(lines):
                    line = json.loads(line)
                    # if data has been specified data_type yet, load data as what was specified before
                    if "type" in line:
                        dataset[line["type"]].append(line)
                        continue
                    # specified data_type randomly
                    if opt.randomized_selection:
                        rand = random.random()
                        if rand < ratios[0]:
                            data_type = "Train"
                        elif rand < ratios[0] + ratios[1]:
                            data_type = "Validate"
                        else:
                            data_type = "Test"
                    else:
                        if i_line < ratios[0] * len(lines):
                            data_type = "Train"
                        elif i_line < (ratios[0] + ratios[1]) * len(lines):
                            data_type = "Validate"
                        else:
                            data_type = "Test"
                    dataset[data_type].append(line)
            # write to file
            self._WriteDataToFile(dataset["Train"], train_dir)
            self._WriteDataToFile(dataset["Validate"], val_dir)
            self._WriteDataToFile(dataset["Test"], test_dir)
        
        self.rid2name, self.id2rid, self.rid2id = load_label(opt.dir + '/label.txt')
        self.num_classes = [len(item)-2 for item in self.rid2name]
        
        # load dataset
        if opt.mode == "Train": 
            logging.info("Load Train Dataset...")
            self.train_set = BaseDataset(self.opt, "TrainSet", self.rid2id)
            logging.info("Load Validate Dataset...")
            self.val_set = BaseDataset(self.opt, "ValidateSet", self.rid2id)
        else:
            # force batch_size for test to 1
            self.opt.batch_size = 1
            self.opt.load_thread = 1
            logging.info("Load Test Dataset...")
            self.test_set = BaseDataset(self.opt, "TestSet", self.rid2id)

    def GetTrainSet(self):
        if self.opt.mode == "Train":
            return self._DataLoader(self.train_set)
        else:
            raise("Train Set DataLoader NOT implemented in Test Mode")

    def GetValSet(self):
        if self.opt.mode == "Train":
            return self._DataLoader(self.val_set)
        else:
            raise("Validation Set DataLoader NOT implemented in Test Mode")

    def GetTestSet(self):
        if self.opt.mode == "Test":
            return self._DataLoader(self.test_set)
        else:
            raise("Test Set DataLoader NOT implemented in Train Mode")
    
    def GetNumClasses(self):
        return self.num_classes
    
    def GetRID2Name(self):
        return self.rid2name
    
    def GetID2RID(self):
        return self.id2rid
    
    def GetiRID2ID(self):
        return self.irid2id

    def _WriteDataToFile(self, src_data, dst_dir):
        """
            write info of each objects to data.txt as predefined format
        """
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        with open(dst_dir + "/data.txt", 'w') as d:
            for line in src_data:
                d.write(json.dumps(line, separators=(',',':'))+'\n')


    def _DataLoader(self, dataset):
        """
            create data loder
        """
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            shuffle=self.opt.shuffle,
            num_workers=int(self.opt.load_thread), 
            pin_memory=self.opt.cuda,
            drop_last=False)
        return dataloader
