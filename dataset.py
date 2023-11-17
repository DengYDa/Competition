# coding: utf-8
# @Author: Salt

import os
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torchvision import transforms
import torch
from PIL import Image
from randaugment import RandomAugment #内置实现了多种图像增强方式
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

transform_train = transforms.Compose([                        
            transforms.Resize((256,256),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])

transform_test = transforms.Compose([
        transforms.Resize((256,256),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,])  

def parse(xml_path):
    from xml.etree import ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for child in root:
        if child.tag == 'object':
            for c_child in child:
                if c_child.tag == 'attributes':
                    age = int(c_child[0][1].text)
                    gender = int(c_child[1][1].text)
                    glasses = int(c_child[2][1].text)
                    race = int(c_child[3][1].text)
                    emotion = int(c_child[4][1].text)
                    mask = int(c_child[5][1].text)
                    hat = int(c_child[6][1].text)
                    whiskers = int(c_child[7][1].text)
                    features = (age, gender, glasses, race, emotion, mask, hat, whiskers)

    return features


class FaceDataset(Dataset):
    def __init__(self, data_path, is_train = True):
        # data_path ---> 数据的路径
        self.data_path = data_path
        self.path = os.listdir(self.data_path)
        self.imgs_path = [x for x in self.path if x.split('.')[1] == 'jpg']
        self.xmls_path = [x for x in self.path if x.split('.')[1] == 'xml']
        self.transforms = transform_train if is_train else transform_test

    def augment(self, image):
        pass

    def __getitem__(self, index):
        image_child_path = self.imgs_path[index]
        image_path = os.path.join(self.data_path, image_child_path)
        xml_path = image_path.split('.')[0] + '.xml'
        labels = parse(xml_path)
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        labels = torch.from_numpy(np.array(labels))
        return image, labels[0].float(), labels[1].long(), labels[2].long(), labels[3].long(), labels[4].long(), labels[5].long(), labels[6].long(), labels[7].long()

    def __len__(self):
        return len(self.imgs_path)


if __name__ == "__main__":
    dataset = FaceDataset("home/data/2792")
    loader = DataLoader(dataset, batch_size=1)
    for images, age_target, gender_target, glasses_target, race_target, emotion_target, mask_target, hat_target, whiskers_target in loader:
        print(images.shape)
        print(gender_target)

