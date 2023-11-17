from torch import nn
from torchvision import models
import torch.nn.functional as F

class Sub_Base(nn.Module):
    
    def __init__(self,target_class,in_feature = 2048 ,dropout = 0.2):
        super().__init__()
        self.target_class = target_class
        
        self.dense = nn.Linear(in_feature, in_feature//2)
        self.to_predict = nn.Linear(in_feature//2, self.target_class)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = nn.ReLU()
        self.bn = nn.BatchNorm1d(in_feature//2)
    
    def forward(self,feature):
        x = self.dense(feature)
        x = self.bn(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.to_predict(x)
        return x

        
class ResNet50_pretrained(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
    
    def forward(self,image):
        
        output = self.resnet(image)
        
        return output

class Baseline(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.bone = ResNet50_pretrained()
        self.age = Sub_Base(target_class = 1)
        self.gender = Sub_Base(target_class = 3) 
        self.glasses = Sub_Base(target_class = 4)
        self.race = Sub_Base(target_class = 5)
        self.emotion = Sub_Base(target_class = 4)
        self.mask = Sub_Base(target_class = 3)
        self.hat = Sub_Base(target_class = 3)
        self.whiskers = Sub_Base(target_class = 3)
    
    def forward(self,image):
        
        feature = self.bone(image)
        
        age = self.age(feature)
        gender = self.gender(feature)
        glasses = self.glasses(feature)
        race = self.race(feature)
        emotion = self.emotion(feature)
        mask = self.mask(feature)
        hat = self.hat(feature)
        whiskers = self.whiskers(feature)
        
        return age,gender,glasses,race,emotion,mask,hat,whiskers