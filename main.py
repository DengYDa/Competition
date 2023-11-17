import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models import Baseline
from tools import warmup_lr_schedule, step_lr_schedule
from sklearn.metrics import f1_score
from dataset import FaceDataset
import time
import numpy as np
import random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
# Logging
import logging

# config
config = {
    'epoch': 2,
    'init_lr'  : 3e-4,
    'min_lr': 1e-6,
    'warmup_lr': 1e-6,
    'lr_decay_rate': 0.9,
    'weight_decay' : 0.05,
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed' : 42,
    'output_dir' : '/project/train/models/'
}



# DataLoader
train_dataset = FaceDataset("/home/data/2792")
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, 
    shuffle=True, num_workers=8,pin_memory=True,
    drop_last = False
)

val_dataset = FaceDataset("/home/data/2792",is_train = False)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=128,shuffle=False, 
    num_workers=8,pin_memory=True,
    drop_last = False
)


def f1_score_com(x,target):
    res = []
    for i in x:
        max_idx = max_idx = torch.argmax(x, dim=1)
        output = max_idx.unsqueeze(1)
        res.append(output)
    pred = torch.cat(res,dim = 1) - 1
    f1 = f1_score(target.cpu().numpy(),pred.cpu().numpy())
    
    return f1
    
    




def train(model,config,train_loader,val_loader,optimizer):
    # Training
    model.train()
    train_loss = 0.0
    for images, age_target, gender_target, glasses_target, race_target, emotion_target, mask_target, hat_target, whiskers_target in train_loader:
        optimizer.zero_grad()
        images, age_target, gender_target, glasses_target, race_target, emotion_target, mask_target, hat_target, whiskers_target = images.to(config['device']), \
            age_target.to(config['device']), gender_target.to(config['device']), \
            glasses_target.to(config['device']), race_target.to(config['device']), emotion_target.to(config['device']), \
            mask_target.to(config['device']), hat_target.to(
                config['device']), whiskers_target.to(config['device'])

        age, gender, glasses, race, emotion, mask, hat, whiskers = model(
            images)
        gender, glasses, race, emotion, mask, hat, whiskers = map(lambda x: F.softmax(
            x, dim=1), [gender, glasses, race, emotion, mask, hat, whiskers])
        loss_age = F.mse_loss(age, age_target.unsqueeze(1))
        loss_gender = F.cross_entropy(gender, gender_target)
        loss_glasses = F.cross_entropy(glasses, glasses_target)
        loss_race = F.cross_entropy(race, race_target)
        loss_emotion = F.cross_entropy(emotion, emotion_target)
        loss_mask = F.cross_entropy(mask, mask_target)
        loss_hat = F.cross_entropy(hat, hat_target)
        loss_whiskers = F.cross_entropy(whiskers, whiskers_target)
        loss = loss_age + loss_gender + loss_glasses + loss_race + loss_emotion + loss_mask + loss_hat + loss_whiskers
        loss.backward()
        optimizer.step()
    train_loss += loss.item()
        
    # Evaluation 
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, age_target, gender_target, glasses_target, race_target, emotion_target, mask_target, hat_target, whiskers_target in val_loader:
            images, age_target, gender_target, glasses_target, race_target, emotion_target, mask_target, hat_target, whiskers_target = images.to(config['device']), \
                age_target.to(config['device']), gender_target.to(config['device']), \
                glasses_target.to(config['device']), race_target.to(config['device']), emotion_target.to(config['device']), \
                mask_target.to(config['device']), hat_target.to(
                    config['device']), whiskers_target.to(config['device'])
            age, gender, glasses, race, emotion, mask, hat, whiskers = model(
                images)
            gender, glasses, race, emotion, mask, hat, whiskers = map(lambda x: F.softmax(
                x, dim=1), [gender, glasses, race, emotion, mask, hat, whiskers])

            loss_age = F.mse_loss(age, age_target.unsqueeze(1))
            loss_gender = F.cross_entropy(gender, gender_target)
            loss_glasses = F.cross_entropy(glasses, glasses_target)
            loss_race = F.cross_entropy(race, race_target)
            loss_emotion = F.cross_entropy(emotion, emotion_target)
            loss_mask = F.cross_entropy(mask, mask_target)
            loss_hat = F.cross_entropy(hat, hat_target)
            loss_whiskers = F.cross_entropy(whiskers, whiskers_target)
            loss = loss_age + loss_gender + loss_glasses + loss_race + loss_emotion + loss_mask + loss_hat + loss_whiskers
            f1 = f1_score_com([gender_target, glasses_target, race_target, emotion_target, mask_target,
                              hat_target, whiskers_target], [gender, glasses, race, emotion, mask, hat, whiskers])
        val_loss += loss.item()
        
    if f1 > f1_best:
        f1_best = f1
        save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
        torch.save(save_obj, os.path.join(config['output_dir'], 'checkpoint_%02d.pth'%epoch))
        logger.info('best_f1', f1_best, epoch)
    # Logging
    logger.info(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')
    writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)
    writer.add_scalar('Loss/val', val_loss/len(val_loader), epoch)



def main(config):
    # Tensorboard
    writer = SummaryWriter(log_dir = '/project/train/tensorboard/')
    #logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler('/project/train/log/log.txt')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    
    device = config['device']

    # fix the seed for reproducibility
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    
    # Model 
    model = Baseline()
    model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    
    f1_best = 0.0
    start_time = time.time()
    logger.info(f"开始训练, 共 {config['epoch']}个epoch, 训练数据共 {len(train_dataset)}, 验证数据共 {len(val_dataset)}")
    for epoch in range(0, config['epoch']):
        step_lr_schedule(optimizer, epoch, config['init_lr'], config['min_lr'], config['lr_decay_rate'])
        
        train(model,config,train_loader,val_loader,optimizer)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    
    writer.close()
    
    
if __name__ == '__main__':
    main(config)