import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from data_utils.dataset_PIAD_GREAT import PIAD
from model.GREAT import GREAT
from utils.eval import SIM
from numpy import nan
import numpy as np
import pdb
import random
import os
import pandas as pd
import yaml
import argparse
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import logging

def Evalization(dataset, data_loader, model_path, use_gpu, Setting):
    if opt.use_gpu:
        rank = dist.get_rank()
        size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        if opt.use_gpu:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
      
    log_dir = os.path.join(opt.save_dir, opt.name)  
    os.makedirs(log_dir, exist_ok=True)  

    log_name = os.path.join(log_dir, opt.log_name)  
    logging.basicConfig(filename=log_name, level=logging.INFO)

    loger = logging.getLogger('Evalization')
    loger.setLevel(logging.INFO)         
    log_name = opt.save_dir + opt.name + '/' + opt.log_name
    logging.basicConfig(filename=log_name, level=logging.INFO)
    def log_string(str):
        loger.info(str)
        print(str)  

    if(Setting == 'Unseen_aff'):
        object_list = ['Backpack', 'Bed', 'Bottle', 'Earphone', 'Kettle', 'Knife',
                        'Mug', 'Scissors', 'Suitcase', 'Surfboard','TrashCan']#11

        Affordance_list = ['carry', 'listen', 'lay', 'pour', 'cut', 'pull']#6    
    

    if(Setting == 'Unseen_obj'):
        object_list = ['Baseballbat', 'Bucket', 'Clock', 'Fork', 'Kettle', 'Laptop', 'Mop', 'Motorcycle', 'Refrigerator', 
                       'Scissors', 'Skateboard']#11
        Affordance_list = ['grasp', 'contain', 'lift', 'open', 
                         'support', 'wrapgrasp', 'pour', 'display',
                        'press', 'cut', 'stab',  'ride',
                        'clean']#13
    
    if(Setting == 'Seen'):
        object_list = ['Bag', 'Microphone', 'Toothbrush', 'TrashCan', 'Bicycle',
                        'Guitar', 'Glasses', 'Hat', 'Microwave', 'Backpack', 'Door', 'Scissors', 'Bowl',
                        'Baseballbat', 'Mop', 'Dishwasher', 'Bed', 'Keyboard', 'Clock', 'Vase', 'Knife',
                        'Suitcase', 'Hammer', 'Refrigerator', 'Chair', 'Umbrella', 'Bucket',
                        'Display', 'Earphone', 'Motorcycle', 'StorageFurniture', 'Fork', 'Broom', 'Skateboard',
                        'Tennisracket', 'Laptop', 'Table', 'Bottle', 'Faucet', 'Kettle', 'Surfboard', 'Mug',
                        'Spoon']#43

        Affordance_list = ['grasp', 'contain', 'lift', 'open', 
                        'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                        'push', 'listen', 'wear', 'press', 'cut', 'stab', 'carry', 'ride',
                        'clean', 'play', 'beat', 'speak', 'pull']#24

    for obj in object_list:
        exec(f'{obj} = [[], [], [], []]')
    for aff in Affordance_list:
        exec(f'{aff} = [[], [], [], []]')

    model = GREAT(pre_train=False)

    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)
   
    checkpoint = torch.load(model_path, map_location='cuda:0')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
  
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    results = torch.zeros((len(dataset), 2048, 1))
    targets = torch.zeros((len(dataset), 2048, 1))
    total_point = 0
    num = 0
    with torch.no_grad():
        model.eval()
        Object = []
        Affordance = []
        for i,(img, text_hd, text_od, point, label, img_path, point_path) in enumerate(data_loader):
            print(f'iteration: {i} start----')
            B = img.shape[0]
            for iter in range(B):
                
                object_class = point_path[iter].split('/')[-4]
                affordance_cls = img_path[iter].split('/')[-2]
                Object.append(object_class)
                Affordance.append(affordance_cls)

            point, label = point.float(), label.float()

            if(use_gpu):
                img = img.to(device)
                point = point.to(device)
                label = label.to(device)
        
            pred = model(img, point, text_hd, text_od)

            pred_num = pred.shape[0]
            print(f'num:{num}, pred_num:{pred_num}')
            results[num : num+pred_num, :, :] = pred
            targets[num : num+pred_num, :, :] = label
            num += pred_num

        results = results.detach().numpy()
        targets = targets.detach().numpy()
        SIM_matrix = np.zeros(targets.shape[0])
        MAE_martrix = np.zeros(targets.shape[0])
        for i in range(targets.shape[0]):
            Sim = SIM(results[i], targets[i])
            mAE = np.sum(np.absolute(results[i]-targets[i])) / 2048
            SIM_matrix[i] = Sim
            MAE_martrix[i] = mAE
            
            object_cls = Object[i]
            aff_cls = Affordance[i]
            exec(f'{object_cls}[1].append({Sim})')
            exec(f'{aff_cls}[1].append({Sim})')
            exec(f'{object_cls}[3].append({mAE})')
            exec(f'{aff_cls}[3].append({mAE})')


        sim = np.mean(SIM_matrix)
        mean_MAE = np.mean(MAE_martrix)
        AUC = np.zeros((targets.shape[0], targets.shape[2]))
        IOU = np.zeros((targets.shape[0], targets.shape[2]))
        IOU_thres = np.linspace(0, 1, 20)
        targets = targets >= 0.5
        targets = targets.astype(int)
        for i in range(AUC.shape[0]):
            t_true = targets[i]
            p_score = results[i]
            object_cls = Object[i]
            aff_cls = Affordance[i]
            if np.sum(t_true) == 0:
                AUC[i] = np.nan
                IOU[i] = np.nan
                obj_auc = AUC[i]
                aff_auc = AUC[i]
                obj_iou = IOU[i]
                aff_iou = IOU[i]
                exec(f'{object_cls}[2].append({obj_auc})')
                exec(f'{aff_cls}[2].append({aff_auc})')
                exec(f'{object_cls}[0].append({obj_iou})')
                exec(f'{aff_cls}[0].append({aff_iou})')
            else:
                auc = roc_auc_score(t_true, p_score)
                AUC[i] = auc

                p_mask = (p_score > 0.5).astype(int)
                temp_iou = []
                for thre in IOU_thres:
                    p_mask = (p_score >= thre).astype(int)
                    intersect = np.sum(p_mask & t_true)
                    union = np.sum(p_mask | t_true)
                    temp_iou.append(1.*intersect/union)
                temp_iou = np.array(temp_iou)
                aiou = np.mean(temp_iou)
                IOU[i] = aiou

                obj_auc = AUC[i]
                aff_auc = AUC[i]
                obj_iou = IOU[i]
                aff_iou = IOU[i]
                exec(f'{object_cls}[2].append({obj_auc})')   
                exec(f'{aff_cls}[2].append({aff_auc})')
                exec(f'{object_cls}[0].append({obj_iou})')
                exec(f'{aff_cls}[0].append({aff_iou})')

        AUC = np.nanmean(AUC)
        IOU = np.nanmean(IOU)

        log_string('------Object-------')

        for obj in object_list:
            aiou = np.nanmean(eval(obj)[0])*100
            sim_ = np.mean(eval(obj)[1])
            auc_ = np.nanmean(eval(obj)[2])*100
            mae_ = np.mean(eval(obj)[3])

            log_string(f'{obj} | AUC:{auc_} | IOU:{aiou} | SIM:{sim_} | MAE:{mae_}')

        avg_mertics = [0, 0, 0, 0]
        log_string('------Affordance-------')

        for i,aff in enumerate(Affordance_list):
            aiou = np.nanmean(eval(aff)[0])*100
            sim_ = np.mean(eval(aff)[1])
            auc_ = np.nanmean(eval(aff)[2])*100
            mae_ = np.mean(eval(aff)[3])
            avg_mertics[0] += aiou
            avg_mertics[1] += sim_
            avg_mertics[2] += auc_
            avg_mertics[3] += mae_
   
    
            log_string(f'{aff} | AUC:{auc_} | IOU:{aiou} | SIM:{sim_} | MAE:{mae_}')

        num_affordance = len(Affordance_list)
        avg_iou, avg_sim = avg_mertics[0] / num_affordance, avg_mertics[1] / num_affordance
        avg_auc, avg_mae = avg_mertics[2] / num_affordance, avg_mertics[3] / num_affordance

        log_string('------ALL-------')
        log_string(f'Overall---AUC:{AUC*100} | IOU:{IOU*100} | SIM:{sim} | MAE:{mean_MAE}')

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)

    return dict

def run(opt):

    dict = read_yaml(opt.yaml)
    point_path = dict['point_test']
    img_path = dict['img_test']
    text_path = dict['text_test']
    text_hd_val_path = dict['human_dictionary_test']
    text_od_val_path = dict['object_dictionary_test']   

    model_path = opt.checkpoint_path

    if opt.use_gpu:
        dist.init_process_group(backend='nccl', init_method='env://')

    val_dataset = PIAD('val', dict['Setting'], point_path, img_path, text_hd_val_path, text_od_val_path)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, dict['batch_size'], sampler=val_sampler, num_workers=8)
    #val_loader = DataLoader(val_dataset, dict['batch_size'], num_workers=8)
    Evalization(val_dataset, val_loader, model_path, opt.use_gpu, Setting=dict['Setting'])

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    #parser.add_argument('--gpu', type=str, default='cuda:0', help='gpu device id')
    parser.add_argument('--use_gpu', type=str, default=True, help='whether or not use gpus')
    parser.add_argument('--checkpoint_path', type=str, default='runs/GREAT/best_seen.pt', help='checkpoint path')
    parser.add_argument('--yaml', type=str, default='config/config_seen_GREAT.yaml', help='yaml path')
    parser.add_argument('--log_name', type=str, default='evalization_seen.log', help='save the results')
    parser.add_argument('--save_dir', type=str, default='/runs/', help='path to save .pt model while training')
    parser.add_argument('--name', type=str, default='GREAT', help='training name to classify each training process')
       

    opt = parser.parse_args()
    seed_torch(42)
    run(opt)
