import numpy as np
import torch

#MAE 
def evaluating(pred, label):

    mae = torch.sum(torch.abs(pred-label), dim=(0,1)) 
    points_num = pred.shape[0] * pred.shape[1]

    return mae, points_num #平均绝对误差和点的数量

#KL散度 衡量两个概率分布之间的差异的指标，map1.2是两个概率分布
def KLD(map1, map2, eps = 1e-12): 
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    kld = np.sum(map2*np.log( map2/(map1+eps) + eps))
    return kld
    
#SIM函数 用于计算两个概率分布之间的相似度    
def SIM(map1, map2, eps=1e-12): 
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)
