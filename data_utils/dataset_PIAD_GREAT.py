import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from PIL import Image
from torchvision import transforms
import pdb
import json
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

def img_normalize_train(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img

def img_normalize_val(img, scale=256/224, input_size=224):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img

class PIAD(Dataset):
    def __init__(self, run_type, setting_type, point_path, img_path, text_hk_path, text_ok_path, pair=2, img_size=(224, 224)):
        super().__init__()

        self.run_type = run_type
        self.p_path = point_path
        self.i_path = img_path
        self.text_hk_path = text_hk_path
        self.text_ok_path = text_ok_path
        self.pair_num = pair
        self.affordance_label_list = ['grasp', 'contain', 'lift', 'open', 
                        'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                        'push', 'listen', 'wear', 'press', 'cut', 'stab', 'carry', 'ride',
                        'clean', 'play', 'beat', 'speak', 'pull']  # 24
        '''
        Unseen affordance
        '''#41

        if setting_type == 'Unseen_aff':
            number_dict = {'Bag': 0, 'Microphone': 0, 'Toothbrush': 0, 'TrashCan': 0, 'Bicycle': 0,
                           'Guitar': 0, 'Glasses': 0, 'Hat': 0, 'Microwave':0, 'Door':0, 'Scissors': 0, 'Bowl': 0,
                           'Baseballbat': 0, 'Mop': 0, 'Dishwasher': 0, 'Bed': 0, 'Keyboard': 0, 'Clock': 0, 'Vase': 0, 'Knife': 0,
                            'Hammer': 0, 'Refrigerator': 0, 'Chair': 0, 'Umbrella': 0, 'Bucket': 0,
                           'Display': 0, 'Earphone': 0, 'Motorcycle': 0, 'StorageFurniture': 0, 'Fork': 0, 'Broom': 0, 'Skateboard': 0,
                           'Tennisracket': 0, 'Laptop': 0, 'Table':0, 'Bottle': 0, 'Faucet': 0, 'Kettle': 0, 'Surfboard': 0, 'Mug': 0,
                            'Spoon': 0
                            }
        '''
        Unseen object
        '''  # 32

        if setting_type == 'Unseen_obj':
            number_dict = {'Bag': 0, 'Microphone': 0, 'Toothbrush': 0, 'TrashCan': 0, 'Bicycle': 0,
                           'Guitar': 0, 'Glasses': 0, 'Hat': 0, 'Microwave':0, 'Backpack': 0, 'Door':0,  'Bowl': 0,
                            'Dishwasher': 0, 'Bed': 0, 'Keyboard': 0,  'Vase': 0, 'Knife': 0,
                           'Suitcase': 0, 'Hammer': 0,  'Chair': 0, 'Umbrella': 0,
                           'Display': 0, 'Earphone': 0, 'StorageFurniture': 0, 'Broom': 0, 
                           'Tennisracket': 0,  'Table':0, 'Bottle': 0, 'Faucet': 0,  'Surfboard': 0, 'Mug': 0,
                            'Spoon': 0
                            }

        '''
        Seen
        '''  # 43

        if setting_type == 'Seen':
            number_dict = {'Bag': 0, 'Microphone': 0, 'Toothbrush': 0, 'TrashCan': 0, 'Bicycle': 0,
                           'Guitar': 0, 'Glasses': 0, 'Hat': 0, 'Microwave':0, 'Backpack': 0, 'Door':0, 'Scissors': 0, 'Bowl': 0,
                           'Baseballbat': 0, 'Mop': 0, 'Dishwasher': 0, 'Bed': 0, 'Keyboard': 0, 'Clock': 0, 'Vase': 0, 'Knife': 0,
                           'Suitcase': 0, 'Hammer': 0, 'Refrigerator': 0, 'Chair': 0, 'Umbrella': 0, 'Bucket': 0,
                           'Display': 0, 'Earphone': 0, 'Motorcycle': 0, 'StorageFurniture': 0, 'Fork': 0, 'Broom': 0, 'Skateboard': 0,
                           'Tennisracket': 0, 'Laptop': 0, 'Table':0, 'Bottle': 0, 'Faucet': 0, 'Kettle': 0, 'Surfboard': 0, 'Mug': 0,
                            'Spoon': 0 
                           }

            aff_number_dict = {'grasp': 0, 'contain': 0, 'lift': 0, 'open': 0, 
                        'lay': 0, 'sit': 0, 'support': 0, 'wrapgrasp': 0, 'pour': 0, 'move': 0, 'display': 0,
                        'push': 0, 'listen': 0, 'wear': 0, 'press': 0, 'cut': 0, 'stab': 0, 'carry': 0, 'ride': 0,
                        'clean': 0, 'play': 0, 'beat': 0, 'speak': 0, 'pull': 0 
                           }        

        self.img_files = self.read_file(self.i_path)
        self.text_human_files = self.read_file(self.text_hk_path)
        self.text_object_files = self.read_file(self.text_ok_path)
        self.img_size = img_size

        if self.run_type == 'train':
            self.point_files, self.number_dict = self.read_file(self.p_path, number_dict)
            self.object_list = list(number_dict.keys())
            self.object_train_split = {}
            start_index = 0
            for obj_ in self.object_list:
                temp_split = [start_index, start_index + self.number_dict[obj_]]
                self.object_train_split[obj_] = temp_split
                start_index += self.number_dict[obj_]
        else:
            self.point_files = self.read_file(self.p_path)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        img_path = self.img_files[index]
        text_hd = self.text_human_files[index]
        text_od = self.text_object_files[index]

        if (self.run_type=='val'):
            point_path = self.point_files[index]
        else:
            object_name = img_path.split('/')[-4]
            affordance_name = img_path.split('/')[-2]
            range_ = self.object_train_split[object_name]
            point_sample_idx = random.sample(range(range_[0],range_[1]), self.pair_num)
      

            for i ,idx in enumerate(point_sample_idx):
                while True:
                    point_path = self.point_files[idx]
                    sele_affordance = point_path.split('/')[-2]
                    if sele_affordance == affordance_name:
                        point_sample_idx[i] = idx 
                        break
                    else:
                        idx = random.randint(range_[0],range_[1]-1)  # re-select idx

        Img = Image.open(img_path).convert('RGB')

        if(self.run_type == 'train'):
            Img = Img.resize(self.img_size)
            Img = img_normalize_train(Img)

            Points_List = []
            affordance_label_List = []
            affordance_index_List = []
            for id_x in point_sample_idx:
                point_path = self.point_files[id_x]
                Points, affordance_label = self.extract_point_file(point_path)
                Points,_,_ = pc_normalize(Points)
                Points = Points.transpose()
                affordance_index = self.get_affordance_label(img_path)
                Points_List.append(Points)
                affordance_label_List.append(affordance_label)
                affordance_index_List.append(affordance_index)

        else:
            Img = Img.resize(self.img_size)
            Img = img_normalize_train(Img)

            Point, affordance_label = self.extract_point_file(point_path)
            Point,_,_ = pc_normalize(Point)
            Point = Point.transpose()

        if(self.run_type == 'train'):
            return Img, text_hd, text_od, Points_List, affordance_label_List, affordance_index_List
        else:
            return Img, text_hd, text_od, Point, affordance_label, img_path, point_path

    def read_file(self, path, number_dict=None):
        file_list = []
        with open(path,'r') as f:
            files = f.readlines()
            for file in files:
                file = file.strip('\n')
                if number_dict != None:
                    object_ = file.split('/')[-4]
                    number_dict[object_] +=1
                file_list.append(file)

            f.close()
        if number_dict != None:
            return file_list, number_dict
        else:
            return file_list
    
    def extract_point_file(self, path):
        lines =  np.load(path)
        data_array = np.array(lines)
        points_coordinates = data_array[:, 0:3]
        affordance_label = data_array[: , 3:]

        return points_coordinates, affordance_label

    def get_affordance_label(self, str):
        cut_str = str.split('/')
        affordance = cut_str[-2]
        index = self.affordance_label_list.index(affordance)
        
        return  index
    
 
