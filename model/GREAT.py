import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
from einops import rearrange
from transformers import AutoModel, AutoTokenizer
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Cross_Attention(nn.Module):    
    def __init__(self, emb_dim, proj_dim):
        super().__init__()
        self.emb_dim = emb_dim 
        self.proj_dim = proj_dim 
        self.proj_hq = nn.Linear(self.emb_dim, proj_dim)
        self.proj_oq = nn.Linear(self.emb_dim, proj_dim)
        self.proj_hk = nn.Linear(self.emb_dim, proj_dim)
        self.proj_hv = nn.Linear(self.emb_dim, proj_dim)
        self.proj_ok = nn.Linear(self.emb_dim, proj_dim)
        self.proj_ov = nn.Linear(self.emb_dim, proj_dim)
        self.scale = self.proj_dim ** (-0.5) 

        self.layernorm = nn.LayerNorm(self.emb_dim)
    def forward(self, hk, ok):

        '''
        hk : human knowledge [B,N_hk,C]
        ok : object knowledge [B,N_ok,C]
        '''

        hk_q = self.proj_hq(hk)                                        
        ok_key = self.proj_ok(ok)                                       
        ok_value = self.proj_ov(ok)

        ok_key_ = torch.cat((hk_q,ok_key),dim=1)  
        ok_value_ = torch.cat((hk_q,ok_value),dim=1)

        ok_q = self.proj_oq(ok)
        hk_key = self.proj_hk(hk)
        hk_value = self.proj_hv(hk)

        hk_key_ = torch.cat((ok_q,hk_key),dim=1)  
        hk_value_ = torch.cat((ok_q,hk_value),dim=1)

        atten_I1 = torch.bmm(hk_q, ok_key_.permute(0, 2, 1))*self.scale                 
        atten_I1 = atten_I1.softmax(dim=-1)                        
        I_1 = torch.bmm(atten_I1, ok_value_)                                

        atten_I2 = torch.bmm(ok_q, hk_key_.permute(0, 2, 1))*self.scale                 
        atten_I2 = atten_I2.softmax(dim=-1)
        I_2 = torch.bmm(atten_I2, hk_value_)                              

        I_1 = self.layernorm(hk + I_1)                                 
        I_2 = self.layernorm(ok + I_2)    
        return I_1, I_2

class Self_Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):  
        super(Self_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads 
        assert self.head_dim * num_heads == hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()                             
        

        queries = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)    # (batch_size, num_heads, seq_len, head_dim)
        keys = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)         # (batch_size, num_heads, seq_len, head_dim)
        values = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)     # (batch_size, num_heads, seq_len, head_dim)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.hidden_size ** 0.5)                  # (batch_size, num_heads, seq_len, seq_len)

        attention_weights = nn.functional.softmax(scores, dim=-1)                                           # (batch_size, num_heads, seq_len, seq_len)   

        out = torch.matmul(attention_weights, values)                                                       # (batch_size, num_heads, seq_len, head_dim)  

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)                         # (batch_size, seq_len, embed_dim)

        out = self.ln(out + x) 
        return out

class Cross_Modal_Feature_Fusion(nn.Module):
    def __init__(self, emb_dim, proj_dim):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.cross_atten1 = Cross_Attention(emb_dim = self.emb_dim, proj_dim = self.proj_dim)


        self.fusion = nn.Sequential(
            nn.Conv1d(2*self.emb_dim, self.emb_dim, 1, 1),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()
    )
        self.fc = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim//2), 
            SwapAxes(),
            nn.BatchNorm1d(self.emb_dim // 2),
            nn.ReLU(),
            SwapAxes(),
            nn.Linear(self.emb_dim//2, self.emb_dim),
            SwapAxes(),
            nn.BatchNorm1d(self.emb_dim),
            SwapAxes(),
        )

        self.norm1 = nn.LayerNorm(self.emb_dim)
        self.norm2 = nn.LayerNorm(self.emb_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fusion = nn.Sequential(                                        
            nn.Conv1d(2*self.emb_dim, self.emb_dim, 1, 1),
            nn.BatchNorm1d(self.emb_dim),   
            nn.ReLU()        
        )
        
    def forward(self,f_t,f_p):
        _, N_P, _ = f_p.size()
        f_to, f_po = self.cross_atten1(f_t, f_p)            
        f_to = f_to + self.fc(f_to)                     
        f_po = f_po + self.fc(f_po)                    
        f_t_p = self.pool(f_to.permute(0,2,1))                 
        f_t_r = f_t_p.repeat(1, 1, N_P)               

        joint = torch.cat((f_po.permute(0,2,1), f_t_r), dim = 1)
        output = self.fusion(joint)   
        return output
        
class Point_Encoder(nn.Module):
    def __init__(self, emb_dim, normal_channel, additional_channel, N_p):
        super().__init__()

        self.N_p = N_p
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionMsg(self.N_p, [0.2,0.4], [16, 32], 256+256, [[128, 128, 256], [128, 196, 256]])

    def forward(self, xyz):

        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  #[B, 3, npoint_sa1] --- [B, 320, npoint_sa1]

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  #[B, 3, npoint_sa2] --- [B, 512, npoint_sa2]

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  #[B, 3, N_p]        --- [B, 512, N_p]

        return [[l0_xyz, l0_points], [l1_xyz, l1_points], [l2_xyz, l2_points], [l3_xyz, l3_points]]

class Img_Encoder(nn.Module):
    def __init__(self):
        super(Img_Encoder, self).__init__()

        self.model = models.resnet18(pretrained=False)
        self.model.relu = nn.ReLU()

    def forward(self, img):
        B, _, _, _ = img.size()
        out = self.model.conv1(img)
        out = self.model.relu(self.model.bn1(out))

        out = self.model.maxpool(out) 
        out = self.model.layer1(out)   
        down_1 = self.model.layer2(out)         
        down_2 = self.model.layer3(down_1)       
        down_3 = self.model.layer4(down_2)
       
        return down_3


class Text_Encoder(nn.Module):
    def __init__(self, emb_dim = 512, freeze_text_encoder = True):
        super().__init__()
        self.emb_dim = emb_dim

        self.text_encoder = AutoModel.from_pretrained('/PATH/roberta-base')
        self.tokenizer = AutoTokenizer.from_pretrained('/PATH/roberta-base')
        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)  
        self.text_resizer = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size, emb_dim, bias=True),
                                          nn.LayerNorm(emb_dim, eps=1e-12))
    
    def forward(self, text_queries):
        """
        input: text_queries 
        output:
            text_embedding:  [batch_size, num_phrases, emb_dim] 
            attention_mask:  [batch_size, num_phrases, seq_len] 
        """
        # Separate each input sentence by comma to get multiple phrases
        split_queries = [query.split(',') for query in text_queries]
        
        all_encoded_text = []

        for phrases in split_queries:
            tokenized_phrases = self.tokenizer.batch_encode_plus(phrases, padding='longest', return_tensors='pt')  
            tokenized_phrases = tokenized_phrases.to(device)
            outputs = self.text_encoder(**tokenized_phrases)
            pooled_output = outputs.pooler_output
            resized_phrases = self.text_resizer(pooled_output)
            all_encoded_text.append(resized_phrases)

        text_embeddings = torch.stack(all_encoded_text)  

        return text_embeddings


class Text_Encoder2(nn.Module):
    def __init__(self, emb_dim = 512, freeze_text_encoder = True):
        super().__init__()
        self.emb_dim = emb_dim

        self.text_encoder = AutoModel.from_pretrained('/PATH/roberta-base')
        self.tokenizer = AutoTokenizer.from_pretrained('/PATH/roberta-base')
        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)   
        self.text_resizer = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size, emb_dim, bias=True),
                                          nn.LayerNorm(emb_dim, eps=1e-12))
    
    def forward(self, text_queries):
        
        with torch.inference_mode(mode=self.freeze_text_encoder):
            tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding='longest', max_length=512, truncation=True, return_tensors='pt')  
        tokenized_queries = tokenized_queries.to(device)
        outputs = self.text_encoder(**tokenized_queries)
        pooled_output = outputs.pooler_output
        pooled_output = pooled_output.unsqueeze(1)
        return self.text_resizer(pooled_output)
    
class affordance_dictionary_fusion(nn.Module):
    def __init__(self, emb_dim = 512, proj_dim = 512, num_heads = 4):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.cross_atten = Cross_Attention(emb_dim = self.emb_dim, proj_dim = self.proj_dim)
        self.h_atten = Self_Attention(self.emb_dim, self.num_heads)
        self.o_atten = Self_Attention(self.emb_dim, self.num_heads)

    def forward(self,f_hk,f_ok):
        H, O = self.cross_atten(f_hk, f_ok)              
        H_= self.h_atten(H)
        O_= self.o_atten(O)
        return H_, O_

class img_text_fusion(nn.Module):
    def __init__(self, emb_dim = 512, proj_dim = 512):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        super().__init__()

        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.fusion = nn.Sequential(
            nn.Conv1d(2*self.emb_dim, self.emb_dim, 1, 1),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()
        )         
        self.reshape = nn.Sequential(
            nn.Linear(3, 3 * 8),
            SwapAxes(),
            nn.BatchNorm1d(3 * 8),
            nn.ReLU(),
            SwapAxes(),
            nn.Linear(3 * 8, 49),
        )                  
    def forward(self,F_i,T_h_):    
        T_h_ = self.reshape(T_h_.permute(0,2,1))  
        I_ = torch.cat((F_i, T_h_),dim=1)
        I_ = self.fusion(I_)  
        return I_
    
class Decoder(nn.Module):
    def __init__(self, additional_channel, emb_dim, proj_dim):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        super().__init__()
        
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        #upsample
        self.fp3 = PointNetFeaturePropagation(in_channel=512+self.emb_dim, mlp=[768, 512])   
        self.fp2 = PointNetFeaturePropagation(in_channel=832, mlp=[768, 512])  
        self.fp1 = PointNetFeaturePropagation(in_channel=518+additional_channel, mlp=[512, 512]) 

        self.cmff = Cross_Modal_Feature_Fusion(emb_dim, proj_dim)
        self.out_head = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 8),
            SwapAxes(),
            nn.BatchNorm1d(self.emb_dim // 8),
            nn.ReLU(),
            SwapAxes(),
            nn.Linear(self.emb_dim // 8, 1),
        )
        self.reshape = nn.Sequential(
            nn.Linear(49, 49 * 8),
            SwapAxes(),
            nn.BatchNorm1d(49 * 8),
            nn.ReLU(),
            SwapAxes(),
            nn.Linear(49 * 8, 2048),
        )          
        self.sigmoid = nn.Sigmoid()
        self.fusion = nn.Sequential(
            nn.Conv1d(2*self.emb_dim, self.emb_dim, 1, 1),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()
        )  
    def forward(self, T_o, I_h, encoder_p):

        '''
        T_o --->object knowledge embedding       
        I_h ---> [B, N_i, C]
        encoder_p  ---> [Hierarchy feature]
        '''
        B, _, _ = I_h.shape

        p_0, p_1, p_2, p_3 = encoder_p  

        p_3[1] = self.cmff(T_o, p_3[1].transpose(-2, -1))     
        up_sample = self.fp3(p_2[0], p_3[0], p_2[1], p_3[1])   
        

        up_sample = self.fp2(p_1[0], p_2[0], p_1[1], up_sample)    
        
       
        up_sample = self.fp1(p_0[0], p_1[0], torch.cat([p_0[0], p_0[1]],1), up_sample) 
        
        F_I = self.reshape(I_h.permute(0,2,1))  

        F_j = torch.cat((F_I, up_sample),dim=1)
        F_j_fusion = self.fusion(F_j)        

        _3daffordance = self.out_head(F_j_fusion.permute(0, 2, 1))                   
        _3daffordance = self.sigmoid(_3daffordance)

        return _3daffordance

class GREAT(nn.Module):
    def __init__(self, img_model_path=None, pre_train = True, normal_channel=False, local_rank=None,
                N_p = 64, emb_dim = 512, proj_dim = 512, num_heads = 4, freeze_text_encoder = True):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        super().__init__()

        self.emb_dim = emb_dim
        self.N_p = N_p
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.local_rank = local_rank
        self.normal_channel = normal_channel

        if self.normal_channel:
            self.additional_channel = 3
        else:
            self.additional_channel = 0

        self.img_encoder = Img_Encoder()
        if pre_train:
            pretrain_dict = torch.load(img_model_path)
            img_model_dict = self.img_encoder.state_dict()
            for k in list(pretrain_dict.keys()):
                new_key = 'model.' + k
                pretrain_dict[new_key] = pretrain_dict.pop(k)
            pretrain_dict={ k : v for k, v in pretrain_dict.items() if k in img_model_dict}
            img_model_dict.update(pretrain_dict)
            self.img_encoder.load_state_dict(img_model_dict)

        self.point_encoder = Point_Encoder(self.emb_dim, self.normal_channel, self.additional_channel, self.N_p)
        self.text_encoder = Text_Encoder(self.emb_dim, freeze_text_encoder = True)
        self.text_encoder2 = Text_Encoder2(self.emb_dim, freeze_text_encoder = True)

        self.affordance_dictionary_fusion  = affordance_dictionary_fusion(self.emb_dim, self.proj_dim, self.num_heads)
        self.img_text_fusion = img_text_fusion(self.emb_dim, self.proj_dim)
        self.decoder = Decoder(self.additional_channel, self.emb_dim, self.proj_dim)


    def forward(self, img, xyz, text_human, text_object):

        '''
        img: [B, 3, H, W]
        xyz: [B, 3, 2048]
        '''

        B, C, N = xyz.size()
        F_I = self.img_encoder(img)     
        F_i = F_I.view(B, self.emb_dim, -1)         

        F_p_wise = self.point_encoder(xyz)
        T_h= self.text_encoder(text_human)
        T_o = self.text_encoder2(text_object)

        T_h_, T_o_ =self.affordance_dictionary_fusion(T_h, T_o)     
        I_h = self.img_text_fusion(F_i,T_h_)         

        _3daffordance = self.decoder(T_o_, I_h.permute(0,2,1), F_p_wise)

        return _3daffordance


def get_GREAT(img_model_path=None, pre_train = True, normal_channel=False, local_rank=None,
    N_p = 64, emb_dim = 512, proj_dim = 512, num_heads = 4, freeze_text_encoder = True):
    
    model = GREAT(img_model_path, pre_train, normal_channel, local_rank,
    N_p, emb_dim, proj_dim, num_heads, freeze_text_encoder)
    return model


