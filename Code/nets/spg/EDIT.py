import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import random

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


# 初始化ema模型
def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

# 更新ema模型
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        # ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        
        # Ensure both model and EMA parameters are on the same device
        ema_param = ema_params[name]
        if ema_param.device != param.device:
            ema_param.data = ema_param.data.to(param.device)
        # Update EMA parameter
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

# -----------------------------dropout0.8+mask0.5+dropout_audio0.1+EMA
# SOTA model

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from collections import OrderedDict
from copy import deepcopy
import os
from nets.spg.models import DiT_models

class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
        super().__init__()
        # input_dim= 2048 ; dim= 256 ; n_layers= 15 ; n_classes= 4 ; bh_model= True
        if torch.cuda.device_count() > 1:
            local_rank = int(os.environ['LOCAL_RANK'])
            self.device = torch.device("cuda", local_rank)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = dim
        self.audio = audio
        self.bh_model = bh_model
        # dim=256
        if self.audio:
            self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)
            self.fusion_h = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        self.apply(weights_init)
        # 0.8概率归零
        self.dp_x = nn.Dropout(0.8)
        # self.dp_a = nn.Dropout(0.5)
        self.dp_a = nn.Dropout(0.1)
        #后面加入DiT初始化
        # latent_size = 32
        self.ViT_model = DiT_models["DiT-XL/2"](
            input_size=(22,22),
            num_classes=n_classes,
            in_channels=2,
            learn_sigma = False
        )
        # Note that parameter initialization is done within the DiT constructor
        self.ema = deepcopy(self.ViT_model).to(self.device)  # Create an EMA of the model for use after training
        requires_grad(self.ema, False)
        update_ema(self.ema, self.ViT_model, decay=0)  # Ensure EMA is initialized with synced weights
        self.ViT_model.train()  # important! This enables embedding dropout for classifier-free guidance
        self.ema.eval()  # EMA model should always be in eval mode
        self.latent_audio_encoder = nn.Conv2d(256, 22, 1, 1)
        self.latent_audio_decoder = nn.Conv2d(22, 256, 1, 1)
        print("using EDIT model!!!")

    # 随机mask手部和身体
    def mask_pose(self,cond,cond_drop_prob):
        bs= cond.shape[1] 
        if self.training: 
            mask = 1 - torch.bernoulli(torch.ones(bs*2, device=cond.device) * cond_drop_prob).view(bs, 2) 
            return cond * mask
        else: 
            return cond  

    def forward(self, x, label, aud):
        # 输入的x.shape: torch.Size([B, 22, 2])
        x = self.mask_pose(x,0.5).long()           # mask pose
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        # 处理后的x.shape: torch.Size([B, 256, 22, 2])
        x_train = x

        if self.audio is True:
            # 未处理过aud.shape: torch.Size([B, 256, 22, 2])
            aud = self.embedding_aud(aud)
            # 处理过的aud.shape: torch.Size([B, 256, 22, 2])
            dropout_x = torch.ones(x.shape[-2]).to(x.device)
            dropout_x = self.dp_x(dropout_x)
            x_train = (x_train.transpose(-1, -2) * dropout_x).transpose(-1, -2) # 以概率将pose归零

            dropout_a = torch.ones(aud.shape[-2]).to(aud.device)
            dropout_a = self.dp_a(dropout_a)
            aud = (aud.transpose(-1, -2) * dropout_a).transpose(-1, -2) # 以概率将audio归零
            # 按照维度1连接起来，所以变成了torch.Size([B, 512, 22, 2])
            
            if self.bh_model:
                x_train = self.fusion_h(torch.cat([x_train, aud], dim=1))
            label = label.to(self.device)        # 将label放到cuda:0上
            
            model_kwargs = dict(y = label)
            x_train = self.latent_audio_encoder(x_train)
            x_train = x_train.permute(0,3,1,2)

            if self.training:
                logits = self.ViT_model(x_train, **model_kwargs) 
            else:
                logits = self.ema(x_train, **model_kwargs)   
                # logits = self.ViT_model(x_train, **model_kwargs)   #  non-ema模型

            logits = logits.permute(0,2,3,1)   
            logits = self.latent_audio_decoder(logits)
            # logits.shape :(B, 256, 22, 2)
            if self.training:
                update_ema(self.ema, self.ViT_model)
            # 这里的logits应该要为torch.Size([B, 256, 22, 2])
            logits = self.output_conv(logits)
        
        # 输出的logits.shape: torch.Size([B, 2048, 22, 2])
        return logits

    def generate(self, label, shape=(8, 8), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None,edit = None):
        """
        batch_size: 1
        shape: [22, 2]
        aud_feat: torch.Size([1, 256, 22, 2])
        return : the x_predict
        """
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )# torch.Size([1, 22, 2]) zero
        infer_choice_position = [[0] * 2 for _ in range(22)]        # 用来标记选中的位置
        if edit is not None:
            x[:,:3,:2] = edit
            # 将列表转换为 LongTensor，并移动到 CUDA 设备
            new_values = torch.tensor([[15, 1507], [1991, 1760], [1657, 1344]], dtype=torch.long).to(x.device)
            # 执行赋值操作
            x[:,-3:, :2] = new_values
            for i in range(3):
                infer_choice_position[i] = [1, 1]
            for j in range(3):
                infer_choice_position[21-j] = [1, 1]
            
        h0 = 0
        h = shape[0] 

        # for i in range(h0, h):
        #     for j in range(shape[1]):
        #         logits = self.forward(x, label, aud_feat)
        #         probs = F.softmax(logits[:, :, i, j], -1)
        #         x.data[:, i, j].copy_(
        #             probs.multinomial(1).squeeze().data
        #         )   
           
        if pre_latents is not None:
            x[:, :pre_latents.size(1), :] = pre_latents.squeeze(0)
            for i in range(pre_latents.size(1)):
                for j in range(pre_latents.size(2)):
                    infer_choice_position[i][j] = 1


        condition_prob = 0.9
        for num in range(100):
            if condition_prob <= -0.2:          # 限制无用循环
                break
            logits = self.forward(x, label, aud_feat)   
            # ------------Classifier Free Guidance----------   
            aud_feat_zeros = torch.zeros_like(aud_feat)
            without_condition_logits = self.forward(x, label, aud_feat_zeros)  
            logits = without_condition_logits + (logits - without_condition_logits) * 7
            # ----------------------------------------------
            Get_flag = False
            for i in range(h0, h):
                for j in range(shape[1]):
                    if infer_choice_position[i][j] == 1:            #已经选择过，那么跳出
                        continue
                    probs = F.softmax(logits[:, :, i, j], -1)       # 计算概率
                    check_condition = torch.max(probs).item()    

                    if check_condition >= condition_prob:           # 只有大于一定概率的运动才会被选中
                        Get_flag = True
                        if infer_choice_position[i][j] == 0: 
                            infer_choice_position[i][j] = 1   
                            x.data[:, i, j].copy_(
                                probs.multinomial(1).squeeze().data
                            )
            if Get_flag == False:
                 condition_prob -= 0.1
        # print(x)
        return x[:, h0:h]




# Classifier-Free Diffusion Guidance train

# import torch
# # the first flag below was False when we tested this script but True makes A100 training a lot faster:
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# from collections import OrderedDict
# from copy import deepcopy
# import os
# from nets.spg.models import DiT_models

# class GatedPixelCNN(nn.Module):
#     def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
#         super().__init__()
#         # input_dim= 2048 ; dim= 256 ; n_layers= 15 ; n_classes= 4 ; bh_model= True
#         if torch.cuda.device_count() > 1:
#             local_rank = int(os.environ['LOCAL_RANK'])
#             self.device = torch.device("cuda", local_rank)
#         else:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.dim = dim
#         self.audio = audio
#         self.bh_model = bh_model
#         # dim=256
#         if self.audio:
#             self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)
#             self.fusion_h = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)

#         # Create embedding layer to embed input
#         self.embedding = nn.Embedding(input_dim, dim)

        
#         self.output_conv = nn.Sequential(
#             nn.Conv2d(dim, 512, 1),
#             nn.ReLU(True),
#             nn.Conv2d(512, input_dim, 1)
#         )

#         self.apply(weights_init)
#         # 0.8概率归零
#         self.dp_x = nn.Dropout(0.8)
#         # self.dp_a = nn.Dropout(0.5)
#         self.dp_a = nn.Dropout(0.1)
#         #后面加入DiT初始化
#         # latent_size = 32
#         self.ViT_model = DiT_models["DiT-XL/2"](
#             input_size=(22,22),
#             num_classes=n_classes,
#             in_channels=2,
#             learn_sigma = False
#         )
#         # Note that parameter initialization is done within the DiT constructor
#         self.ema = deepcopy(self.ViT_model).to(self.device)  # Create an EMA of the model for use after training
#         requires_grad(self.ema, False)
#         update_ema(self.ema, self.ViT_model, decay=0)  # Ensure EMA is initialized with synced weights
#         self.ViT_model.train()  # important! This enables embedding dropout for classifier-free guidance
#         self.ema.eval()  # EMA model should always be in eval mode
#         self.latent_audio_encoder = nn.Conv2d(256, 22, 1, 1)
#         self.latent_audio_decoder = nn.Conv2d(22, 256, 1, 1)
#         print("using EDIT model!!!")

#     # 随机mask手部和身体
#     def mask_pose(self,cond,cond_drop_prob):
#         bs= cond.shape[1] 
#         if self.training: 
#             mask = 1 - torch.bernoulli(torch.ones(bs*2, device=cond.device) * cond_drop_prob).view(bs, 2) 
#             return cond * mask
#         else: 
#             return cond  
#     def full_mask_audio(self,cond,cond_drop_prob):
#         if self.training: 
#             probability = random.random()
#             if probability < cond_drop_prob:
#                 return torch.zeros_like(cond)
#         return cond



#     def forward(self, x, label, aud):
#         # 输入的x.shape: torch.Size([B, 22, 2])
#         x = self.mask_pose(x,0.5).long()           # mask pose
#         aud = self.full_mask_audio(aud,0.1)
#         shp = x.size() + (-1,)
#         x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
#         x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
#         # 处理后的x.shape: torch.Size([B, 256, 22, 2])
#         x_train = x

#         if self.audio is True:
#             # 未处理过aud.shape: torch.Size([B, 256, 22, 2])
#             aud = self.embedding_aud(aud)
#             # 处理过的aud.shape: torch.Size([B, 256, 22, 2])
#             dropout_x = torch.ones(x.shape[-2]).to(x.device)
#             dropout_x = self.dp_x(dropout_x)
#             x_train = (x_train.transpose(-1, -2) * dropout_x).transpose(-1, -2) # 以概率将pose归零

#             dropout_a = torch.ones(aud.shape[-2]).to(aud.device)
#             dropout_a = self.dp_a(dropout_a)
#             aud = (aud.transpose(-1, -2) * dropout_a).transpose(-1, -2) # 以概率将audio归零
#             # 按照维度1连接起来，所以变成了torch.Size([B, 512, 22, 2])
            
#             if self.bh_model:
#                 x_train = self.fusion_h(torch.cat([x_train, aud], dim=1))
#             label = label.to(self.device)        # 将label放到cuda:0上
            
#             model_kwargs = dict(y = label)
#             x_train = self.latent_audio_encoder(x_train)
#             x_train = x_train.permute(0,3,1,2)

#             if self.training:
#                 logits = self.ViT_model(x_train, **model_kwargs) 
#             else:
#                 logits = self.ema(x_train, **model_kwargs)   
#                 # logits = self.ViT_model(x_train, **model_kwargs)   #  non-ema模型

#             logits = logits.permute(0,2,3,1)   
#             logits = self.latent_audio_decoder(logits)
#             # logits.shape :(B, 256, 22, 2)
#             if self.training:
#                 update_ema(self.ema, self.ViT_model)
#             # 这里的logits应该要为torch.Size([B, 256, 22, 2])
#             logits = self.output_conv(logits)
        
#         # 输出的logits.shape: torch.Size([B, 2048, 22, 2])
#         return logits

#     def generate(self, label, shape=(8, 8), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None,edit = None):
#         """
#         batch_size: 1
#         shape: [22, 2]
#         aud_feat: torch.Size([1, 256, 22, 2])
#         return : the x_predict
#         """
#         param = next(self.parameters())
#         x = torch.zeros(
#             (batch_size, *shape),
#             dtype=torch.int64, device=param.device
#         )
#         # 这里x = 0 是不行的，要用音频作为输入。
#         if edit is not None:
#             x = edit
#         h0 = 0
#         h = shape[0] 

#         # for i in range(h0, h):
#         #     for j in range(shape[1]):
#         #         logits = self.forward(x, label, aud_feat)
#         #         probs = F.softmax(logits[:, :, i, j], -1)
#         #         x.data[:, i, j].copy_(
#         #             probs.multinomial(1).squeeze().data
#         #         )   
            
#         infer_choice_position = [[0] * 2 for _ in range(22)]        # 用来标记选中的位置

#         if pre_latents is not None:
#             x[:, :pre_latents.size(1), :] = pre_latents.squeeze(0)
#             for i in range(pre_latents.size(1)):
#                 for j in range(pre_latents.size(2)):
#                     infer_choice_position[i][j] = 1


#         condition_prob = 0.9
#         for num in range(100):
#             if condition_prob <= -0.2:          # 限制无用循环
#                 break
#             logits = self.forward(x, label, aud_feat)   
#             # ------------Classifier Free Guidance----------   
#             # aud_feat_zeros = torch.zeros_like(aud_feat)
#             # without_condition_logits = self.forward(x, label, aud_feat_zeros)  
#             # logits = without_condition_logits + (logits - without_condition_logits) * 3
#             # ----------------------------------------------
#             Get_flag = False
#             for i in range(h0, h):
#                 for j in range(shape[1]):
#                     if infer_choice_position[i][j] == 1:            #已经选择过，那么跳出
#                         continue
#                     probs = F.softmax(logits[:, :, i, j], -1)       # 计算概率
#                     check_condition = torch.max(probs).item()    

#                     if check_condition >= condition_prob:           # 只有大于一定概率的运动才会被选中
#                         Get_flag = True
#                         if infer_choice_position[i][j] == 0: 
#                             infer_choice_position[i][j] = 1   
#                             x.data[:, i, j].copy_(
#                                 probs.multinomial(1).squeeze().data
#                             )
#             if Get_flag == False:
#                  condition_prob -= 0.1

#         return x[:, h0:h]


# Ablation 
# -----------------------------dropout0.9+mask0.5+dropout_audio0.1+EMA
# 

# import torch
# # the first flag below was False when we tested this script but True makes A100 training a lot faster:
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# from collections import OrderedDict
# from copy import deepcopy
# import os
# from nets.spg.models import DiT_models

# class GatedPixelCNN(nn.Module):
#     def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
#         super().__init__()
#         # input_dim= 2048 ; dim= 256 ; n_layers= 15 ; n_classes= 4 ; bh_model= True
#         if torch.cuda.device_count() > 1:
#             local_rank = int(os.environ['LOCAL_RANK'])
#             self.device = torch.device("cuda", local_rank)
#         else:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.dim = dim
#         self.audio = audio
#         self.bh_model = bh_model
#         # dim=256
#         if self.audio:
#             self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)
#             self.fusion_h = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)

#         # Create embedding layer to embed input
#         self.embedding = nn.Embedding(input_dim, dim)

        
#         self.output_conv = nn.Sequential(
#             nn.Conv2d(dim, 512, 1),
#             nn.ReLU(True),
#             nn.Conv2d(512, input_dim, 1)
#         )

#         self.apply(weights_init)
#         self.dp_x = nn.Dropout(0.9)
#         # self.dp_a = nn.Dropout(0.5)
#         self.dp_a = nn.Dropout(0.1)
#         #后面加入DiT初始化
#         # latent_size = 32
#         self.ViT_model = DiT_models["DiT-XL/2"](
#             input_size=(22,22),
#             num_classes=n_classes,
#             in_channels=2,
#             learn_sigma = False
#         )
#         # Note that parameter initialization is done within the DiT constructor
#         self.ema = deepcopy(self.ViT_model).to(self.device)  # Create an EMA of the model for use after training
#         requires_grad(self.ema, False)
#         update_ema(self.ema, self.ViT_model, decay=0)  # Ensure EMA is initialized with synced weights
#         self.ViT_model.train()  # important! This enables embedding dropout for classifier-free guidance
#         self.ema.eval()  # EMA model should always be in eval mode
#         self.latent_audio_encoder = nn.Conv2d(256, 22, 1, 1)
#         self.latent_audio_decoder = nn.Conv2d(22, 256, 1, 1)
#         print("using EDIT model!!!")

#     # 随机mask手部和身体
#     def mask_pose(self,cond,cond_drop_prob):
#         bs= cond.shape[1] 
#         if self.training: 
#             mask = 1 - torch.bernoulli(torch.ones(bs*2, device=cond.device) * cond_drop_prob).view(bs, 2) 
#             return cond * mask
#         else: 
#             return cond  

#     def forward(self, x, label, aud):
#         # 输入的x.shape: torch.Size([B, 22, 2])
#         x = self.mask_pose(x,0.5).long()           # mask pose
#         shp = x.size() + (-1,)
#         x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
#         x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
#         # 处理后的x.shape: torch.Size([B, 256, 22, 2])
#         x_train = x

#         if self.audio is True:
#             # 未处理过aud.shape: torch.Size([B, 256, 22, 2])
#             aud = self.embedding_aud(aud)
#             # 处理过的aud.shape: torch.Size([B, 256, 22, 2])
#             dropout_x = torch.ones(x.shape[-2]).to(x.device)
#             dropout_x = self.dp_x(dropout_x)
#             x_train = (x_train.transpose(-1, -2) * dropout_x).transpose(-1, -2) # 以概率将pose归零

#             dropout_a = torch.ones(aud.shape[-2]).to(aud.device)
#             dropout_a = self.dp_a(dropout_a)
#             aud = (aud.transpose(-1, -2) * dropout_a).transpose(-1, -2) # 以概率将audio归零
#             # 按照维度1连接起来，所以变成了torch.Size([B, 512, 22, 2])
            
#             if self.bh_model:
#                 x_train = self.fusion_h(torch.cat([x_train, aud], dim=1))
#             label = label.to(self.device)        # 将label放到cuda:0上
            
#             model_kwargs = dict(y = label)
#             x_train = self.latent_audio_encoder(x_train)
#             x_train = x_train.permute(0,3,1,2)

#             if self.training:
#                 logits = self.ViT_model(x_train, **model_kwargs) 
#             else:
#                 logits = self.ema(x_train, **model_kwargs)   
#                 # logits = self.ViT_model(x_train, **model_kwargs)   #  non-ema模型

#             logits = logits.permute(0,2,3,1)   
#             logits = self.latent_audio_decoder(logits)
#             # logits.shape :(B, 256, 22, 2)
#             if self.training:
#                 update_ema(self.ema, self.ViT_model)
#             # 这里的logits应该要为torch.Size([B, 256, 22, 2])

#         # 输出的logits.shape: torch.Size([B, 2048, 22, 2])
#         return self.output_conv(logits)

#     def generate(self, label, shape=(8, 8), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None,edit = None):
#         """
#         batch_size: 1
#         shape: [22, 2]
#         aud_feat: torch.Size([1, 256, 22, 2])
#         return : the x_predict
#         """
#         param = next(self.parameters())
#         x = torch.zeros(
#             (batch_size, *shape),
#             dtype=torch.int64, device=param.device
#         )
#         # 这里x = 0 是不行的，要用音频作为输入。
#         if edit is not None:
#             x = edit
#         h0 = 0
#         h = shape[0] 

#         # for i in range(h0, h):
#         #     for j in range(shape[1]):
#         #         logits = self.forward(x, label, aud_feat)
#         #         probs = F.softmax(logits[:, :, i, j], -1)
#         #         x.data[:, i, j].copy_(
#         #             probs.multinomial(1).squeeze().data
#         #         )   
            
#         infer_choice_position = [[0] * 2 for _ in range(22)]        # 用来标记选中的位置

#         if pre_latents is not None:
#             x[:, :pre_latents.size(1), :] = pre_latents.squeeze(0)
#             for i in range(pre_latents.size(1)):
#                 for j in range(pre_latents.size(2)):
#                     infer_choice_position[i][j] = 1


#         condition_prob = 0.9
#         for num in range(100):
#             if condition_prob <= -0.2:          # 限制无用循环
#                 break
#             logits = self.forward(x, label, aud_feat)      # 优化若干次
#             # ------------Classifier Free Guidance----------   
#             aud_feat_zeros = torch.zeros_like(aud_feat)
#             without_condition_logits = self.forward(x, label, aud_feat_zeros)  
#             logits = without_condition_logits + (logits - without_condition_logits) * 3
#             # ----------------------------------------------
#             Get_flag = False
#             for i in range(h0, h):
#                 for j in range(shape[1]):
#                     if infer_choice_position[i][j] == 1:            #已经选择过，那么跳出
#                         continue
#                     probs = F.softmax(logits[:, :, i, j], -1)       # 计算概率
#                     check_condition = torch.max(probs).item()    

#                     if check_condition >= condition_prob:           # 只有大于一定概率的运动才会被选中
#                         Get_flag = True
#                         if infer_choice_position[i][j] == 0: 
#                             infer_choice_position[i][j] = 1   
#                             x.data[:, i, j].copy_(
#                                 probs.multinomial(1).squeeze().data
#                             )
#             if Get_flag == False:
#                  condition_prob -= 0.1

#         return x[:, h0:h]




# --------------dropout0.8+ema策略--------------------

# import torch
# # the first flag below was False when we tested this script but True makes A100 training a lot faster:
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# from collections import OrderedDict
# from copy import deepcopy
# import os

# from nets.spg.models import DiT_models
# from nets.spg.diffusion import create_diffusion


# # def requires_grad(model, flag=True):
# #     """
# #     Set requires_grad flag for all parameters in a model.
# #     """
# #     for p in model.parameters():
# #         p.requires_grad = flag

# # @torch.no_grad()
# # def update_ema(ema_model, model, decay=0.9999):
# #     """
# #     Step the EMA model towards the current model.
# #     """
# #     ema_params = OrderedDict(ema_model.named_parameters())
# #     model_params = OrderedDict(model.named_parameters())

# #     for name, param in model_params.items():
# #         # # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
# #         # ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        
# #         # Ensure both model and EMA parameters are on the same device
# #         ema_param = ema_params[name]
# #         if ema_param.device != param.device:
# #             ema_param.data = ema_param.data.to(param.device)
# #         # Update EMA parameter
# #         ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

# class GatedPixelCNN(nn.Module):
#     def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
#         super().__init__()
#         # input_dim= 2048 ; dim= 256 ; n_layers= 15 ; n_classes= 4 ; bh_model= True
#         self.dim = dim
#         self.audio = audio
#         self.bh_model = bh_model
#         # dim=256
#         if self.audio:
#             self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)
#             self.fusion_v = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)
#             self.fusion_h = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)

#         # Create embedding layer to embed input
#         self.embedding = nn.Embedding(input_dim, dim)

#         # Add the output layer
#         self.output_conv = nn.Sequential(
#             nn.Conv2d(dim, 512, 1),
#             nn.ReLU(True),
#             nn.Conv2d(512, input_dim, 1)
#         )

#         self.apply(weights_init)
#         # 0.1概率归零
#         self.dp = nn.Dropout(0.1)
#         self.dp_x = nn.Dropout(0.8)

#         #后面加入DiT初始化
#         # latent_size = 32
#         self.ViT_model = DiT_models["DiT-XL/2"](
#             input_size=(22,22),
#             num_classes=n_classes,
#             in_channels=2,
#             learn_sigma = False
#         )
#         # Note that parameter initialization is done within the DiT constructor
#         if torch.cuda.device_count() > 1:
#             local_rank = int(os.environ['LOCAL_RANK'])
#             self.device = torch.device("cuda", local_rank)
#         else:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.ema = deepcopy(self.ViT_model).to(self.device)  # Create an EMA of the model for use after training
#         requires_grad(self.ema, False)
#         self.diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
#         update_ema(self.ema, self.ViT_model, decay=0)  # Ensure EMA is initialized with synced weights
#         self.ViT_model.train()  # important! This enables embedding dropout for classifier-free guidance
#         self.ema.eval()  # EMA model should always be in eval mode
#         self.latent_audio_encoder = nn.Conv2d(256, 22, 1, 1)
#         self.latent_audio_decoder = nn.Conv2d(22, 256, 1, 1)


#     def forward(self, x, label, aud):
#         # 输入的x.shape: torch.Size([B, 22, 2])
#         shp = x.size() + (-1,)
#         x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
#         x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
#         # 处理后的x.shape: torch.Size([B, 256, 22, 2])
#         x_v, x_h = (x, x)
#         # 15层的layers被遍历，每次都给到layer使用
#         if self.audio is True:
#             # 未处理过aud.shape: torch.Size([B, 256, 22, 2])
#             aud = self.embedding_aud(aud)
#             # 处理过的aud.shape: torch.Size([B, 256, 22, 2])
#             a = torch.ones(aud.shape[-2]).to(aud.device)
#             # a = self.dp(a)
#             a = self.dp_x(a)
#             x_h = (x_h.transpose(-1, -2) * a).transpose(-1, -2) # 以概率将pose归零
#             x_v = self.fusion_v(torch.cat([x_v, aud], dim=1))
#             # 按照维度1连接起来，所以变成了torch.Size([32, 512, 22, 2])
            
#             if self.bh_model:
#                 x_h = self.fusion_h(torch.cat([x_h, aud], dim=1))
        

#             label = label.to(self.device)        # 将label放到cuda:0上
#             # x_h.shape torch.Size([B, 256, 22, 2]) 

#             # 以下为DiT
#             t = torch.randint(0, self.diffusion.num_timesteps, (x_h.shape[0],), device=self.device)
#             model_kwargs = dict(y = label)

#             x_h = self.latent_audio_encoder(x_h)
#             x_h = x_h.permute(0,3,1,2)
            
#             # logits = self.diffusion.training_losses(self.DiT_model, x_h, t, model_kwargs)    # torch.Size([32, 2, casual, 22])   
#             if self.training:
#                 logits = self.ViT_model(x_h, t, **model_kwargs) 
#             #   delete
#             else:
#                 logits = self.ema(x_h, t, **model_kwargs)       # 推理的时候使用ema模型指标和生成效果都更好
#             # logits = self.ViT_model(x_h, t, **model_kwargs)  
#             logits = logits.permute(0,2,3,1)   
#             logits = self.latent_audio_decoder(logits)
#             # logits.shape :(B, 256, 22, 2)
#             if self.training:               # 训练的时候更新ema模型
#                 update_ema(self.ema, self.ViT_model)
            
#             # 这里的logits应该要为torch.Size([B, 256, 22, 2])
            

#         if self.bh_model:
#             return self.output_conv(logits)
#         else:
#             # print("输出的logits.shape:",self.output_conv(x_v).shape)
#             return self.output_conv(x_v)

#     def generate(self, label, shape=(8, 8), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None):
#         """
#         batch_size: 1
#         shape: [22, 2]
#         aud_feat: torch.Size([1, 256, 22, 2])
#         return : the x_predict
#         """
#         param = next(self.parameters())
#         x = torch.zeros(
#             (batch_size, *shape),
#             dtype=torch.int64, device=param.device
#         )
#         # 这里x = 0 是不行的，要用音频作为输入。
#         if pre_latents is not None:
#             x = torch.cat([pre_latents, x], dim=1)
#             aud_feat = torch.cat([pre_audio, aud_feat], dim=2)
#             h0 = pre_latents.shape[1]
#             h = h0 + shape[0]
#         else:
#             h0 = 0
#             h = shape[0]
        
#         # logits = self.forward(x, label, aud_feat)
#         # for i in range(h0, h):
#         #     for j in range(shape[1]):
#         #         probs = F.softmax(logits[:, :, i, j], -1)
#         #         x.data[:, i, j].copy_(
#         #             probs.multinomial(1).squeeze().data
#         #         )   

#         # for i in range(h0, h):
#         #     for j in range(shape[1]):
#         #         logits = self.forward(x, label, aud_feat)
#         #         probs = F.softmax(logits[:, :, i, j], -1)
#         #         x.data[:, i, j].copy_(
#         #             probs.multinomial(1).squeeze().data
#         #         ) 
              
#         infer_choice_position = [[0] * 2 for _ in range(22)]        # 用来标记选中的位置
#         condition_prob = 0.9
#         for num in range(44):
#             if condition_prob <= -0.2:          # 限制无用循环
#                 break
#             logits = self.forward(x, label, aud_feat)      # 优化若干次
#             Get_flag = False
#             for i in range(h0, h):
#                 for j in range(shape[1]):
#                     if infer_choice_position[i][j] == 1:            #已经选择过，那么跳出
#                         continue
#                     probs = F.softmax(logits[:, :, i, j], -1)       # 计算概率
#                     check_condition = torch.max(probs).item()             
#                     if check_condition >= condition_prob:           # 只有大于一定概率的运动才会被选中
#                         Get_flag = True
#                         if infer_choice_position[i][j] == 0: 
#                             infer_choice_position[i][j] = 1   
#                             x.data[:, i, j].copy_(
#                                 probs.multinomial(1).squeeze().data
#                             )
#             if Get_flag == False:
#                  condition_prob -= 0.1
 
#         return x[:, h0:h]

#--------------------dropoput 0.8+mask0.5+ w/o audio-dropout

# import torch
# # the first flag below was False when we tested this script but True makes A100 training a lot faster:
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# from collections import OrderedDict
# from copy import deepcopy
# import os
# from nets.spg.models import DiT_models

# class GatedPixelCNN(nn.Module):
#     def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
#         super().__init__()
#         # input_dim= 2048 ; dim= 256 ; n_layers= 15 ; n_classes= 4 ; bh_model= True
#         if torch.cuda.device_count() > 1:
#             local_rank = int(os.environ['LOCAL_RANK'])
#             self.device = torch.device("cuda", local_rank)
#         else:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.dim = dim
#         self.audio = audio
#         self.bh_model = bh_model
#         # dim=256
#         if self.audio:
#             self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)
#             self.fusion_h = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)

#         # Create embedding layer to embed input
#         self.embedding = nn.Embedding(input_dim, dim)

        
#         self.output_conv = nn.Sequential(
#             nn.Conv2d(dim, 512, 1),
#             nn.ReLU(True),
#             nn.Conv2d(512, input_dim, 1)
#         )

#         self.apply(weights_init)
#         # 0.8概率归零
#         self.dp_x = nn.Dropout(0.8)
#         #后面加入DiT初始化
#         # latent_size = 32
#         self.ViT_model = DiT_models["DiT-XL/2"](
#             input_size=(22,22),
#             num_classes=n_classes,
#             in_channels=2,
#             learn_sigma = False
#         )
#         # Note that parameter initialization is done within the DiT constructor
#         self.ema = deepcopy(self.ViT_model).to(self.device)  # Create an EMA of the model for use after training
#         requires_grad(self.ema, False)
#         update_ema(self.ema, self.ViT_model, decay=0)  # Ensure EMA is initialized with synced weights
#         self.ViT_model.train()  # important! This enables embedding dropout for classifier-free guidance
#         self.ema.eval()  # EMA model should always be in eval mode
#         self.latent_audio_encoder = nn.Conv2d(256, 22, 1, 1)
#         self.latent_audio_decoder = nn.Conv2d(22, 256, 1, 1)

#     # 随机mask手部和身体
#     def mask_pose(self,cond,cond_drop_prob):
#         bs= cond.shape[1] 
#         if self.training: 
#             mask = 1 - torch.bernoulli(torch.ones(bs*2, device=cond.device) * cond_drop_prob).view(bs, 2) 
#             return cond * mask
#         else: 
#             return cond  

#     def forward(self, x, label, aud):
#         # 输入的x.shape: torch.Size([B, 22, 2])
#         x = self.mask_pose(x,0.5).long()           # mask pose
#         shp = x.size() + (-1,)
#         x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
#         x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
#         # 处理后的x.shape: torch.Size([B, 256, 22, 2])
#         x_train = x

#         if self.audio is True:
#             # 未处理过aud.shape: torch.Size([B, 256, 22, 2])
#             aud = self.embedding_aud(aud)
#             # 处理过的aud.shape: torch.Size([B, 256, 22, 2])
#             a = torch.ones(aud.shape[-2]).to(aud.device)
#             # a = self.dp(a)
#             a = self.dp_x(a)
#             x_train = (x_train.transpose(-1, -2) * a).transpose(-1, -2) # 以概率将pose归零
#             # 按照维度1连接起来，所以变成了torch.Size([32, 512, 22, 2])
            
#             if self.bh_model:
#                 x_train = self.fusion_h(torch.cat([x_train, aud], dim=1))
#             label = label.to(self.device)        # 将label放到cuda:0上
#             model_kwargs = dict(y = label)
#             x_train = self.latent_audio_encoder(x_train)
#             x_train = x_train.permute(0,3,1,2)

#             if self.training:
#                 logits = self.ViT_model(x_train, **model_kwargs) 
#             else:
#               logits = self.ema(x_train, **model_kwargs)  
#               # logits = self.ViT_model(x_train, t, **model_kwargs)    

#             logits = logits.permute(0,2,3,1)   
#             logits = self.latent_audio_decoder(logits)
#             # logits.shape :(B, 256, 22, 2)
#             if self.training:
#                 update_ema(self.ema, self.ViT_model)
#             # 这里的logits应该要为torch.Size([B, 256, 22, 2])

#         # 输出的logits.shape: torch.Size([B, 2048, 22, 2])
#         return self.output_conv(logits)

#     def generate(self, label, shape=(8, 8), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None):
#         """
#         batch_size: 1
#         shape: [22, 2]
#         aud_feat: torch.Size([1, 256, 22, 2])
#         return : the x_predict
#         """
#         param = next(self.parameters())
#         x = torch.zeros(
#             (batch_size, *shape),
#             dtype=torch.int64, device=param.device
#         )
#         # 这里x = 0 是不行的，要用音频作为输入。
#         if pre_latents is not None:
#             x = torch.cat([pre_latents, x], dim=1)
#             aud_feat = torch.cat([pre_audio, aud_feat], dim=2)
#             h0 = pre_latents.shape[1]
#             h = h0 + shape[0]
#         else:
#             h0 = 0
#             h = shape[0] 

#         # for i in range(h0, h):
#         #     for j in range(shape[1]):
#         #         logits = self.forward(x, label, aud_feat)
#         #         probs = F.softmax(logits[:, :, i, j], -1)
#         #         x.data[:, i, j].copy_(
#         #             probs.multinomial(1).squeeze().data
#         #         )  
             
#         infer_choice_position = [[0] * 2 for _ in range(22)]        # 用来标记选中的位置
#         condition_prob = 0.9
#         for num in range(100):
#             if condition_prob <= -0.2:          # 限制无用循环
#                 break
#             logits = self.forward(x, label, aud_feat)      # 优化若干次
#             # ----
#             # aud_feat_zeros = torch.zeros_like(aud_feat)
#             # without_condition_logits = self.forward(x, label, aud_feat_zeros)  
#             # logits = without_condition_logits + (logits - without_condition_logits) * 3
#             # ----
#             Get_flag = False
#             for i in range(h0, h):
#                 for j in range(shape[1]):
#                     if infer_choice_position[i][j] == 1:            #已经选择过，那么跳出
#                         continue
#                     probs = F.softmax(logits[:, :, i, j], -1)       # 计算概率
#                     check_condition = torch.max(probs).item()    

#                     if check_condition >= condition_prob:           # 只有大于一定概率的运动才会被选中
#                         Get_flag = True
#                         if infer_choice_position[i][j] == 0: 
#                             infer_choice_position[i][j] = 1   
#                             x.data[:, i, j].copy_(
#                                 probs.multinomial(1).squeeze().data
#                             )
#             if Get_flag == False:
#                  condition_prob -= 0.1
            
        

#         return x[:, h0:h]




        
#----------------------------------audiod+mask+ema+cross attention

# import torch
# # the first flag below was False when we tested this script but True makes A100 training a lot faster:
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# from collections import OrderedDict
# from copy import deepcopy
# import os
# from nets.spg.models import DiT_models

# class CrossAttention(nn.Module):  
#     def __init__(self, embed_size, heads):  
#         super(CrossAttention, self).__init__()  
#         self.embed_size = embed_size  
#         self.heads = heads  
#         self.head_dim = embed_size // heads  
  
#         assert (  
#             self.head_dim * heads == embed_size  
#         ), "Embedding size needs to be divisible by the number of heads"  
  
#         self.values_linear = nn.Linear(embed_size, embed_size, bias=False)  
#         self.keys_linear = nn.Linear(embed_size, embed_size, bias=False)  
#         self.queries_linear = nn.Linear(embed_size, embed_size, bias=False)  
#         self.final_linear = nn.Linear(heads * self.head_dim, embed_size)  
  
#     def forward(self, values, keys, query, mask=None):  
#         N = query.shape[0]  
#         value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]  
  
#         # Linear projections  
#         values = self.values_linear(values)  
#         keys = self.keys_linear(keys)  
#         queries = self.queries_linear(query)  
  
#         # Reshape for multi-head attention  
#         values = values.reshape(N, value_len, self.heads, self.head_dim)  
#         keys = keys.reshape(N, key_len, self.heads, self.head_dim)  
#         queries = queries.reshape(N, query_len, self.heads, self.head_dim)  
  
#         # Scaled dot-product attention  
#         energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  
#         if mask is not None:  
#             energy = energy.masked_fill(mask == 0, float("-1e20"))  
  
#         attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  
  
#         out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(  
#             N, query_len, self.heads * self.head_dim  
#         )  
  
#         out = self.final_linear(out)  
#         return out  



# class GatedPixelCNN(nn.Module):
#     def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
#         super().__init__()
#         # input_dim= 2048 ; dim= 256 ; n_layers= 15 ; n_classes= 4 ; bh_model= True
#         if torch.cuda.device_count() > 1:
#             local_rank = int(os.environ['LOCAL_RANK'])
#             self.device = torch.device("cuda", local_rank)
#         else:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.dim = dim
#         self.audio = audio
#         self.bh_model = bh_model
#         # dim=256
#         self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)
#         # self.fusion_h = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)

#         # Create embedding layer to embed input
#         self.embedding = nn.Embedding(input_dim, dim)

        
#         self.output_conv = nn.Sequential(
#             nn.Conv2d(dim, 512, 1),
#             nn.ReLU(True),
#             nn.Conv2d(512, input_dim, 1)
#         )

#         self.apply(weights_init)
#         # 0.8概率归零
#         self.dp_x = nn.Dropout(0.8)
#         # self.dp_a = nn.Dropout(0.5)
#         self.dp_a = nn.Dropout(0.1)
#         #后面加入DiT初始化
#         # latent_size = 32
#         self.ViT_model = DiT_models["DiT-XL/2"](
#             input_size=(22,22),
#             num_classes=n_classes,
#             in_channels=2,
#             learn_sigma = False
#         )
#         # Note that parameter initialization is done within the DiT constructor
#         self.ema = deepcopy(self.ViT_model).to(self.device)  # Create an EMA of the model for use after training
#         requires_grad(self.ema, False)
#         update_ema(self.ema, self.ViT_model, decay=0)  # Ensure EMA is initialized with synced weights
#         self.ViT_model.train()  # important! This enables embedding dropout for classifier-free guidance
#         self.ema.eval()  # EMA model should always be in eval mode
#         self.latent_audio_encoder = nn.Conv2d(256, 22, 1, 1)
#         self.latent_audio_decoder = nn.Conv2d(22, 256, 1, 1)
#         self.cross_attn = CrossAttention(256, 8)  # cross attention

#     # 随机mask手部和身体
#     def mask_pose(self,cond,cond_drop_prob):
#         bs= cond.shape[1] 
#         if self.training: 
#             mask = 1 - torch.bernoulli(torch.ones(bs*2, device=cond.device) * cond_drop_prob).view(bs, 2) 
#             return cond * mask
#         else: 
#             return cond  

#     def forward(self, x, label, aud):
#         # 输入的x.shape: torch.Size([B, 22, 2])
#         x = self.mask_pose(x,0.5).long()           # mask pose
#         shp = x.size() + (-1,)
#         x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
#         x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
#         # 处理后的x.shape: torch.Size([B, 256, 22, 2])
#         x_train = x

#         if self.audio is True:
#             # 未处理过aud.shape: torch.Size([B, 256, 22, 2])
#             aud = self.embedding_aud(aud)
#             # 处理过的aud.shape: torch.Size([B, 256, 22, 2])
#             dropout_x = torch.ones(x.shape[-2]).to(x.device)
#             dropout_x = self.dp_x(dropout_x)
#             x_train = (x_train.transpose(-1, -2) * dropout_x).transpose(-1, -2) # 以概率将pose归零

#             dropout_a = torch.ones(aud.shape[-2]).to(aud.device)
#             dropout_a = self.dp_a(dropout_a)
#             aud = (aud.transpose(-1, -2) * dropout_a).transpose(-1, -2) # 以概率将audio归零
#             # 按照维度1连接起来，所以变成了torch.Size([B, 512, 22, 2])
            
#             # x_train = self.fusion_h(torch.cat([x_train, aud], dim=1))
#             # cross attention
#             query = aud.view(x_train.shape[0], x_train.shape[1], -1).transpose(-1,-2).contiguous()  # 44个查询token  
#             values = keys = x_train.view(x_train.shape[0], x_train.shape[1], -1).transpose(-1,-2).contiguous()
#             x_train = self.cross_attn(values, keys, query).transpose(-1,-2).view(x_train.shape[0], x_train.shape[1], 22, 2).contiguous()
#             # cross attention
            
#             label = label.to(self.device)        # 将label放到cuda:0上
            
#             model_kwargs = dict(y = label)
#             x_train = self.latent_audio_encoder(x_train)
#             x_train = x_train.permute(0,3,1,2)

#             if self.training:
#                 logits = self.ViT_model(x_train, **model_kwargs) 
#             else:
#                 logits = self.ema(x_train, **model_kwargs)     

#             logits = logits.permute(0,2,3,1)   
#             logits = self.latent_audio_decoder(logits)
#             # logits.shape :(B, 256, 22, 2)
#             if self.training:
#                 update_ema(self.ema, self.ViT_model)
#             # 这里的logits应该要为torch.Size([B, 256, 22, 2])

#         # 输出的logits.shape: torch.Size([B, 2048, 22, 2])
#         return self.output_conv(logits)

#     def generate(self, label, shape=(8, 8), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None):
#         """
#         batch_size: 1
#         shape: [22, 2]
#         aud_feat: torch.Size([1, 256, 22, 2])
#         return : the x_predict
#         """
#         param = next(self.parameters())
#         x = torch.zeros(
#             (batch_size, *shape),
#             dtype=torch.int64, device=param.device
#         )
#         # 这里x = 0 是不行的，要用音频作为输入。
#         if pre_latents is not None:
#             x = torch.cat([pre_latents, x], dim=1)
#             aud_feat = torch.cat([pre_audio, aud_feat], dim=2)
#             h0 = pre_latents.shape[1]
#             h = h0 + shape[0]
#         else:
#             h0 = 0
#             h = shape[0] 

#         # for i in range(h0, h):
#         #     for j in range(shape[1]):
#         #         logits = self.forward(x, label, aud_feat)
#         #         probs = F.softmax(logits[:, :, i, j], -1)
#         #         x.data[:, i, j].copy_(
#         #             probs.multinomial(1).squeeze().data
#         #         )   
            
#         infer_choice_position = [[0] * 2 for _ in range(22)]        # 用来标记选中的位置
#         condition_prob = 0.9
#         for num in range(100):
#             if condition_prob <= -0.2:          # 限制无用循环
#                 break
#             logits = self.forward(x, label, aud_feat)      # 优化若干次
#             Get_flag = False
#             for i in range(h0, h):
#                 for j in range(shape[1]):
#                     if infer_choice_position[i][j] == 1:            #已经选择过，那么跳出
#                         continue
#                     probs = F.softmax(logits[:, :, i, j], -1)       # 计算概率
#                     check_condition = torch.max(probs).item()    

#                     if check_condition >= condition_prob:           # 只有大于一定概率的运动才会被选中
#                         Get_flag = True
#                         if infer_choice_position[i][j] == 0: 
#                             infer_choice_position[i][j] = 1   
#                             x.data[:, i, j].copy_(
#                                 probs.multinomial(1).squeeze().data
#                             )
#             if Get_flag == False:
#                  condition_prob -= 0.1

#         return x[:, h0:h]


# ------------------------------non-ed Ablation
# import torch
# # the first flag below was False when we tested this script but True makes A100 training a lot faster:
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# from collections import OrderedDict
# from copy import deepcopy
# import os
# from nets.spg.models import DiT_models

# class GatedPixelCNN(nn.Module):
#     def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
#         super().__init__()
#         # input_dim= 2048 ; dim= 256 ; n_layers= 15 ; n_classes= 4 ; bh_model= True
#         if torch.cuda.device_count() > 1:
#             local_rank = int(os.environ['LOCAL_RANK'])
#             self.device = torch.device("cuda", local_rank)
#         else:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.dim = dim
#         self.audio = audio
#         self.bh_model = bh_model
#         # dim=256
#         if self.audio:
#             self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)
#             self.fusion_h = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)

#         # Create embedding layer to embed input
#         self.embedding = nn.Embedding(input_dim, dim)

        
#         self.output_conv = nn.Sequential(
#             nn.Conv2d(dim, 512, 1),
#             nn.ReLU(True),
#             nn.Conv2d(512, input_dim, 1)
#         )

#         self.apply(weights_init)
#         # 0.8概率归零
#         # self.dp_x = nn.Dropout(0.8)
#         self.dp_a = nn.Dropout(0.1)
#         #后面加入DiT初始化
#         # latent_size = 32
#         self.ViT_model = DiT_models["DiT-XL/2"](
#             input_size=(22,22),
#             num_classes=n_classes,
#             in_channels=2,
#             learn_sigma = False
#         )
#         # Note that parameter initialization is done within the DiT constructor
#         self.ema = deepcopy(self.ViT_model).to(self.device)  # Create an EMA of the model for use after training
#         requires_grad(self.ema, False)
#         update_ema(self.ema, self.ViT_model, decay=0)  # Ensure EMA is initialized with synced weights
#         self.ViT_model.train()  # important! This enables embedding dropout for classifier-free guidance
#         self.ema.eval()  # EMA model should always be in eval mode
#         self.latent_audio_encoder = nn.Conv2d(256, 22, 1, 1)
#         self.latent_audio_decoder = nn.Conv2d(22, 256, 1, 1)

#     # 随机mask手部和身体
#     def mask_pose(self,cond,cond_drop_prob):
#         bs= cond.shape[1] 
#         if self.training: 
#             mask = 1 - torch.bernoulli(torch.ones(bs*2, device=cond.device) * cond_drop_prob).view(bs, 2) 
#             return cond * mask
#         else: 
#             return cond  

#     def forward(self, x, label, aud):
#         # 输入的x.shape: torch.Size([B, 22, 2])
#         x = self.mask_pose(x,0.5).long()           # mask pose
#         shp = x.size() + (-1,)
#         x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
#         x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
#         # 处理后的x.shape: torch.Size([B, 256, 22, 2])
#         x_train = x

#         if self.audio is True:
#             # 未处理过aud.shape: torch.Size([B, 256, 22, 2])
#             aud = self.embedding_aud(aud)
#             # 处理过的aud.shape: torch.Size([B, 256, 22, 2])
#             # dropout_x = torch.ones(x.shape[-2]).to(x.device)
#             # dropout_x = self.dp_x(dropout_x)
#             # x_train = (x_train.transpose(-1, -2) * dropout_x).transpose(-1, -2) # 以概率将pose归零

#             dropout_a = torch.ones(aud.shape[-2]).to(aud.device)
#             dropout_a = self.dp_a(dropout_a)
#             aud = (aud.transpose(-1, -2) * dropout_a).transpose(-1, -2) # 以概率将audio归零
#             # 按照维度1连接起来，所以变成了torch.Size([B, 512, 22, 2])
            
#             if self.bh_model:
#                 x_train = self.fusion_h(torch.cat([x_train, aud], dim=1))
#             label = label.to(self.device)        # 将label放到cuda:0上
            
#             model_kwargs = dict(y = label)
#             x_train = self.latent_audio_encoder(x_train)
#             x_train = x_train.permute(0,3,1,2)

#             if self.training:
#                 logits = self.ViT_model(x_train, **model_kwargs) 
#             else:
#                 logits = self.ema(x_train, **model_kwargs)     

#             logits = logits.permute(0,2,3,1)   
#             logits = self.latent_audio_decoder(logits)
#             # logits.shape :(B, 256, 22, 2)
#             if self.training:
#                 update_ema(self.ema, self.ViT_model)
#             # 这里的logits应该要为torch.Size([B, 256, 22, 2])

#         # 输出的logits.shape: torch.Size([B, 2048, 22, 2])
#         return self.output_conv(logits)

#     def generate(self, label, shape=(8, 8), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None,edit = None):
#         """
#         batch_size: 1
#         shape: [22, 2]
#         aud_feat: torch.Size([1, 256, 22, 2])
#         return : the x_predict
#         """
#         param = next(self.parameters())
#         x = torch.zeros(
#             (batch_size, *shape),
#             dtype=torch.int64, device=param.device
#         )
#         # 这里x = 0 是不行的，要用音频作为输入。
#         if edit is not None:    # 编辑
#             x = edit
#         h0 = 0
#         h = shape[0] 

#         # for i in range(h0, h):
#         #     for j in range(shape[1]):
#         #         logits = self.forward(x, label, aud_feat)
#         #         probs = F.softmax(logits[:, :, i, j], -1)
#         #         x.data[:, i, j].copy_(
#         #             probs.multinomial(1).squeeze().data
#         #         )   

#         infer_choice_position = [[0] * 2 for _ in range(22)]        # 用来标记选中的位置

#         if pre_latents is not None:
#             x[:, :pre_latents.size(1), :] = pre_latents.squeeze(0)
#             for i in range(pre_latents.size(1)):
#                 for j in range(pre_latents.size(2)):
#                     infer_choice_position[i][j] = 1

#         condition_prob = 0.9
#         for num in range(100):
#             if condition_prob <= -0.2:          # 限制无用循环
#                 break
#             logits = self.forward(x, label, aud_feat)      # 优化若干次
#             Get_flag = False
#             for i in range(h0, h):
#                 for j in range(shape[1]):
#                     if infer_choice_position[i][j] == 1:            #已经选择过，那么跳出
#                         continue
#                     probs = F.softmax(logits[:, :, i, j], -1)       # 计算概率
#                     check_condition = torch.max(probs).item()    

#                     if check_condition >= condition_prob:           # 只有大于一定概率的运动才会被选中
#                         Get_flag = True
#                         if infer_choice_position[i][j] == 0: 
#                             infer_choice_position[i][j] = 1   
#                             x.data[:, i, j].copy_(
#                                 probs.multinomial(1).squeeze().data
#                             )
#             if Get_flag == False:
#                  condition_prob -= 0.1

#         return x[:, h0:h]


# ---------------------------------audio_drop0.1+mask+ face
    
# import torch
# # the first flag below was False when we tested this script but True makes A100 training a lot faster:
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# from collections import OrderedDict
# from copy import deepcopy
# import os
# from nets.spg.models import DiT_models

# class GatedPixelCNN(nn.Module):
#     def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
#         super().__init__()
#         # input_dim= 2048 ; dim= 256 ; n_layers= 15 ; n_classes= 4 ; bh_model= True
#         if torch.cuda.device_count() > 1:
#             local_rank = int(os.environ['LOCAL_RANK'])
#             self.device = torch.device("cuda", local_rank)
#         else:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.dim = dim
#         self.audio = audio
#         self.bh_model = bh_model
#         # dim=256
#         if self.audio:
#             self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)
#             self.fusion_h = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)

#         # Create embedding layer to embed input
#         self.embedding = nn.Embedding(input_dim, dim)

        
#         self.output_conv = nn.Sequential(
#             nn.Conv2d(dim, 512, 1),
#             nn.ReLU(True),
#             nn.Conv2d(512, input_dim, 1)
#         )

#         self.apply(weights_init)
#         # 0.8概率归零
#         self.dp_x = nn.Dropout(0.8)
#         # self.dp_a = nn.Dropout(0.5)
#         self.dp_a = nn.Dropout(0.1)
#         #后面加入DiT初始化
#         # latent_size = 32
#         self.ViT_model = DiT_models["DiT-XL/2"](
#             input_size=(22,22),
#             num_classes=n_classes,
#             in_channels=2,
#             learn_sigma = False
#         )
#         # Note that parameter initialization is done within the DiT constructor
#         self.ema = deepcopy(self.ViT_model).to(self.device)  # Create an EMA of the model for use after training
#         requires_grad(self.ema, False)
#         update_ema(self.ema, self.ViT_model, decay=0)  # Ensure EMA is initialized with synced weights
#         self.ViT_model.train()  # important! This enables embedding dropout for classifier-free guidance
#         self.ema.eval()  # EMA model should always be in eval mode
#         self.latent_audio_encoder = nn.Conv2d(256, 22, 1, 1)
#         self.latent_audio_decoder = nn.Conv2d(22, 256, 1, 1)

#     # 随机mask手部和身体
#     def mask_pose(self,cond,cond_drop_prob):
#         bs= cond.shape[1] 
#         if self.training: 
#             mask = 1 - torch.bernoulli(torch.ones(bs*2, device=cond.device) * cond_drop_prob).view(bs, 2) 
#             return cond * mask
#         else: 
#             return cond  

#     def forward(self, x, label, aud, expression):
#         # 输入的x.shape: torch.Size([B, 22, 2])
#         x = self.mask_pose(x,0.5).long()           # mask pose
#         shp = x.size() + (-1,)
#         x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
#         x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
#         # 处理后的x.shape: torch.Size([B, 256, 22, 2])
#         x_train = x
#         if self.audio is True:
#             # 未处理过aud.shape: torch.Size([B, 256, 22, 2])
#             aud = self.embedding_aud(aud)
#             # 处理过的aud.shape: torch.Size([B, 256, 22, 2])
#             dropout_x = torch.ones(x.shape[-2]).to(x.device)
#             dropout_x = self.dp_x(dropout_x)
#             x_train = (x_train.transpose(-1, -2) * dropout_x).transpose(-1, -2) # 以概率将pose归零

#             dropout_a = torch.ones(aud.shape[-2]).to(aud.device)
#             dropout_a = self.dp_a(dropout_a)
#             aud = (aud.transpose(-1, -2) * dropout_a).transpose(-1, -2) # 以概率将audio归零
#             # 按照维度1连接起来，所以变成了torch.Size([32, 512, 22, 2])
            
#             if self.bh_model:
#                 x_train = self.fusion_h(torch.cat([x_train, aud], dim=1))
#             label = label.to(self.device)        # 将label放到cuda:0上
#             model_kwargs = dict(y = label,expression = expression)
#             x_train = self.latent_audio_encoder(x_train)
#             x_train = x_train.permute(0,3,1,2)

#             if self.training:
#                 logits = self.ViT_model(x_train, **model_kwargs) 
#             else:
#                 logits = self.ema(x_train, **model_kwargs)     

#             logits = logits.permute(0,2,3,1)   
#             logits = self.latent_audio_decoder(logits)
#             # logits.shape :(B, 256, 22, 2)
#             if self.training:
#                 update_ema(self.ema, self.ViT_model)
#             # 这里的logits应该要为torch.Size([B, 256, 22, 2])

#         # 输出的logits.shape: torch.Size([B, 2048, 22, 2])
#         return self.output_conv(logits)

#     def generate(self, label, shape=(8, 8), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None,**kwargs):
#         """
#         batch_size: 1
#         shape: [22, 2]
#         aud_feat: torch.Size([1, 256, 22, 2])
#         return : the x_predict
#         """
#         generate_expression = kwargs['generate_expression'].unsqueeze(0)
#         param = next(self.parameters())
#         x = torch.zeros(
#             (batch_size, *shape),
#             dtype=torch.int64, device=param.device
#         )
#         # 这里x = 0 是不行的，要用音频作为输入。
#         if pre_latents is not None:
#             x = torch.cat([pre_latents, x], dim=1)
#             aud_feat = torch.cat([pre_audio, aud_feat], dim=2)
#             h0 = pre_latents.shape[1]
#             h = h0 + shape[0]
#         else:
#             h0 = 0
#             h = shape[0] 

#         for i in range(h0, h):
#             for j in range(shape[1]):
#                 logits = self.forward(x, label, aud_feat, generate_expression)
#                 probs = F.softmax(logits[:, :, i, j], -1)
#                 x.data[:, i, j].copy_(
#                     probs.multinomial(1).squeeze().data
#                 )   
            
#         infer_choice_position = [[0] * 2 for _ in range(22)]        # 用来标记选中的位置
#         condition_prob = 0.9
#         for num in range(100):
#             if condition_prob <= -0.2:          # 限制无用循环
#                 break
#             logits = self.forward(x, label, aud_feat, generate_expression)      # 优化若干次
#             Get_flag = False
#             for i in range(h0, h):
#                 for j in range(shape[1]):
#                     if infer_choice_position[i][j] == 1:            #已经选择过，那么跳出
#                         continue
#                     probs = F.softmax(logits[:, :, i, j], -1)       # 计算概率
#                     check_condition = torch.max(probs).item()    

#                     if check_condition >= condition_prob:           # 只有大于一定概率的运动才会被选中
#                         Get_flag = True
#                         if infer_choice_position[i][j] == 0: 
#                             infer_choice_position[i][j] = 1   
#                             x.data[:, i, j].copy_(
#                                 probs.multinomial(1).squeeze().data
#                             )
#             if Get_flag == False:
#                  condition_prob -= 0.1
                 
#         return x[:, h0:h]

# non-masking Ablation
    
# import torch
# # the first flag below was False when we tested this script but True makes A100 training a lot faster:
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# from collections import OrderedDict
# from copy import deepcopy
# import os
# from nets.spg.models import DiT_models

# class GatedPixelCNN(nn.Module):
#     def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
#         super().__init__()
#         # input_dim= 2048 ; dim= 256 ; n_layers= 15 ; n_classes= 4 ; bh_model= True
#         if torch.cuda.device_count() > 1:
#             local_rank = int(os.environ['LOCAL_RANK'])
#             self.device = torch.device("cuda", local_rank)
#         else:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.dim = dim
#         self.audio = audio
#         self.bh_model = bh_model
#         # dim=256
#         if self.audio:
#             self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)
#             self.fusion_h = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)

#         # Create embedding layer to embed input
#         self.embedding = nn.Embedding(input_dim, dim)

        
#         self.output_conv = nn.Sequential(
#             nn.Conv2d(dim, 512, 1),
#             nn.ReLU(True),
#             nn.Conv2d(512, input_dim, 1)
#         )

#         self.apply(weights_init)
#         # 0.8概率归零
#         self.dp_x = nn.Dropout(0.8)
#         # self.dp_a = nn.Dropout(0.5)
#         self.dp_a = nn.Dropout(0.1)
#         #后面加入DiT初始化
#         # latent_size = 32
#         self.ViT_model = DiT_models["DiT-XL/2"](
#             input_size=(22,22),
#             num_classes=n_classes,
#             in_channels=2,
#             learn_sigma = False
#         )
#         # Note that parameter initialization is done within the DiT constructor
#         self.ema = deepcopy(self.ViT_model).to(self.device)  # Create an EMA of the model for use after training
#         requires_grad(self.ema, False)
#         update_ema(self.ema, self.ViT_model, decay=0)  # Ensure EMA is initialized with synced weights
#         self.ViT_model.train()  # important! This enables embedding dropout for classifier-free guidance
#         self.ema.eval()  # EMA model should always be in eval mode
#         self.latent_audio_encoder = nn.Conv2d(256, 22, 1, 1)
#         self.latent_audio_decoder = nn.Conv2d(22, 256, 1, 1)

#     # 随机mask手部和身体
#     def mask_pose(self,cond,cond_drop_prob):
#         bs= cond.shape[1] 
#         if self.training: 
#             mask = 1 - torch.bernoulli(torch.ones(bs*2, device=cond.device) * cond_drop_prob).view(bs, 2) 
#             return cond * mask
#         else: 
#             return cond  

#     def forward(self, x, label, aud):
#         # 输入的x.shape: torch.Size([B, 22, 2])
#         # x = self.mask_pose(x,0.5).long()           # mask pose
#         shp = x.size() + (-1,)
#         x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
#         x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
#         # 处理后的x.shape: torch.Size([B, 256, 22, 2])
#         x_train = x

#         if self.audio is True:
#             # 未处理过aud.shape: torch.Size([B, 256, 22, 2])
#             aud = self.embedding_aud(aud)
#             # 处理过的aud.shape: torch.Size([B, 256, 22, 2])
#             dropout_x = torch.ones(x.shape[-2]).to(x.device)
#             dropout_x = self.dp_x(dropout_x)
#             x_train = (x_train.transpose(-1, -2) * dropout_x).transpose(-1, -2) # 以概率将pose归零

#             dropout_a = torch.ones(aud.shape[-2]).to(aud.device)
#             dropout_a = self.dp_a(dropout_a)
#             aud = (aud.transpose(-1, -2) * dropout_a).transpose(-1, -2) # 以概率将audio归零
#             # 按照维度1连接起来，所以变成了torch.Size([32, 512, 22, 2])
            
#             if self.bh_model:
#                 x_train = self.fusion_h(torch.cat([x_train, aud], dim=1))
#             label = label.to(self.device)        # 将label放到cuda:0上
            
#             model_kwargs = dict(y = label)
#             x_train = self.latent_audio_encoder(x_train)
#             x_train = x_train.permute(0,3,1,2)

#             if self.training:
#                 logits = self.ViT_model(x_train, **model_kwargs) 
#             else:
#                 logits = self.ema(x_train, **model_kwargs)   
#                 # logits = self.ViT_model(x_train, **model_kwargs)   #  non-ema模型

#             logits = logits.permute(0,2,3,1)   
#             logits = self.latent_audio_decoder(logits)
#             # logits.shape :(B, 256, 22, 2)
#             if self.training:
#                 update_ema(self.ema, self.ViT_model)
#             # 这里的logits应该要为torch.Size([B, 256, 22, 2])

#         # 输出的logits.shape: torch.Size([B, 2048, 22, 2])
#         return self.output_conv(logits)

#     def generate(self, label, shape=(8, 8), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None,edit = None):
#         """
#         batch_size: 1
#         shape: [22, 2]
#         aud_feat: torch.Size([1, 256, 22, 2])
#         return : the x_predict
#         """
#         param = next(self.parameters())
#         x = torch.zeros(
#             (batch_size, *shape),
#             dtype=torch.int64, device=param.device
#         )
#         # 这里x = 0 是不行的，要用音频作为输入。
#         if edit is not None:
#             x = edit
#         h0 = 0
#         h = shape[0] 

#         # for i in range(h0, h):
#         #     for j in range(shape[1]):
#         #         logits = self.forward(x, label, aud_feat)
#         #         probs = F.softmax(logits[:, :, i, j], -1)
#         #         x.data[:, i, j].copy_(
#         #             probs.multinomial(1).squeeze().data
#         #         )   
            
#         infer_choice_position = [[0] * 2 for _ in range(22)]        # 用来标记选中的位置

#         if pre_latents is not None:
#             x[:, :pre_latents.size(1), :] = pre_latents.squeeze(0)
#             for i in range(pre_latents.size(1)):
#                 for j in range(pre_latents.size(2)):
#                     infer_choice_position[i][j] = 1

#         condition_prob = 0.9
#         for num in range(100):
#             if condition_prob <= -0.2:          # 限制无用循环
#                 break
#             logits = self.forward(x, label, aud_feat)      # 优化若干次
#             Get_flag = False
#             for i in range(h0, h):
#                 for j in range(shape[1]):
#                     if infer_choice_position[i][j] == 1:            #已经选择过，那么跳出
#                         continue
#                     probs = F.softmax(logits[:, :, i, j], -1)       # 计算概率
#                     check_condition = torch.max(probs).item()    

#                     if check_condition >= condition_prob:           # 只有大于一定概率的运动才会被选中
#                         Get_flag = True
#                         if infer_choice_position[i][j] == 0: 
#                             infer_choice_position[i][j] = 1   
#                             x.data[:, i, j].copy_(
#                                 probs.multinomial(1).squeeze().data
#                             )
#             if Get_flag == False:
#                  condition_prob -= 0.1
#         return x[:, h0:h]