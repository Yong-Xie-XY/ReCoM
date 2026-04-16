import os
import sys


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.getcwd())

from tqdm import tqdm
from transformers import Wav2Vec2Processor

from evaluation.FGD import EmbeddingSpaceEvaluator

from evaluation.metrics import LVD

import numpy as np
import smplx as smpl

from data_utils.lower_body import part2full, poses2pred
from data_utils.utils import get_mfcc_ta
from nets import *
from nets.utils import get_path, get_dpath
from trainer.options import parse_args
from data_utils import torch_data
from trainer.config import load_JsonConfig

import torch
from torch.utils import data
from data_utils.get_j import to3d, get_joints
from torch.utils.data import Dataset, DataLoader
import torchaudio
import csv
import datetime

def init_model(model_name, model_path, args, config):
    if model_name == 's2g_face':
        generator = s2g_face(
            args,
            config,
        )
    elif model_name == 's2g_body_vq':
        generator = s2g_body_vq(
            args,
            config,
        )
    elif model_name == 's2g_body_pixel':
        generator = s2g_body_pixel(
            args,
            config,
        )
    elif model_name == 's2g_body_ae':
        generator = s2g_body_ae(
            args,
            config,
        )
    elif model_name == 's2g_LS3DCG':
        generator = LS3DCG(
            args,
            config,
        )
    else:
        raise NotImplementedError

    model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    generator.load_state_dict(model_ckpt['generator'])

    return generator


def init_dataloader(data_root, speakers, args, config):
    data_base = torch_data(
        data_root=data_root,
        speakers=speakers,
        split='test',
        limbscaling=False,
        normalization=config.Data.pose.normalization,
        norm_method=config.Data.pose.norm_method,
        split_trans_zero=False,
        num_pre_frames=config.Data.pose.pre_pose_length,
        num_generate_length=config.Data.pose.generate_length,
        num_frames=30,
        aud_feat_win_size=config.Data.aud.aud_feat_win_size,
        aud_feat_dim=config.Data.aud.aud_feat_dim,
        feat_method=config.Data.aud.feat_method,
        smplx=True,
        audio_sr=22000,
        convert_to_6d=config.Data.pose.convert_to_6d,
        expression=config.Data.pose.expression,
        config=config
    )

    if config.Data.pose.normalization:
        norm_stats_fn = os.path.join(os.path.dirname(args.model_path), "norm_stats.npy")
        norm_stats = np.load(norm_stats_fn, allow_pickle=True)
        data_base.data_mean = norm_stats[0]
        data_base.data_std = norm_stats[1]
    else:
        norm_stats = None

    data_base.get_dataset()     # deal with all_dataset
    test_set = data_base.all_dataset
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False)

    return test_set, test_loader, norm_stats


def body_loss(gt, prs , pred_pose = None , GTposes = None):
    loss_dict = {}
    # gt:([frame, 127, 3]) ; prs:([B, frame, 127, 3]) , 2 denote test batchsize
    # pred_pose([2 , frame, 129]) , GTposes([150, 129])

    # --------------test smothing---------
    # smoth_prs = prs[0]
    # distances = []
    # for i in range(0, gt.shape[0] - 1):
    #     dis_temp = torch.norm(smoth_prs[i] - smoth_prs[i+1], dim=1).mean()
    #     distances.append(dis_temp)
    # mean_distance = sum(distances)/len(distances)
    # variance_distance = sum((x - mean_distance) ** 2 for x in distances) / len(distances)
    # max_distance = max(distances)
    # min_distance = min(distances)
    # loss_dict['mean_distance'] = mean_distance
    # loss_dict['variance_distance'] = variance_distance
    # loss_dict['max_distance'] = max_distance
    # loss_dict['min_distance'] = min_distance
    # --------------test smothing---------
    
    # 22个关于身体的关节点,用了缩放参数
    v_diff = LVD(gt[:, :22, :], prs[:, :, :22, :], symmetrical=False, weight=False)
    loss_dict['LVD'] = v_diff
    # Accuracy
    error = (gt - prs).norm(p=2, dim=-1).sum(dim=-1).mean()
    
    loss_dict['error'] = error
    # Diversity
    var = prs.var(dim=0).norm(p=2, dim=-1).sum(dim=-1).mean()
    # # use
    # gt_var = gt.var(dim=0).norm(p=2, dim=-1).sum(dim=-1).mean()
    # loss_dict['gtdiverse'] = gt_var
    # # dele
    loss_dict['diverse'] = var
    # 
    pred_pose = prs.reshape(prs.shape[0],prs.shape[1],127*3)  # generator number
    #
    pred_pose = torch.mean(pred_pose,dim = 0)       # 求平均，squeeze()，所以不论是B=n还是B=1都会变成(T,Joints)
    temp_frame = pred_pose.shape[0]
    mean_prs = torch.mean(pred_pose,dim = 0)
    for i in range(temp_frame):
        pred_pose[i, :] = abs(pred_pose[i, :] - mean_prs)
    sum_l1 = pred_pose.sum()
    
    loss_dict['L1_div'] = sum_l1/temp_frame

    return loss_dict

def PosetoPredShape(poses):

    # 265 -> 232 , remove lowerpose
    part1 = poses[:, :3]
    part2 = poses[:, 18:21]
    part3 = poses[:, 27:30]
    part4 = poses[:, 36:39]
    part5 = poses[:, 45:]

    poses = torch.cat([part1, part2, part3, part4, part5], dim=1)

    poses = poses[:, 3:132]    
    return poses



def test(test_loader, generator, g_face, FGD_handler, smplx_model, config, args):
    print('start testing')

    am = Wav2Vec2Processor.from_pretrained("wav2wecProcess/wav2vec2-xls-r-300m-phoneme")
    am_sr = 16000


    loss_dict = {}
    
    if config.beat2_test == True:       # 防止显存爆炸
        B = 1
    else: 
        B = 2

    with torch.no_grad():
        count = 0
        for bat in tqdm(test_loader, desc="Testing......"):
            count = count + 1
            # if count == 10:
            #     break
            if config.beat2_test == True:
                # bat : dict_keys(['poses', 'betas', 'audio_file'])
                poses = bat["poses"].to('cuda')       # torch.Size([1, F, 165])
                exp = bat["expressions"].squeeze().to('cuda')    # torch.Size([1, F, 100])
                betas = bat["betas"].to('cuda').to(torch.float64)
                cur_wav_file =  "./ExpressiveWholeBodyDatasetReleaseV1.0/beat_english_v2.0.0/wave16k/" + bat["audio_file"][0]
                id = torch.tensor(0)
                
            
            else: 
                _, poses, exp = bat['aud_feat'].to('cuda').to(torch.float32), bat['poses'].to('cuda').to(torch.float32), \
                                bat['expression'].to('cuda').to(torch.float32)
                # poses : torch.Size([1, 165, frame])
                id = bat['speaker'].to('cuda') - 20
                betas = bat['betas'][0].to('cuda').to(torch.float64)
                poses = torch.cat([poses, exp], dim=-2).transpose(-1, -2) # torch.Size([1, F, 265])
                cur_wav_file = bat['aud_file'][0]


            zero_face = torch.zeros([B, poses.shape[1], 103], device='cuda')
            
            joints_list = []
            # pred_face = g_face.infer_on_audio(cur_wav_file,
            #                           initial_pose=None,
            #                           norm_stats=None,
            #                           w_pre=False,
            #                           # id=id,
            #                           frame=None,
            #                           am=am,
            #                           am_sr=am_sr
            #                           )
    
            # pred_face = torch.tensor(pred_face).squeeze().to('cuda')
            # pred_face = pred_face[:, 3:]

            # extra_kwargs = {'generate_expression': pred_face}
            pred = generator.infer_on_audio(cur_wav_file,
                                            id=id,
                                            fps=30,
                                            B=B,
                                            am=am,
                                            am_sr=am_sr,
                                            frame=poses.shape[0]
                                            )
            pred = torch.tensor(pred, device='cuda')
            
            # pred: torch.Size([B, frame, 129])
            if len(pred.shape) == 2:
                pred = pred.unsqueeze(dim=0)
                
            if  pred.shape[1] > poses.shape[1]:  # balance frame
                pred = pred[:,:poses.shape[1],:]

            # pred:torch.Size([2, 300, 129]) poses: torch.Size([1, 300, 265])
            FGD_handler.push_samples(pred, poses)
            # poses ： torch.Size([f, 265])
            poses = poses.squeeze()
            # poses_joints129 = PosetoPredShape(poses)
            # poses_joints129 ： torch.Size([f, 129])
            if config.beat2_test == False:
                poses = to3d(poses, config)         # unchanged


            if pred.shape[2] > 129:
                pred = pred[:, :, 103:]

            pred_pose = pred
            # print(pred.shape) [2, f, 129]
            pred = torch.cat([zero_face[:, :pred.shape[1], :3], pred, zero_face[:, :pred.shape[1], 3:]], dim=-1)                    # 空下巴 + 身体 + 空脸部
            # print(pred.shape) [2, f, 232])  =  3 + 129 + 100 

            if config.beat2_test == True:
                stand = True
            else:
                stand = False

            full_pred = []
            for j in range(B): 
                f_pred = part2full(pred[j],stand)         # 232 -> 265 , joint-lowerbody
                full_pred.append(f_pred)

            for i in range(full_pred.__len__()):
                full_pred[i] = full_pred[i].unsqueeze(dim=0)
            full_pred = torch.cat(full_pred, dim=0)
            
            pred_joints = get_joints(smplx_model, betas, full_pred)         
            # get_joints results in265 -> 127 x 3, go through the smplx model , so the pred is not joint that can be use for visualization,it is smplx number
            if config.beat2_test == False:
                poses = poses2pred(poses)
                poses = torch.cat([zero_face[0, :, :3], poses[:, 3:165], zero_face[0, :, 3:]], dim=-1)
                gt_joints = get_joints(smplx_model, betas, poses[:pred_joints.shape[1]])
            else:
                global_orient = torch.tensor([3.0747, -0.0158, -0.0152]).repeat(poses.shape[0], 1).to('cuda')
                smplx_model.batch_size = poses.shape[0] # 1 or poses.shape[0] output shape are equal
                gt_joints = smplx_model(
                            betas=betas, 
                            expression=zero_face[0].squeeze()[:,:100],
                            jaw_pose=poses[:, 66:69], 
                            global_orient=global_orient, 
                            body_pose=poses[:,3:21*3+3], 
                            left_hand_pose=poses[:,25*3:40*3], 
                            right_hand_pose=poses[:,40*3:55*3], 
                            leye_pose=poses[:, 69:72], 
                            reye_pose=poses[:, 72:75],
                            return_joints=True, 
                            return_verts=True
                            )['joints']

            
            FGD_handler.push_joints(pred_joints, gt_joints)
            aud = get_mfcc_ta(cur_wav_file, fps=30, sr=16000, am='not None', encoder_choice='onset')
            FGD_handler.push_aud(torch.from_numpy(aud))

            bat_loss_dict = body_loss(gt_joints, pred_joints)

            if loss_dict:  # 非空
                for key in list(bat_loss_dict.keys()):
                    loss_dict[key] += bat_loss_dict[key]
            else:
                for key in list(bat_loss_dict.keys()):
                    loss_dict[key] = bat_loss_dict[key]
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key] / count
            print(key + '=' + str(loss_dict[key].item()))

        
        fgd_dist, feat_dist = FGD_handler.get_scores()
        print('fgd_dist=', fgd_dist.item())
        print('feat_dist=', feat_dist.item())
        BCscore = FGD_handler.get_BCscore()
        print('Beat consistency score=', BCscore)

        # MAAC = FGD_handler.get_MAAC()
        # print(MAAC)
        
        # write metric.txt , # 将输出写入文件  
        with open('body_metric.txt', 'a') as file:  
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            file.write("Time: " + current_time + "\n")
            file.write(args.body_model_path + " : \n{\n")
            for key in loss_dict.keys():
                file.write(key + '=' + str(loss_dict[key].item()) + '\n')
            
            file.write('fgd_dist=' + str(fgd_dist.item()) + '\n')  
            file.write('feat_dist=' + str(feat_dist.item()) + '\n') 
            file.write('Beat consistency score=' + str(BCscore) + '\n}\n\n')  



class Beat2Dataset(Dataset):
    def __init__(self, root_dir, audio_length_file):
        """
        Args:
            root_dir (string): Dataset 的根目录
            audio_length_file (string): 包含音频文件长度的CSV文件路径
        """
        self.root_dir = root_dir
        smplxflame_dir = os.path.join(root_dir, 'smplxflame_30')
        self.audio_dir = os.path.join(root_dir, 'wave16k')
        
        # 加载音频长度数据
        self.audio_lengths = self.load_audio_lengths(audio_length_file)
        
        self.pose_files = sorted([
            os.path.join(smplxflame_dir, f) for f in os.listdir(smplxflame_dir)
            if f.endswith('.npz')
        ])

        self.data_files = []
        for pose_file in self.pose_files:
            audio_filename = os.path.basename(pose_file).replace('.npz', '.wav')
            audio_file = os.path.join(self.audio_dir, audio_filename)

            # 只有存在且时长少于300秒的音频文件才添加
            if audio_file in self.audio_lengths and self.audio_lengths[audio_file] <= 300:
                self.data_files.append((pose_file, audio_file))
            else:
                # print(f"Warning: Audio file {audio_file} does not meet requirements.")
                pass

    def load_audio_lengths(self, audio_length_file):
        lengths = {}
        with open(audio_length_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                file_path, duration = row[0], float(row[1])
                lengths[file_path] = duration
        return lengths

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        pose_file, audio_file_name = self.data_files[idx]

        # Load pose data
        pose_data = np.load(pose_file)['poses']
        betas_data = np.load(pose_file)['betas']
        expression_data = np.load(pose_file)['expressions']
        
        return {
            'poses': torch.tensor(pose_data, dtype=torch.float32),
            'expressions': torch.tensor(expression_data, dtype=torch.float32),
            'betas': torch.tensor(betas_data, dtype=torch.float32),
            'audio_file': os.path.basename(audio_file_name)
        }




def main():
    
    parser = parse_args()
    args = parser.parse_args()
    device = torch.device(args.gpu)
    torch.cuda.set_device(device)
    
    config = load_JsonConfig(args.config_file)

    os.environ['smplx_npz_path'] = config.smplx_npz_path
    os.environ['extra_joint_path'] = config.extra_joint_path
    os.environ['j14_regressor_path'] = config.j14_regressor_path
    
    print('init dataloader...')
    if config.beat2_test == True:
        # test_set, test_loader, norm_stats = init_dataloader(config.Data.data_root, args.speakers, args, config) 

        # test_data = np.load('ExpressiveWholeBodyDatasetReleaseV1.0/beat_english_v2.0.0/smplxflame_30/1_wayne_0_1_1.npz')
        # print(test_data.files)
        # ['betas', 'poses', 'expressions', 'trans', 'model', 'gender', 'mocap_frame_rate']
        # (300,), (F, 165), (F, 100), (F, 3) ,None, None ,None
        # 设置数据集根目录
        dataset_root = 'ExpressiveWholeBodyDatasetReleaseV1.0/beat_english_v2.0.0'
        # we are only use some key, pose and audio_file, not expression
        # 创建数据集实例
        test_set = Beat2Dataset(root_dir=dataset_root, audio_length_file='audio_lengths.csv')
        print("datasets len is",len(test_set))
        print("we does't use time length of audio beyond 300 seconds.")
        # 创建 DataLoader
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False) # dict_keys(['poses', 'audio_file'])

    else :
        # norm_stats is None ; test_set len is 1708 , test_set[i] keys(['poses', 'expression', 'aud_feat', 'speaker', 'aud_file', 'betas'])
        # poses:(165, F) , expression: (100, F) , aud_feat : (64, F) , speaker : int +20 , 
        # aud_file[0] :./ExpressiveWholeBodyDatasetReleaseV1.0/oliver/Tobacco_-_Last_Week_Tonight_with_John_Oliver_HBO-6UsHHOCH4q8.mkv/test/103065-00_08_50-00_08_55/103065-00_08_50-00_08_55.wav 
        # ,betas :(1,300) all is equal
        test_set, test_loader, norm_stats = init_dataloader(config.Data.data_root, args.speakers, args, config) 
    print('init model...')
    model_name = args.body_model_name
    # model_path = get_path(model_name, model_type)
    model_path = args.body_model_path
    generator = init_model(model_name, model_path, args, config)

    # face
    face_model_name = args.face_model_name
    face_model_path = args.face_model_path
    generator_face = init_model(face_model_name, face_model_path, args, config)
    # face

    ae = init_model('s2g_body_ae', './experiments/feature_extractor.pth', args,
                    config)
    FGD_handler = EmbeddingSpaceEvaluator(ae, None, 'cuda')

    print('init smlpx model...')
    dtype = torch.float64
    smplx_path = './visualise/'
    model_params = dict(model_path=smplx_path,
                        model_type='smplx',
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        num_betas=300,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        use_pca=False,
                        flat_hand_mean=False,
                        create_expression=True,
                        num_expression_coeffs=100,
                        num_pca_comps=12,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype, )

    smplx_model = smpl.create(**model_params).to('cuda').eval()

    test(test_loader, generator, generator_face, FGD_handler, smplx_model, config, args)


if __name__ == '__main__':
    main()
