import os
import sys
import platform
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    # os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pickle    


# os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ['DISPLAY'] = ':1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.getcwd())

from transformers import Wav2Vec2Processor
from glob import glob

import numpy as np
import json
import smplx as smpl

from nets import *
from trainer.options import parse_args
from data_utils import torch_data
from trainer.config import load_JsonConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from data_utils.rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle
from data_utils.lower_body import part2full, pred2poses, poses2pred, poses2poses
from visualise.rendering import RenderTool

global device
device = 'cuda'

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
    elif model_name == 's2g_LS3DCG':
        generator = LS3DCG(
            args,
            config,
        )
    else:
        raise NotImplementedError

    model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    if model_name == 'smplx_S2G':
        generator.generator.load_state_dict(model_ckpt['generator']['generator'])

    elif 'generator' in list(model_ckpt.keys()):
        generator.load_state_dict(model_ckpt['generator'])
    else:
        model_ckpt = {'generator': model_ckpt}
        generator.load_state_dict(model_ckpt)

    return generator


def init_dataloader(data_root, speakers, args, config):
    if data_root.endswith('.csv'):
        raise NotImplementedError
    else:
        data_class = torch_data
    if 'smplx' in config.Model.model_name or 's2g' in config.Model.model_name:
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
    else:
        data_base = torch_data(
            data_root=data_root,
            speakers=speakers,
            split='val',
            limbscaling=False,
            normalization=config.Data.pose.normalization,
            norm_method=config.Data.pose.norm_method,
            split_trans_zero=False,
            num_pre_frames=config.Data.pose.pre_pose_length,
            aud_feat_win_size=config.Data.aud.aud_feat_win_size,
            aud_feat_dim=config.Data.aud.aud_feat_dim,
            feat_method=config.Data.aud.feat_method
        )
    if config.Data.pose.normalization:
        norm_stats_fn = os.path.join(os.path.dirname(args.model_path), "norm_stats.npy")
        norm_stats = np.load(norm_stats_fn, allow_pickle=True)
        data_base.data_mean = norm_stats[0]
        data_base.data_std = norm_stats[1]
    else:
        norm_stats = None

    data_base.get_dataset()
    infer_set = data_base.all_dataset
    infer_loader = data.DataLoader(data_base.all_dataset, batch_size=1, shuffle=False)

    return infer_set, infer_loader, norm_stats


def get_vertices(smplx_model, betas, result_list, exp, require_pose=False):
    vertices_list = []
    poses_list = []
    expression = torch.zeros([1, 50])

    for i in result_list:
        vertices = []
        poses = []
        for j in range(i.shape[0]):

            output = smplx_model(betas=betas,
                                 expression=i[j][165:265].unsqueeze_(dim=0) if exp else expression,
                                 jaw_pose=i[j][0:3].unsqueeze_(dim=0),
                                 leye_pose=i[j][3:6].unsqueeze_(dim=0),
                                 reye_pose=i[j][6:9].unsqueeze_(dim=0),
                                 global_orient=i[j][9:12].unsqueeze_(dim=0),
                                 body_pose=i[j][12:75].unsqueeze_(dim=0),
                                 left_hand_pose=i[j][75:120].unsqueeze_(dim=0),
                                 right_hand_pose=i[j][120:165].unsqueeze_(dim=0),
                                 return_verts=True)
            vertices.append(output.vertices.detach().cpu().numpy().squeeze())
            # pose = torch.cat([output.body_pose, output.left_hand_pose, output.right_hand_pose], dim=1)
            pose = output.body_pose
            poses.append(pose.detach().cpu())
        vertices = np.asarray(vertices)
        vertices_list.append(vertices)
        poses = torch.cat(poses, dim=0)
        poses_list.append(poses)
    if require_pose:
        return vertices_list, poses_list
    else:
        return vertices_list, None


global_orient = torch.tensor([3.0747, -0.0158, -0.0152])


def infer(g_body, g_face, smplx_model, rendertool, config, args):
    betas = torch.zeros([1, 300], dtype=torch.float64).to(device)
    am = Wav2Vec2Processor.from_pretrained("wav2wecProcess/wav2vec2-xls-r-300m-phoneme")
    # am = Wav2Vec2Processor.from_pretrained("./wave2wec2-base960h")
    am_sr = 16000
    num_sample = args.num_sample

    cur_wav_file = args.audio_file
    id = args.id
    face = args.only_face
    stand = args.stand
    if face:
        body_static = torch.zeros([1, 162], device=device)
        body_static[:, 6:9] = torch.tensor([3.0747, -0.0158, -0.0152]).reshape(1, 3).repeat(body_static.shape[0], 1)

    result_list = []

    pred_face = g_face.infer_on_audio(cur_wav_file,
                                      initial_pose=None,
                                      norm_stats=None,
                                      w_pre=False,
                                      # id=id,
                                      frame=None,
                                      am=am,
                                      am_sr=am_sr
                                      )
    
    pred_face = torch.tensor(pred_face).squeeze().to(device)                    # torch.Size([len, 103])
    if config.Data.pose.convert_to_6d:
        pred_jaw = pred_face[:, :6].reshape(pred_face.shape[0], -1, 6)
        pred_jaw = matrix_to_axis_angle(rotation_6d_to_matrix(pred_jaw)).reshape(pred_face.shape[0], -1)
        pred_face = pred_face[:, 6:] 
        
    else:
        pred_jaw = pred_face[:, :3]
        # :3选择0~3，3:选择3~（-1）
        pred_face = pred_face[:, 3:]
        # torch.Size([len, 100])
    id = torch.tensor([id], device=device)

    extra_kwargs = {'generate_expression': pred_face}       # facemodel 

    # num_sample可改
    for i in range(num_sample):
        pred_res = g_body.infer_on_audio(cur_wav_file,
                                         initial_pose=None,
                                         norm_stats=None,
                                         txgfile=None,
                                         id=id,
                                         var=None,
                                         fps=30,
                                         w_pre=False,
                                         **extra_kwargs
                                         )
        pred = torch.tensor(pred_res).squeeze().to(device)
        # pred = pred.permute(1,0)
        if pred.shape[0] < pred_face.shape[0]:
            # 处理身体时间长度小于脸部时间长度的情况
            repeat_frame = pred[-1].unsqueeze(dim=0).repeat(pred_face.shape[0] - pred.shape[0], 1)
            pred = torch.cat([pred, repeat_frame], dim=0)
        else:
            # 处理身体时间长度大于脸部时间长度的情况
            pred = pred[:pred_face.shape[0], :]

        # 以上预测的1维度都不会改变，0维度代表时间

        body_or_face = False
        if pred.shape[1] < 275:
            body_or_face = True
        # config.Data.pose.convert_to_6d： False
        if config.Data.pose.convert_to_6d:
            pred = pred.reshape(pred.shape[0], -1, 6)
            pred = matrix_to_axis_angle(rotation_6d_to_matrix(pred))
            pred = pred.reshape(pred.shape[0], -1)
        
        if config.Model.model_name == 's2g_LS3DCG':
            pred = torch.cat([pred[:, :3], pred[:, 103:], pred[:, 3:103]], dim=-1)
        else:
            pred = torch.cat([pred_jaw, pred, pred_face], dim=-1)
            # pred.shape:(time,232) 这里实现了pose关键点连接
            

        # pred[:, 9:12] = global_orient
        # stand表示站立姿势,也就是作者用的demo的样子
        stand = True
        pred = part2full(pred, stand)
        if face:
            pred = torch.cat([pred[:, :3], body_static.repeat(pred.shape[0], 1), pred[:, -100:]], dim=-1)
        # result_list[0] = poses2pred(result_list[0], stand)
        # if gt_0 is None:
        #     gt_0 = gt
        # pred = pred2poses(pred, gt_0)
        # result_list[0] = poses2poses(result_list[0], gt_0)
        visual_test_beat = False
        if visual_test_beat:
            test_path = "./ExpressiveWholeBodyDatasetReleaseV1.0/chemistry/2nd_Order_Rate_Laws-6BZb96mqmbg.mp4/test/70395-00_02_39-00_02_49/70395-00_02_39-00_02_49.pkl"
            with open(test_path, 'rb') as file:
                test_data = pickle.load(file)
            # 现在 test_data 将包含 pkl 文件中的对象
            expression_tensor = torch.tensor(test_data["expression"], dtype=torch.float32).to(device) 
            jaw_pose_tensor = torch.tensor(test_data["jaw_pose"], dtype=torch.float32).to(device) 
            leye_pose_tensor = torch.tensor(test_data["leye_pose"], dtype=torch.float32).to(device) 
            reye_pose_tensor = torch.tensor(test_data["reye_pose"], dtype=torch.float32).to(device) 
            global_orient_tensor = torch.tensor(test_data["global_orient"], dtype=torch.float32).to(device) 
            body_pose_axis_tensor = torch.tensor(test_data["body_pose_axis"], dtype=torch.float32).to(device) 
            left_hand_pose_tensor = torch.tensor(test_data["left_hand_pose"], dtype=torch.float32).to(device) 
            right_hand_pose_tensor = torch.tensor(test_data["right_hand_pose"], dtype=torch.float32).to(device) 
            print(expression_tensor.shape)
            print(jaw_pose_tensor.shape)
            print(leye_pose_tensor.shape)
            print(global_orient_tensor.shape)
            print(body_pose_axis_tensor.shape)
            print(left_hand_pose_tensor.shape)
            print(right_hand_pose_tensor.shape)
            cur_wav_file = "./ExpressiveWholeBodyDatasetReleaseV1.0/chemistry/2nd_Order_Rate_Laws-6BZb96mqmbg.mp4/test/70395-00_02_39-00_02_49/70395-00_02_39-00_02_49.wav"
            test_output = smplx_model(betas=test_data["betas"][0].to(device),
                                 expression=expression_tensor.unsqueeze_(dim=0),
                                 jaw_pose=jaw_pose_tensor.unsqueeze_(dim=0),
                                 leye_pose=leye_pose_tensor.unsqueeze_(dim=0),
                                 reye_pose=reye_pose_tensor.unsqueeze_(dim=0),
                                 global_orient=global_orient_tensor,
                                 body_pose=body_pose_axis_tensor.unsqueeze_(dim=0),
                                 
                                 return_verts=True)
            print(test_output.shape)
            assert False,"断点"
        else:
            result_list.append(pred)

    vertices_list, _ = get_vertices(smplx_model, betas, result_list, config.Data.pose.expression)

    result_list = [res.to('cpu') for res in result_list]
    dict = np.concatenate(result_list[:], axis=0)
    file_name = 'visualise/video/' + config.Log.name + '/' + \
                cur_wav_file.split('\\')[-1].split('.')[-2].split('/')[-1]
    np.save(file_name, dict)

    rendertool._render_sequences(cur_wav_file, vertices_list, stand=stand, face=face, whole_body=args.whole_body)


def main():
    # 改面部和生成器的地方，测试的时候记得修改
    parser = parse_args()
    args = parser.parse_args()
    # device = torch.device(args.gpu)
    # torch.cuda.set_device(device)
    # 设置可见显卡为1张，如果需要更改直接删除即可
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"


    if torch.cuda.device_count() > 1:
        print("cuda数量为：",torch.cuda.device_count())
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda", local_rank)
    else:
        print("cuda数量为：",torch.cuda.device_count())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_JsonConfig(args.config_file)

    face_model_name = args.face_model_name
    face_model_path = args.face_model_path
    body_model_name = args.body_model_name
    if args.talkshow_visual:
        body_model_path = "./experiments/2022-11-02-smplx_S2G-body-pixel-3d/ckpt-99.pth"
    else:
        body_model_path = args.body_model_path
    smplx_path = './visualise/'
    os.environ['smplx_npz_path'] = config.smplx_npz_path
    os.environ['extra_joint_path'] = config.extra_joint_path
    os.environ['j14_regressor_path'] = config.j14_regressor_path

    print('init model...')
    generator = init_model(body_model_name, body_model_path, args, config)
    # generator2 = None
    generator_face = init_model(face_model_name, face_model_path, args, config)

    print('init smlpx model...')
    dtype = torch.float64
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
                        # gender='ne',
                        dtype=dtype, )
    smplx_model = smpl.create(**model_params).to(device)
    print('init rendertool...')
    rendertool = RenderTool('visualise/video/' + config.Log.name)

    infer(generator, generator_face, smplx_model, rendertool, config, args)


if __name__ == '__main__':
    main()
